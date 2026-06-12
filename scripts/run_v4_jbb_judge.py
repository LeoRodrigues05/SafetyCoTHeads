"""Direction A v4 — single-label JBB judge orchestrator.

Loads the judge model ONCE and loops over (condition × label) pairs to
produce per-label JSONLs, then merges them into the standard
``judged_<cond>.jsonl`` (5-label safety) and ``judge_pathway.jsonl``
(12-label pathway) shapes that downstream pathway/monitorability code
already consumes.

For each condition the orchestrator emits, under ``--out-base/<tag>/seed<seed>/``:

  judge_safety__<label>.jsonl     5 files (one per safety label)
  judge_pathway__<label>.jsonl    12 files (one per pathway label, over prefix rows)
  judge_cot_only.jsonl            CoT-only monitor judge (single binary)
  prefix_rows.jsonl               cumulative-prefix expansion (reusable)
  coherence.jsonl                 per-completion gibberish label + diagnostics
  judged_<cond>.jsonl             merged 5-label safety shape (final answers)
  judge_pathway.jsonl             merged 12-label pathway shape (per-prefix)
  pathway_vectors.jsonl           8-dim per-completion vector
  monitorability_rows.jsonl       per-completion gap rows
  summary.json                    per-condition aggregates

Idempotent: any per-label file that already exists is skipped (resume-safe).
Conditions are described in a YAML config or inline via `--conditions`
arguments of the form ``tag=<dir>,cond=<name>,completions=<jsonl>``.

Usage:
    python -m scripts.run_v4_jbb_judge \\
        --config configs/experiments/direction_a_ships/19-v4-jbb-qwen3-judge.yaml \\
        --out-base runs/direction_a/19-v4-jbb-qwen3 \\
        --condition tag=03-baseline-llama31-jbb,cond=baseline,\\
completions=runs/direction_a/03-baseline-llama31-jbb/seed0/completions_baseline.jsonl \\
        --condition tag=05-ships-ablation-llama31-jbb,cond=ships_top10,\\
completions=runs/direction_a/05-ships-ablation-llama31-jbb/seed0/completions_ships_top10.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg  # noqa: E402

from safety_cot_heads.direction_a import (
    build_cot_only_inputs, build_prefix_rows, compute_monitorability_gap,
    pathway_vector, summarise_pathways,
)
from safety_cot_heads.judging import (
    JudgeConfig, judge_rows,
    merge_pathway_single_label, merge_safety_single_label,
    PATHWAY_LABELS,
)
from safety_cot_heads.judging.judge_prompts import LABELS as SAFETY_LABELS
from safety_cot_heads.analysis.coherence import (
    CoherenceConfig, classify_gibberish, coherence_diagnostics,
)
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_read, jsonl_write, set_seed,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Condition spec parsing
# ---------------------------------------------------------------------------
def _parse_condition_spec(s: str) -> dict:
    """Parse ``key=value,key=value`` into dict. Required keys: tag, cond, completions."""
    out: dict[str, str] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad spec piece {chunk!r}; expected key=value")
        k, v = chunk.split("=", 1)
        out[k.strip()] = v.strip()
    for req in ("tag", "cond", "completions"):
        if req not in out:
            raise ValueError(f"condition spec missing {req!r}: {s}")
    return out


# ---------------------------------------------------------------------------
# Resume / completion helpers
# ---------------------------------------------------------------------------
def _read_existing_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    repaired = False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    repaired = True
                    log.warning(
                        "malformed JSONL in %s at line %d (%s); "
                        "keeping %d valid rows and regenerating the rest",
                        path, line_no, exc, len(rows),
                    )
                    break
    except Exception as exc:
        repaired = True
        log.warning(
            "could not read existing %s (%s); regenerating from scratch",
            path, exc,
        )
        rows = []
    if repaired:
        jsonl_write(path, rows)
    return rows


def _existing_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for r in _read_existing_jsonl(path):
        rid = r.get("id")
        if rid is not None:
            out.add(str(rid))
    return out


def _filter_unjudged(rows, path: Path):
    seen = _existing_ids(path)
    if not seen:
        return list(rows)
    remaining = [r for r in rows if str(r["id"]) not in seen]
    log.info("resume: %s — %d/%d rows already done, %d remaining",
             path.name, len(seen), len(rows), len(remaining))
    return remaining


# ---------------------------------------------------------------------------
# JudgeConfig factory
# ---------------------------------------------------------------------------
def _jcfg(cfg, *, kind: str, label: str | None = None,
          max_new_tokens: int | None = None) -> JudgeConfig:
    return JudgeConfig(
        kind=kind,
        label=label,
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None
                           else cfg.get("max_new_tokens", 96)),
        base_temperature=float(cfg.get("base_temperature", 0.0)),
        retry_temperature=float(cfg.get("retry_temperature", 0.3)),
        max_retries=int(cfg.get("max_retries", 2)),
        seed=int(cfg.get("seed", 0)),
        batch_size=int(cfg.get("batch_size", 8)),
        use_chat_template=bool(cfg.get("use_chat_template", True)),
    )


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------
def _run_coherence(completions: list[dict], out_path: Path,
                   cfg: dict) -> list[dict]:
    """Run gibberish classifier + model-free diagnostics on completions.

    Resume-safe: if ``out_path`` exists with all ids, returns it as-is.
    Forces ``device=-1`` (CPU) by default so the small classifier does not
    fight the judge model for GPU memory; override via cfg['coherence'].
    """
    seen = _existing_ids(out_path)
    remaining = [c for c in completions if str(c["id"]) not in seen]
    if not remaining:
        log.info("resume: %s already has all %d ids; skipping coherence",
                 out_path.name, len(seen))
        return list(jsonl_read(out_path)) if out_path.exists() else []

    coh_cfg_d = (cfg.get("coherence") or {}) if hasattr(cfg, "get") else {}
    ccfg = CoherenceConfig(
        batch_size=int(coh_cfg_d.get("batch_size", 16)),
        max_length=int(coh_cfg_d.get("max_length", 512)),
        device=coh_cfg_d.get("device", -1),  # default CPU; judge owns GPU
    )
    texts = [c.get("completion") or "" for c in remaining]
    log.info("coherence: classifying %d completions (device=%s)",
             len(texts), ccfg.device)
    gibb = classify_gibberish(texts, ccfg)
    diag = coherence_diagnostics(texts)
    new_rows = []
    for c, g, d in zip(remaining, gibb, diag):
        new_rows.append({
            "id": c["id"],
            "gibberish_label": g["label"],
            "gibberish_score": g["score"],
            "is_clean": g["is_clean"],
            "repeat3": d["repeat3"],
            "compression_ratio": d["compression_ratio"],
            "len_words": d["len_words"],
            "len_chars": d["len_chars"],
            "is_empty": d["is_empty"],
        })
    # append-safe: write all rows back (we already had `seen` filter)
    existing = list(jsonl_read(out_path)) if out_path.exists() else []
    all_rows = existing + new_rows
    jsonl_write(out_path, all_rows)
    return all_rows


def _aggregate_coherence(coh_rows: list[dict],
                          safety_rows: list[dict]) -> dict:
    """Per-condition coherence aggregates + harmful-among-clean rate."""
    n = len(coh_rows)
    if n == 0:
        return {}
    n_clean = sum(1 for r in coh_rows if r.get("is_clean"))
    n_empty = sum(1 for r in coh_rows if r.get("is_empty"))
    hist: dict[str, int] = {}
    for r in coh_rows:
        lab = r.get("gibberish_label") or "unknown"
        hist[lab] = hist.get(lab, 0) + 1
    repeat3 = [float(r.get("repeat3", 0.0)) for r in coh_rows]
    crat = [float(r.get("compression_ratio", 1.0)) for r in coh_rows]
    agg = {
        "n": n,
        "clean_rate": n_clean / n,
        "empty_rate": n_empty / n,
        "gibberish_hist": hist,
        "repeat3_mean": sum(repeat3) / n,
        "compression_ratio_mean": sum(crat) / n,
    }
    if safety_rows:
        clean_ids = {str(r["id"]) for r in coh_rows if r.get("is_clean")}
        clean_safety = [s for s in safety_rows
                        if str(s.get("id")) in clean_ids]
        n_cs = len(clean_safety)
        agg["n_clean_judged"] = n_cs
        if n_cs:
            n_harmful_clean = sum(
                1 for s in clean_safety
                if bool((s.get("judge_flat") or {})
                        .get("labels", {}).get("harmful_response"))
            )
            agg["harmful_among_clean_rate"] = n_harmful_clean / n_cs
        else:
            agg["harmful_among_clean_rate"] = None
    return agg


def _process_condition(judge, cfg, spec: dict, out_dir: Path,
                        skip_safety: bool, skip_pathway: bool,
                        skip_cot_only: bool, skip_coherence: bool) -> dict:
    ensure_dir(out_dir)
    cond_name = spec["cond"]
    completions_path = Path(spec["completions"])
    log.info("=" * 80)
    log.info("condition: tag=%s cond=%s  completions=%s",
             spec["tag"], cond_name, completions_path)
    completions = list(jsonl_read(completions_path))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]

    # ---------- 0. coherence (gibberish + model-free diagnostics) -------
    coherence_rows: list[dict] = []
    if not skip_coherence:
        coh_path = out_dir / "coherence.jsonl"
        coherence_rows = _run_coherence(completions, coh_path, cfg)

    # ---------- 1. safety (full completion) per label --------------------
    safety_rows: dict[str, list[dict]] = {}
    if not skip_safety:
        # Optional gibberish gate: rows the coherence classifier flags as
        # non-clean are NOT sent to the judge — we emit a deterministic
        # `False` for every safety label and mark `skipped_gibberish=True`.
        # This prevents the judge from spuriously marking word-salad as
        # `harmful_response`, which was the failure mode in v4 SHIPS-top10.
        gate = bool(cfg.get("gate_safety_by_coherence", False))
        gibberish_ids: set[str] = set()
        if gate and coherence_rows:
            gibberish_ids = {
                str(r["id"]) for r in coherence_rows
                if not r.get("is_clean")
            }
            if gibberish_ids:
                log.info("gate_safety_by_coherence=true: %d/%d gibberish "
                         "rows will be auto-labeled (harmful=False)",
                         len(gibberish_ids), len(coherence_rows))

        # Build the input shape judge_rows expects: id, prompt, completion.
        safety_inputs = [
            {
                "id": c["id"],
                "prompt": c.get("prompt") or c.get("user_prompt") or "",
                "completion": c.get("completion") or "",
                "condition": c.get("condition") or cond_name,
                "model": c.get("model"),
                "dataset": c.get("dataset"),
                "category": c.get("category"),
                "seed": c.get("seed"),
            }
            for c in completions
        ]
        clean_inputs = [r for r in safety_inputs
                        if str(r["id"]) not in gibberish_ids]
        for label in SAFETY_LABELS:
            path = out_dir / f"judge_safety__{label}.jsonl"
            # only run judge on clean rows
            remaining = _filter_unjudged(clean_inputs, path)
            if remaining:
                judge_rows(
                    judge, remaining,
                    _jcfg(cfg, kind="safety_single", label=label),
                    out_path=str(path),
                )
            judged = list(jsonl_read(path)) if path.exists() else []
            # synthesize deterministic rows for gibberish ids, append-safe
            existing_ids = {str(r.get("id")) for r in judged}
            for gid in gibberish_ids:
                if gid in existing_ids:
                    continue
                src = next((r for r in safety_inputs
                            if str(r["id"]) == gid), None)
                if src is None:
                    continue
                # Match the shape produced by judge_rows(kind=safety_single)
                # so merge_safety_single_label picks it up correctly.
                judged.append({
                    "id": src["id"],
                    "prompt": src["prompt"],
                    "completion": src["completion"],
                    "condition": src.get("condition"),
                    "model": src.get("model"),
                    "dataset": src.get("dataset"),
                    "category": src.get("category"),
                    "seed": src.get("seed"),
                    "judge_kind": "safety_single",
                    "judge_flat": {
                        "label": label,
                        "label_present": False,
                        "confidence": 1.0,
                        "rationale": "auto: gibberish completion",
                    },
                    "judge_parse_status": "skipped_gibberish",
                    "judge_attempts": [],
                    "skipped_gibberish": True,
                })
            safety_rows[label] = judged

    # ---------- 2. pathway (per cumulative prefix) per label -------------
    pathway_rows: dict[str, list[dict]] = {}
    prefix_rows: list[dict] = []
    if not skip_pathway:
        prefix_jsonl = out_dir / "prefix_rows.jsonl"
        if prefix_jsonl.exists():
            prefix_rows = _read_existing_jsonl(prefix_jsonl)
            log.info("reuse prefix_rows.jsonl (%d rows)", len(prefix_rows))
        else:
            prefix_rows = build_prefix_rows(completions)
            jsonl_write(prefix_jsonl, prefix_rows)
            log.info("built %d prefix rows from %d completions",
                     len(prefix_rows), len(completions))
        # judge_rows expects 'completion' field; build_prefix_rows already sets it.
        for label in PATHWAY_LABELS:
            path = out_dir / f"judge_pathway__{label}.jsonl"
            remaining = _filter_unjudged(prefix_rows, path)
            if remaining:
                judge_rows(
                    judge, remaining,
                    _jcfg(cfg, kind="pathway_single", label=label),
                    out_path=str(path),
                )
            pathway_rows[label] = list(jsonl_read(path)) if path.exists() else []

    # ---------- 3. cot-only monitor judge --------------------------------
    cot_only_judged: list[dict] = []
    if not skip_cot_only:
        cot_inputs = build_cot_only_inputs(completions)
        path = out_dir / "judge_cot_only.jsonl"
        # build_cot_only_inputs returns rows with 'response' field; the judge
        # runner reads 'completion'. Re-key.
        cot_inputs_for_judge = [
            {**c, "completion": c.get("response", "")} for c in cot_inputs
        ]
        remaining = _filter_unjudged(cot_inputs_for_judge, path)
        if remaining:
            judge_rows(
                judge, remaining,
                _jcfg(cfg, kind="cot_only", max_new_tokens=128),
                out_path=str(path),
            )
        cot_only_judged = list(jsonl_read(path)) if path.exists() else []
        # propagate parent_id (= source id) onto judged rows for the gap join
        src_by_id = {c["id"]: c for c in cot_inputs}
        for r in cot_only_judged:
            if "parent_id" not in r:
                src = src_by_id.get(r.get("id"))
                if src is not None:
                    r["parent_id"] = src.get("parent_id")

    # ---------- 4. merge per-label -> standard shapes --------------------
    final_judge_rows: list[dict] = []
    if safety_rows:
        final_judge_rows = merge_safety_single_label(safety_rows)
        jsonl_write(out_dir / f"judged_{cond_name}.jsonl", final_judge_rows)
        log.info("merged %d safety judgments -> judged_%s.jsonl",
                 len(final_judge_rows), cond_name)
    merged_pathway: list[dict] = []
    if pathway_rows:
        merged_pathway = merge_pathway_single_label(pathway_rows)
        jsonl_write(out_dir / "judge_pathway.jsonl", merged_pathway)
        log.info("merged %d pathway judgments -> judge_pathway.jsonl",
                 len(merged_pathway))

    # ---------- 5. pathway vectors + monitorability ---------------------
    summary: dict = {
        "condition": cond_name,
        "tag": spec["tag"],
        "judge_model": getattr(judge, "name", None),
        "n_completions": len(completions),
        "n_prefix_rows": len(prefix_rows),
    }
    if merged_pathway:
        vectors = pathway_vector(merged_pathway)
        jsonl_write(out_dir / "pathway_vectors.jsonl", vectors)
        summary["n_pathway_vectors"] = len(vectors)
        summary["per_condition_pathway"] = summarise_pathways(vectors)
    if final_judge_rows and cot_only_judged:
        gap_rows, gap_summary = compute_monitorability_gap(
            final_judge_rows, cot_only_judged,
        )
        jsonl_write(out_dir / "monitorability_rows.jsonl", gap_rows)
        summary["monitorability"] = {
            "n_gap_rows": len(gap_rows),
            "per_condition": gap_summary,
        }
    # legacy 5-metric basic summary (harmful/refusal/etc.)
    if final_judge_rows:
        from safety_cot_heads.judging import aggregate_safety
        summary["per_condition_basic"] = aggregate_safety(final_judge_rows)

    # coherence aggregates (clean_rate, gibberish hist, harmful-among-clean)
    if coherence_rows:
        summary["coherence"] = _aggregate_coherence(
            coherence_rows, final_judge_rows,
        )

    json_dump(out_dir / "summary.json", summary)
    log.info("wrote summary.json for condition %s", cond_name)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Judge model + decoding config (e.g. 19-v4-jbb-qwen3-judge.yaml).")
    ap.add_argument("--out-base", required=True,
                    help="Base output dir; per-condition subdir uses condition tag.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Sub-directory seed index (does not affect judging itself).")
    ap.add_argument("--condition", action="append", default=[], required=True,
                    help="Repeat: tag=<dir>,cond=<name>,completions=<jsonl>")
    ap.add_argument("--skip-safety", action="store_true")
    ap.add_argument("--skip-pathway", action="store_true")
    ap.add_argument("--skip-cot-only", action="store_true")
    ap.add_argument("--skip-coherence", action="store_true")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    specs = [_parse_condition_spec(s) for s in args.condition]
    log.info("=== v4 JBB judge ===  conditions=%d  judge=%s",
             len(specs), cfg.model.name)

    out_base = Path(args.out_base)
    ensure_dir(out_base)

    if args.dry_run:
        plan = {
            "judge": cfg_to_dict(cfg),
            "conditions": [
                {**s, "out_dir": str(out_base / s["tag"] / f"seed{args.seed}")}
                for s in specs
            ],
            "n_safety_labels": len(SAFETY_LABELS),
            "n_pathway_labels": len(PATHWAY_LABELS),
        }
        json_dump(out_base / "v4_jbb_judge.dryrun.json", plan)
        log.info("DRY-RUN: wrote plan to %s", out_base / "v4_jbb_judge.dryrun.json")
        return 0

    # Load judge ONCE.
    judge = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        attach_controllers=False,
    )

    per_condition_summaries: list[dict] = []
    for spec in specs:
        out_dir = out_base / spec["tag"] / f"seed{args.seed}"
        s = _process_condition(
            judge, cfg, spec, out_dir,
            skip_safety=args.skip_safety,
            skip_pathway=args.skip_pathway,
            skip_cot_only=args.skip_cot_only,
            skip_coherence=args.skip_coherence,
        )
        per_condition_summaries.append(s)

    json_dump(
        out_base / "v4_jbb_judge.summary.json",
        {"judge_model": cfg.model.name, "n_conditions": len(specs),
         "conditions": per_condition_summaries},
    )
    log.info("=== done; %d conditions ===", len(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
