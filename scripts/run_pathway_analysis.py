"""Direction A v4 — pathway taxonomy + monitorability analysis.

Pipeline:
  1. Read a completions JSONL (output of ``run_generation.py``).
  2. If ``prefix_rows.jsonl`` already exists in ``--out-dir``, reuse it
     (idempotent re-judge over an existing trajectory run).
     Otherwise expand via ``build_prefix_rows``.
  3. Judge every prefix with the **pathway-taxonomy** judge (12 labels).
  4. Build CoT-only judge inputs (trace without final answer) and judge with
     the **cot_only** monitor judge.
  5. Aggregate per parent:
        - 8-dim ``pathway_vector`` + ``dominant_pathway`` (per prereg §6.B);
        - join with a final-answer safety judge to compute the
          monitorability gap (``asr_final - asr_cot_pred``).
  6. Write:
        - ``judge_pathway.jsonl``        (per-prefix pathway judge rows)
        - ``judge_cot_only.jsonl``       (per-completion monitor judge rows)
        - ``pathway_vectors.jsonl``      (8-dim vector per completion)
        - ``monitorability_rows.jsonl``  (per-completion gap rows; only if
                                          ``--final-judge`` is supplied)
        - ``pathway_vectors.summary.json``

Usage:
    python -m scripts.run_pathway_analysis \\
        --config configs/experiments/direction_a_ships/12-pathway-judge.yaml \\
        --completions runs/direction_a/07-trajectory/03-baseline-llama31-jbb/seed0/completions.jsonl \\
        --final-judge runs/direction_a/04-ships-llama31-jbb/judge_safety.jsonl \\
        --out-dir     runs/direction_a/12-pathway/03-baseline-llama31-jbb/seed0/
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg  # noqa: E402

from safety_cot_heads.direction_a import (
    build_cot_only_inputs, build_prefix_rows, compute_monitorability_gap,
    pathway_vector, summarise_pathways,
)
from safety_cot_heads.judging import JudgeConfig, judge_rows
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_read, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _merge_traj_meta(judged: list[dict], prefix_rows: list[dict]) -> list[dict]:
    """Re-attach trajectory metadata that ``judge_rows`` does not propagate."""
    by_id = {r["id"]: r for r in prefix_rows}
    out = []
    for jr in judged:
        meta = by_id.get(jr["id"])
        if meta is None:
            out.append(jr); continue
        merged = dict(jr)
        for k in ("parent_id", "traj_prefix_idx", "traj_prefix_kind",
                  "traj_is_answer", "traj_segments_kind",
                  "traj_n_prose_sentences", "traj_n_think_sentences",
                  "seed"):
            if k in meta and k not in merged:
                merged[k] = meta[k]
        out.append(merged)
    return out


def _jcfg(cfg, *, kind: str) -> JudgeConfig:
    return JudgeConfig(
        kind=kind,
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        base_temperature=float(cfg.get("base_temperature", 0.0)),
        retry_temperature=float(cfg.get("retry_temperature", 0.3)),
        max_retries=int(cfg.get("max_retries", 2)),
        seed=int(cfg.get("seed", 0)),
        batch_size=int(cfg.get("batch_size", 1)),
        use_chat_template=bool(cfg.get("use_chat_template", True)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--final-judge", default=None,
        help="Optional pre-existing safety-judge JSONL over full completions; "
             "if supplied, the monitorability gap is computed.",
    )
    ap.add_argument(
        "--skip-pathway", action="store_true",
        help="Reuse an existing judge_pathway.jsonl in --out-dir; skip the "
             "pathway-judge stage.",
    )
    ap.add_argument(
        "--skip-cot-only", action="store_true",
        help="Reuse an existing judge_cot_only.jsonl in --out-dir; skip the "
             "cot-only judge stage.",
    )
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ---- 1. completions ----------------------------------------------------
    completions_path = Path(args.completions)
    if args.dry_run and not completions_path.exists():
        log.info("DRY-RUN: completions file %s missing; empty list", completions_path)
        completions: list[dict] = []
    else:
        completions = list(jsonl_read(args.completions))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]
    log.info("loaded %d completions from %s", len(completions), completions_path)

    # ---- 2. prefix rows (reuse if present) ---------------------------------
    prefix_jsonl = out_dir / "prefix_rows.jsonl"
    if prefix_jsonl.exists():
        prefix_rows = list(jsonl_read(prefix_jsonl))
        log.info("reusing existing prefix_rows.jsonl (%d rows)", len(prefix_rows))
    else:
        prefix_rows = build_prefix_rows(completions)
        jsonl_write(prefix_jsonl, prefix_rows)
        log.info("expanded %d completions -> %d prefix rows",
                 len(completions), len(prefix_rows))
    seg_kinds = Counter(r.get("traj_segments_kind") for r in prefix_rows)

    # ---- dry-run short-circuit --------------------------------------------
    if args.dry_run:
        plan = cfg_to_dict(cfg)
        plan["n_completions"] = len(completions)
        plan["n_prefix_rows"] = len(prefix_rows)
        plan["seg_kinds"] = dict(seg_kinds)
        plan["final_judge"] = args.final_judge
        json_dump(out_dir / "pathway.dryrun.json", plan)
        log.info("DRY-RUN: wrote plan to %s", out_dir / "pathway.dryrun.json")
        return 0

    # ---- 3. load judge model (shared) -------------------------------------
    judge = None
    if not (args.skip_pathway and args.skip_cot_only):
        judge = load_model(
            cfg.model.name,
            dtype=cfg.model.get("dtype", "auto"),
            load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
            device_map=cfg.model.get("device_map"),
            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        )

    # ---- 4. pathway judge --------------------------------------------------
    pathway_path = out_dir / "judge_pathway.jsonl"
    if args.skip_pathway and pathway_path.exists():
        judged_pathway = list(jsonl_read(pathway_path))
        log.info("reusing existing judge_pathway.jsonl (%d rows)", len(judged_pathway))
    else:
        judged_pathway = judge_rows(judge, prefix_rows, _jcfg(cfg, kind="pathway"))
        judged_pathway = _merge_traj_meta(judged_pathway, prefix_rows)
        jsonl_write(pathway_path, judged_pathway)

    # ---- 5. cot-only monitor judge ----------------------------------------
    cot_only_path = out_dir / "judge_cot_only.jsonl"
    cot_inputs = build_cot_only_inputs(completions)
    log.info("built %d cot-only inputs", len(cot_inputs))
    if args.skip_cot_only and cot_only_path.exists():
        judged_cot = list(jsonl_read(cot_only_path))
        log.info("reusing existing judge_cot_only.jsonl (%d rows)", len(judged_cot))
    else:
        judged_cot = judge_rows(judge, cot_inputs, _jcfg(cfg, kind="cot_only"))
        # parent_id is the completion id for these rows
        for r, src in zip(judged_cot, cot_inputs):
            r.setdefault("parent_id", src.get("parent_id"))
        jsonl_write(cot_only_path, judged_cot)

    # ---- 6. aggregate ------------------------------------------------------
    vectors = pathway_vector(judged_pathway)
    jsonl_write(out_dir / "pathway_vectors.jsonl", vectors)

    summary: dict = {
        "judge_model": getattr(judge, "name", None),
        "n_completions": len(completions),
        "n_prefix_rows": len(prefix_rows),
        "n_pathway_vectors": len(vectors),
        "n_cot_only_judged": len(judged_cot),
        "seg_kinds": dict(seg_kinds),
        "per_condition": summarise_pathways(vectors),
    }

    if args.final_judge:
        final_rows = list(jsonl_read(args.final_judge))
        gap_rows, gap_summary = compute_monitorability_gap(final_rows, judged_cot)
        jsonl_write(out_dir / "monitorability_rows.jsonl", gap_rows)
        summary["monitorability"] = {
            "n_gap_rows": len(gap_rows),
            "per_condition": gap_summary,
        }
        log.info("wrote %d monitorability rows", len(gap_rows))

    json_dump(out_dir / "pathway_vectors.summary.json", summary)
    log.info("wrote %d pathway vectors + summary to %s", len(vectors), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
