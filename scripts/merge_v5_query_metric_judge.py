"""Merge Direction A v5 query+metric judge shards into standard outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402

from safety_cot_heads.analysis.coherence import (  # noqa: E402
    CoherenceConfig,
    classify_gibberish,
    coherence_diagnostics,
)
from safety_cot_heads.direction_a import (  # noqa: E402
    build_cot_only_inputs,
    build_prefix_rows,
    compute_monitorability_gap,
    pathway_vector,
    summarise_pathways,
)
from safety_cot_heads.judging import (  # noqa: E402
    PATHWAY_LABELS,
    merge_pathway_single_label,
    merge_safety_single_label,
)
from safety_cot_heads.judging.judge_prompts import LABELS as SAFETY_LABELS  # noqa: E402
from safety_cot_heads.utils import (  # noqa: E402
    ensure_dir,
    json_dump,
    jsonl_read,
    jsonl_write,
)


ROOT = Path(__file__).resolve().parents[1]


def _completion_file(out_base: Path, dkey: str, cond: str) -> Path | None:
    seed_dir = out_base / "gen" / dkey / cond / "seed0"
    preferred = seed_dir / f"completions_{cond}.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(seed_dir.glob("completions_*.jsonl"))
    return matches[0] if matches else None


def _iter_condition_configs(cfg_dir: Path):
    gen_dir = cfg_dir / "gen"
    for dset_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        dkey = dset_dir.name
        for gcfg in sorted(dset_dir.glob("*.yaml")):
            yield dkey, gcfg.stem


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(jsonl_read(path))


def _collect_shards(shard_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not shard_dir.exists():
        return rows
    for p in sorted(shard_dir.glob("*.jsonl")):
        rows.extend(_read_rows(p))
    rows.sort(key=lambda r: str(r.get("id", "")))
    return rows


def _coherence_rows(completions: list[dict], cfg: dict) -> list[dict]:
    coh_cfg_d = (cfg.get("coherence") or {}) if hasattr(cfg, "get") else {}
    ccfg = CoherenceConfig(
        batch_size=int(coh_cfg_d.get("batch_size", 16)),
        max_length=int(coh_cfg_d.get("max_length", 512)),
        device=coh_cfg_d.get("device", -1),
    )
    texts = [c.get("completion") or "" for c in completions]
    gibb = classify_gibberish(texts, ccfg)
    diag = coherence_diagnostics(texts)
    rows = []
    for c, g, d in zip(completions, gibb, diag):
        rows.append({
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
    return rows


def _aggregate_coherence(coh_rows: list[dict],
                         safety_rows: list[dict]) -> dict:
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
        clean_safety = [s for s in safety_rows if str(s.get("id")) in clean_ids]
        n_cs = len(clean_safety)
        agg["n_clean_judged"] = n_cs
        if n_cs:
            n_harmful_clean = sum(
                1 for s in clean_safety
                if bool(((s.get("judge_flat") or {}).get("labels") or {}).get("harmful_response"))
            )
            agg["harmful_among_clean_rate"] = n_harmful_clean / n_cs
        else:
            agg["harmful_among_clean_rate"] = None
    return agg


def _synthetic_gibberish_safety(src: dict, label: str, judge_model: str) -> dict:
    return {
        "id": src["id"],
        "prompt": src.get("prompt") or src.get("user_prompt") or "",
        "completion": src.get("completion") or "",
        "condition": src.get("condition"),
        "model": src.get("model"),
        "dataset": src.get("dataset"),
        "category": src.get("category"),
        "seed": src.get("seed"),
        "judge_model": judge_model,
        "judge_kind": "safety_single",
        "judge_label": label,
        "judge_flat": {
            "label": label,
            "label_present": False,
            "confidence": 1.0,
            "rationale": "auto: gibberish completion",
        },
        "judge_parse_status": "skipped_gibberish",
        "judge_attempts": [],
        "skipped_gibberish": True,
    }


def _merge_condition(model_key: str, dkey: str, cond: str, cfg,
                     *, force_coherence: bool = False,
                     allow_partial: bool = False) -> dict | None:
    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / model_key
    out_base = ROOT / "runs" / "direction_a_v5" / model_key
    comp_file = _completion_file(out_base, dkey, cond)
    if comp_file is None:
        print(f"skip missing completions: {model_key}/{dkey}/{cond}")
        return None

    completions = list(jsonl_read(comp_file))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]

    out_dir = out_base / "judge" / dkey / cond / "seed0"
    shard_root = out_base / "judge" / "_query_metric_shards" / dkey / cond
    ensure_dir(out_dir)

    coh_path = out_dir / "coherence.jsonl"
    if coh_path.exists() and not force_coherence:
        coherence_rows = _read_rows(coh_path)
    else:
        coherence_rows = _coherence_rows(completions, cfg)
        jsonl_write(coh_path, coherence_rows)
    gibberish_ids = {
        str(r["id"]) for r in coherence_rows
        if bool(cfg.get("gate_safety_by_coherence", False)) and not r.get("is_clean")
    }
    completion_by_id = {str(c["id"]): c for c in completions}
    missing: list[str] = []

    safety_rows: dict[str, list[dict]] = {}
    judge_model = cfg.model.name
    for label in SAFETY_LABELS:
        metric = f"safety_single__{label}"
        rows = _collect_shards(shard_root / metric)
        by_id = {str(r.get("id")): r for r in rows}
        merged_label_rows: list[dict] = []
        for c in completions:
            cid = str(c["id"])
            if cid in gibberish_ids:
                merged_label_rows.append(_synthetic_gibberish_safety(c, label, judge_model))
            elif cid in by_id:
                merged_label_rows.append(by_id[cid])
            else:
                msg = f"safety/{label}/{cid}"
                missing.append(msg)
                print(f"missing safety shard: {model_key}/{dkey}/{cond}/{msg}")
        safety_rows[label] = merged_label_rows
        jsonl_write(out_dir / f"judge_safety__{label}.jsonl", merged_label_rows)

    prefix_rows = build_prefix_rows(completions)
    jsonl_write(out_dir / "prefix_rows.jsonl", prefix_rows)

    pathway_rows: dict[str, list[dict]] = {}
    for label in PATHWAY_LABELS:
        metric = f"pathway_single__{label}"
        rows = _collect_shards(shard_root / metric)
        pathway_rows[label] = rows
        jsonl_write(out_dir / f"judge_pathway__{label}.jsonl", rows)
        expected = len(prefix_rows)
        if len(rows) != expected:
            missing.append(f"pathway/{label}: {len(rows)}/{expected}")
            print(
                f"warning: {model_key}/{dkey}/{cond}/{label}: "
                f"{len(rows)}/{expected} pathway rows"
            )

    cot_rows = _collect_shards(shard_root / "cot_only")
    jsonl_write(out_dir / "judge_cot_only.jsonl", cot_rows)
    expected_cot = len(build_cot_only_inputs(completions))
    if len(cot_rows) != expected_cot:
        missing.append(f"cot_only: {len(cot_rows)}/{expected_cot}")
        print(
            f"warning: {model_key}/{dkey}/{cond}/cot_only: "
            f"{len(cot_rows)}/{expected_cot} rows"
        )

    if missing and not allow_partial:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f", ... +{len(missing) - 10} more"
        raise RuntimeError(
            f"incomplete shards for {model_key}/{dkey}/{cond}: "
            f"{preview}{extra}. Re-run missing shards or pass --allow-partial."
        )

    final_judge_rows = merge_safety_single_label(safety_rows)
    jsonl_write(out_dir / f"judged_{cond}.jsonl", final_judge_rows)

    merged_pathway = merge_pathway_single_label(pathway_rows)
    jsonl_write(out_dir / "judge_pathway.jsonl", merged_pathway)

    summary: dict = {
        "condition": cond,
        "tag": cond,
        "judge_model": judge_model,
        "n_completions": len(completions),
        "n_prefix_rows": len(prefix_rows),
        "split_query_metric": True,
    }
    if merged_pathway:
        vectors = pathway_vector(merged_pathway)
        jsonl_write(out_dir / "pathway_vectors.jsonl", vectors)
        summary["n_pathway_vectors"] = len(vectors)
        summary["per_condition_pathway"] = summarise_pathways(vectors)
    if final_judge_rows and cot_rows:
        gap_rows, gap_summary = compute_monitorability_gap(final_judge_rows, cot_rows)
        jsonl_write(out_dir / "monitorability_rows.jsonl", gap_rows)
        summary["monitorability"] = {
            "n_gap_rows": len(gap_rows),
            "per_condition": gap_summary,
        }
    if final_judge_rows:
        from safety_cot_heads.judging import aggregate_safety
        summary["per_condition_basic"] = aggregate_safety(final_judge_rows)
    if coherence_rows:
        summary["coherence"] = _aggregate_coherence(coherence_rows, final_judge_rows)

    json_dump(out_dir / "summary.json", summary)
    print(f"merged {model_key}/{dkey}/{cond} -> {out_dir}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--force-coherence", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / args.model_key
    cfg = load_cfg(cfg_dir / "judge.yaml")

    summaries: list[dict] = []
    for dkey, cond in _iter_condition_configs(cfg_dir):
        if args.dataset and dkey != args.dataset:
            continue
        if args.condition and cond != args.condition:
            continue
        s = _merge_condition(
            args.model_key,
            dkey,
            cond,
            cfg,
            force_coherence=args.force_coherence,
            allow_partial=args.allow_partial,
        )
        if s is not None:
            summaries.append(s)

    out_base = ROOT / "runs" / "direction_a_v5" / args.model_key / "judge"
    json_dump(
        out_base / "v5_query_metric_merge.summary.json",
        {"model_key": args.model_key, "n_conditions": len(summaries),
         "conditions": summaries},
    )
    print(f"merged {len(summaries)} conditions for {args.model_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
