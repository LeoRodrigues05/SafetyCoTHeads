"""Cross-condition analysis: aggregate judge results, head overlaps, plots.

Usage:
    python -m scripts.run_analysis --config configs/experiments/exp05_joint_disentangled_ablation/00-analysis.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

from safety_cot_heads.analysis import (
    head_grid_heatmap, head_set, overlap_report,
)
from safety_cot_heads.judging import aggregate_safety
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, json_load, jsonl_read,
)

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    out_dir = Path(cfg.output.dir)
    ensure_dir(out_dir)

    # 1. judge aggregation across conditions
    per_cond_summaries = {}
    all_judge_rows = []
    for cond, path in (cfg.get("judge_files") or {}).items():
        rows = list(jsonl_read(path))
        for r in rows:
            r["condition"] = cond
        all_judge_rows.extend(rows)
        per_cond_summaries[cond] = aggregate_safety(rows).get(cond, {})
    if all_judge_rows:
        json_dump(out_dir / "summary_by_condition.json", per_cond_summaries)
        log.info("wrote summary_by_condition.json")

    # 2. head-set overlap
    if cfg.get("attribution_files"):
        named: dict = {}
        for name, spec in cfg.attribution_files.items():
            data = json_load(spec.path)
            if isinstance(data, dict):
                ranked = data.get("ranked_heads") or data.get("dataset_ranking") or []
            else:
                ranked = data
            named[name] = head_set(ranked, top_k=int(spec.get("top_k", 16)))
        report = overlap_report(named)
        json_dump(out_dir / "overlap_report.json", {
            "sizes": report["sizes"],
            "intersections": {k: [list(t) for t in v]
                              for k, v in report["intersections"].items()},
            "jaccard": report["jaccard"],
        })
        log.info("wrote overlap_report.json")

    # 3. heatmaps for each ranked attribution
    if cfg.get("heatmaps"):
        for name, spec in cfg.heatmaps.items():
            d = json_load(spec.path)
            scores = (d.get("dataset_ranking") and
                      {x["head_id"]: x["mean_score"] for x in d["dataset_ranking"]})
            scores = scores or (d.get("all_scores") if isinstance(d, dict) else None)
            if not scores:
                continue
            head_grid_heatmap(
                {str(k): float(v) for k, v in scores.items()},
                n_layers=int(spec.n_layers), n_heads=int(spec.n_heads),
                out_path=out_dir / f"heatmap_{name}.png",
                title=name,
            )

    json_dump(out_dir / "analysis_config.snapshot.json", {"config": cfg_to_dict(cfg)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
