"""Judge a completions JSONL with the safety / coherence judge.

Usage:
    python -m scripts.run_judge --config configs/experiments/exp02_judge_pipeline/judge.yaml \
        --completions runs/exp03/completions_baseline.jsonl \
        --out         runs/exp03/judge_baseline.jsonl
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

from safety_cot_heads.judging import (
    JudgeConfig, aggregate_safety, judge_rows,
)
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_read, jsonl_write, set_seed,
)

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    completions = list(jsonl_read(args.completions))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]
    log.info("judging %d completions from %s", len(completions), args.completions)

    if args.dry_run:
        log.info("DRY-RUN: would load judge %s and process %d rows",
                 cfg.model.name, len(completions))
        json_dump(out_path.with_suffix(".dryrun.json"),
                  {"plan": cfg_to_dict(cfg), "n": len(completions)})
        return 0

    judge = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    jcfg = JudgeConfig(
        kind=cfg.get("kind", "safety"),
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        base_temperature=float(cfg.get("base_temperature", 0.0)),
        retry_temperature=float(cfg.get("retry_temperature", 0.3)),
        max_retries=int(cfg.get("max_retries", 2)),
        seed=int(cfg.get("seed", 0)),
    )
    rows = judge_rows(judge, completions, jcfg)
    jsonl_write(out_path, rows)
    summary = aggregate_safety(rows) if jcfg.kind == "safety" else {}
    json_dump(out_path.with_suffix(".summary.json"),
              {"judge_model": judge.name, "summary": summary,
               "n_completions": len(rows)})
    log.info("wrote %d judged rows to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
