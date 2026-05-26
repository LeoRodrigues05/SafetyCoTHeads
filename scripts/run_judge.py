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
    JudgeConfig, aggregate_beavertails, aggregate_safety, judge_rows,
    build_beavertails_fewshot_prefix,
)
from safety_cot_heads.data import load_beavertails_judge_examples
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_read, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _build_fewshot_prefix(cfg) -> str:
    fs = cfg.get("fewshot") or {}
    if not fs or not fs.get("enabled"):
        return ""
    source = fs.get("source", "beavertails")
    if source == "beavertails":
        examples = load_beavertails_judge_examples(
            n_per_category=int(fs.get("n_per_category", 1)),
            seed=int(fs.get("seed", 0)),
            balance_safe=bool(fs.get("balance_safe", True)),
        )
        cap = fs.get("max_examples")
        if cap is not None:
            examples = examples[: int(cap)]
        return build_beavertails_fewshot_prefix(examples)
    raise ValueError(f"unknown fewshot.source {source!r}")


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

    completions_path = Path(args.completions)
    if args.dry_run and not completions_path.exists():
        log.info("DRY-RUN: completions file %s missing; proceeding with 0 rows",
                 completions_path)
        completions: list[dict] = []
    else:
        completions = list(jsonl_read(args.completions))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]
    log.info("judging %d completions from %s", len(completions), args.completions)

    if args.dry_run:
        log.info("DRY-RUN: would load judge %s and process %d rows",
                 cfg.model.name, len(completions))
        plan = cfg_to_dict(cfg)
        # Surface few-shot info in the dryrun trace so it's reviewable without
        # spinning up the model.
        fewshot_prefix = _build_fewshot_prefix(cfg)
        plan["fewshot_prefix_chars"] = len(fewshot_prefix)
        json_dump(out_path.with_suffix(".dryrun.json"),
                  {"plan": plan, "n": len(completions)})
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
        batch_size=int(cfg.get("batch_size", 1)),
        use_chat_template=bool(cfg.get("use_chat_template", True)),
        fewshot_prefix=_build_fewshot_prefix(cfg),
    )
    rows = judge_rows(judge, completions, jcfg)
    jsonl_write(out_path, rows)
    if jcfg.kind == "safety":
        summary = aggregate_safety(rows)
    elif jcfg.kind == "beavertails":
        summary = aggregate_beavertails(rows)
    else:
        summary = {}
    json_dump(out_path.with_suffix(".summary.json"),
              {"judge_model": judge.name, "judge_kind": jcfg.kind,
               "summary": summary, "n_completions": len(rows)})
    log.info("wrote %d judged rows to %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
