"""Direction A trajectory analysis: judge sentence prefixes -> 7-dim vectors.

Pipeline:
  1. Read a completions JSONL (output of ``run_generation.py``).
  2. Segment each completion into cumulative prefixes (prose or R1 ``<think>``).
  3. Judge every prefix with the safety judge.
  4. Compute the 7 trajectory metrics per parent generation.
  5. Write ``judge_prefixes.jsonl`` (raw per-prefix judge rows) and
     ``trajectory_vectors.jsonl`` (one vector per parent), plus a
     ``trajectory_vectors.summary.json`` aggregate.

Usage:
    python -m scripts.run_trajectory \
        --config configs/experiments/direction_a_ships/07-trajectory-judge.yaml \
        --completions runs/direction_a/04-ships-llama31-jbb/completions_ships_top10.jsonl \
        --out-dir     runs/direction_a/04-ships-llama31-jbb/trajectory_ships_top10/
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg  # noqa: E402

from safety_cot_heads.direction_a import (
    build_prefix_rows, trajectory_vector, METRIC_FIELDS,
)
from safety_cot_heads.judging import JudgeConfig, judge_rows
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_read, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _merge_traj_meta(judged: list[dict], prefix_rows: list[dict]) -> list[dict]:
    """``judge_rows`` strips fields it doesn't recognise; re-attach trajectory
    metadata by id."""
    by_id = {r["id"]: r for r in prefix_rows}
    out = []
    for jr in judged:
        meta = by_id.get(jr["id"])
        if meta is None:
            out.append(jr)
            continue
        merged = dict(jr)
        for k in ("parent_id", "traj_prefix_idx", "traj_prefix_kind",
                  "traj_is_answer", "traj_segments_kind",
                  "traj_n_prose_sentences", "traj_n_think_sentences",
                  "seed"):
            if k in meta and k not in merged:
                merged[k] = meta[k]
        out.append(merged)
    return out


def _aggregate(vectors: list[dict]) -> dict:
    """Per-condition mean of each metric + parse-status counters."""
    by_cond: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "sums": {f: 0.0 for f in METRIC_FIELDS},
    })
    for v in vectors:
        cond = v.get("condition") or "unknown"
        slot = by_cond[cond]
        slot["n"] += 1
        for f in METRIC_FIELDS:
            slot["sums"][f] += float(v.get(f, 0.0))
    out = {}
    for cond, slot in by_cond.items():
        n = max(slot["n"], 1)
        out[cond] = {"n": slot["n"], **{f: slot["sums"][f] / n for f in METRIC_FIELDS}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    completions_path = Path(args.completions)
    if args.dry_run and not completions_path.exists():
        log.info("DRY-RUN: completions file %s missing; using empty list",
                 completions_path)
        completions: list[dict] = []
    else:
        completions = list(jsonl_read(args.completions))
    n_limit = cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]
    log.info("loaded %d completions from %s", len(completions), completions_path)

    prefix_rows = build_prefix_rows(completions)
    seg_kinds = Counter(r["traj_segments_kind"] for r in prefix_rows)
    log.info("expanded to %d prefix rows (segment kinds: %s)",
             len(prefix_rows), dict(seg_kinds))

    prefix_jsonl = out_dir / "prefix_rows.jsonl"
    jsonl_write(prefix_jsonl, prefix_rows)

    if args.dry_run:
        plan = cfg_to_dict(cfg)
        plan["n_completions"] = len(completions)
        plan["n_prefix_rows"] = len(prefix_rows)
        plan["seg_kinds"] = dict(seg_kinds)
        json_dump(out_dir / "trajectory.dryrun.json", plan)
        log.info("DRY-RUN: wrote plan to %s", out_dir / "trajectory.dryrun.json")
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
    )
    judged = judge_rows(judge, prefix_rows, jcfg)
    judged = _merge_traj_meta(judged, prefix_rows)
    jsonl_write(out_dir / "judge_prefixes.jsonl", judged)

    vectors = trajectory_vector(judged)
    jsonl_write(out_dir / "trajectory_vectors.jsonl", vectors)

    summary = {
        "judge_model": judge.name,
        "n_completions": len(completions),
        "n_prefix_rows": len(prefix_rows),
        "n_trajectory_vectors": len(vectors),
        "seg_kinds": dict(seg_kinds),
        "per_condition": _aggregate(vectors),
    }
    json_dump(out_dir / "trajectory_vectors.summary.json", summary)
    log.info("wrote %d trajectory vectors to %s", len(vectors), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
