"""Build Direction A v5 query+metric judge task manifests.

The split v5 pipeline judges at the granularity requested for this sweep:
one query and one metric per manifest row. A worker may process several
manifest rows to amortize model-load overhead, but every output shard is still
scoped to exactly one query+metric pair.

Usage:
    python -m scripts.make_v5_judge_manifest \
        --model-key qwen3_8b \
        --out runs/direction_a_v5/qwen3_8b/judge/_manifests/query_metric_tasks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402

from safety_cot_heads.judging.judge_prompts import (  # noqa: E402
    LABELS as SAFETY_LABELS,
    PATHWAY_LABELS,
)
from safety_cot_heads.utils import ensure_dir, jsonl_read  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _metric_rows() -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    for label in SAFETY_LABELS:
        rows.append(("safety_single", label))
    for label in PATHWAY_LABELS:
        rows.append(("pathway_single", label))
    rows.append(("cot_only", None))
    return rows


def _metric_name(kind: str, label: str | None) -> str:
    return kind if label is None else f"{kind}__{label}"


def _short_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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


def build_manifest(model_key: str, out_path: Path) -> int:
    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / model_key
    out_base = ROOT / "runs" / "direction_a_v5" / model_key
    judge_cfg = load_cfg(cfg_dir / "judge.yaml")
    n_limit = judge_cfg.get("n_limit")

    rows: list[dict] = []
    metrics = _metric_rows()
    task_index = 0

    for dkey, cond in _iter_condition_configs(cfg_dir):
        comp_file = _completion_file(out_base, dkey, cond)
        if comp_file is None:
            raise FileNotFoundError(
                f"missing completions for {model_key}/{dkey}/{cond} under "
                f"{out_base / 'gen' / dkey / cond / 'seed0'}"
            )
        completions = list(jsonl_read(comp_file))
        if n_limit is not None:
            completions = completions[: int(n_limit)]

        for query_index, row in enumerate(completions):
            query_id = str(row.get("id", query_index))
            safe_query = f"{query_index:05d}_{_short_id(query_id)}"
            for kind, label in metrics:
                metric = _metric_name(kind, label)
                shard_path = (
                    out_base
                    / "judge"
                    / "_query_metric_shards"
                    / dkey
                    / cond
                    / metric
                    / f"{safe_query}.jsonl"
                )
                rows.append({
                    "task_index": task_index,
                    "model_key": model_key,
                    "dataset": dkey,
                    "condition": cond,
                    "query_index": query_index,
                    "query_id": query_id,
                    "metric_kind": kind,
                    "metric_label": label,
                    "metric": metric,
                    "completions_path": str(comp_file),
                    "shard_path": str(shard_path),
                })
                task_index += 1

    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--print-count-only", action="store_true")
    args = ap.parse_args()

    n = build_manifest(args.model_key, Path(args.out))
    if args.print_count_only:
        print(n)
    else:
        print(f"wrote {n} query+metric tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
