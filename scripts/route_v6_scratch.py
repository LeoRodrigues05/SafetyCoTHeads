#!/usr/bin/env python3
"""Route a static-shard scratch file into per-cell judge outputs.

The static shard runner (run_v6_judge_shard.py) pools a shard's rows into a
scratch JSONL keyed by a composite ``<model>/<dataset>/<condition>/<seed>||<id>``
id, and only splits it into per-cell files when the shard *finishes*. If a shard
is stopped early (e.g. to switch to the dynamic dual-GPU runner), that progress
lives only in the scratch file and the dynamic runner — which resumes from
per-cell files — cannot see it.

This tool flushes the scratch into per-cell files NOW, so no judged rows are
lost and the dynamic runner resumes from exactly where the static run stopped.
It is idempotent: per-cell files are merged and de-duplicated by id.

Read-only w.r.t. the live scratch; only writes per-cell outputs (or a temp
``--out-root`` for verification). Run it AFTER stopping the shard so the scratch
is final.

Usage:
  python scripts/route_v6_scratch.py --stage safety-reasoning            # route live
  python scripts/route_v6_scratch.py --stage safety-reasoning --dry-run  # report only
  python scripts/route_v6_scratch.py --stage safety-reasoning \
      --out-root /tmp/verify_judge                                       # route to temp
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import v6_common as C
from run_v6_judge_shard import STAGE_KIND


def _out_name(stage: str, prose_prefix: bool) -> str:
    name = STAGE_KIND[stage][2]
    if prose_prefix and stage in ("monitor", "pathway", "safety-reasoning"):
        name = name.replace(".jsonl", "__prefix.jsonl")
    return name


def find_scratch_files(stage: str, prose_prefix: bool, scratch_dir: Path) -> list[Path]:
    tag = stage + ("_prefix" if prose_prefix else "")
    # names look like: <stage>_gpu<g>of<n>[_prefix].jsonl
    hits = []
    for p in sorted(scratch_dir.glob(f"{stage}_gpu*of*.jsonl")):
        is_prefix = p.stem.endswith("_prefix")
        if prose_prefix and not is_prefix:
            continue
        if not prose_prefix and is_prefix:
            continue
        hits.append(p)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["answer", "monitor", "pathway", "safety-reasoning"])
    ap.add_argument("--prose-prefix", action="store_true")
    ap.add_argument("--scratch-dir", default=str(C.V6_ROOT / "judge" / "_shard_scratch"))
    ap.add_argument("--out-root", default=str(C.V6_ROOT / "judge"),
                    help="root under which per-cell files are written "
                         "(<out-root>/<model>/<dataset>/<condition>/<seed>/<out_name>)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scratch_dir = Path(args.scratch_dir)
    out_root = Path(args.out_root)
    out_name = _out_name(args.stage, args.prose_prefix)

    files = find_scratch_files(args.stage, args.prose_prefix, scratch_dir)
    if not files:
        print(f"[route] no scratch files for stage={args.stage} "
              f"prose_prefix={args.prose_prefix} in {scratch_dir}")
        return

    # group scratch rows by cell.key, restoring the original id (last wins)
    by_cell: dict[str, dict[str, dict]] = defaultdict(dict)
    n_scratch = 0
    n_bad = 0
    for f in files:
        for jr in C.iter_jsonl(f):
            n_scratch += 1
            cid = str(jr.get("id", ""))
            cellkey, sep, orig = cid.partition("||")
            if not sep:
                n_bad += 1
                continue
            row = dict(jr)
            row["id"] = orig
            by_cell[cellkey][orig] = row

    total_rows = sum(len(v) for v in by_cell.values())
    print(f"[route] stage={args.stage} scratch_files={len(files)} scratch_rows={n_scratch} "
          f"cells={len(by_cell)} unique_rows={total_rows} malformed_ids={n_bad} "
          f"out_name={out_name} dry_run={args.dry_run}")

    if args.dry_run:
        for ck in list(by_cell)[:5]:
            print(f"   {ck}: {len(by_cell[ck])} rows")
        print(f"[route] dry-run: would write {len(by_cell)} per-cell files ({total_rows} rows)")
        return

    n_written = 0
    for cellkey, rows_by_id in by_cell.items():
        parts = cellkey.split("/")
        if len(parts) != 4:
            print(f"[route] WARNING skipping malformed cell key: {cellkey!r}")
            continue
        model, dataset, condition, seed = parts
        out_path = out_root / model / dataset / condition / seed / out_name
        # merge with any existing per-cell rows, dedup by id
        merged = {str(x["id"]): x for x in (C.read_jsonl(out_path) if out_path.exists() else [])}
        merged.update(rows_by_id)
        C.write_jsonl(out_path, list(merged.values()))
        n_written += len(rows_by_id)
    print(f"[route] wrote/merged {total_rows} rows into {len(by_cell)} per-cell files "
          f"under {out_root}")


if __name__ == "__main__":
    main()
