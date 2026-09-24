#!/usr/bin/env python3
"""Rename pathway files that were NOT produced by the validated protocol.

Every v6 ``judge_pathway.jsonl`` written before 2026-09-24 came from the
zero-shot Qwen3-30B on the multi-label prompt. The v6.1 queue re-judges the
paper's cells with the validated 14B single-label protocol (``--stage
pathway14b``). Cells outside that scope (e.g. the defence-side Exp-5 cells)
keep the old file; this script renames it to
``judge_pathway__30b_multilabel.jsonl`` so no cell carries an unmarked,
unvalidated ``judge_pathway.jsonl`` and the release stays coherent. Nothing is
deleted. Cells already holding 14B single-label labels are left alone.

Usage:
  .venv/bin/python scripts/preprocessing/quarantine_legacy_pathway.py \
      --models qwen3_8b olmo3_7b_think --keep-conditions baseline ships_top3 ...   # dry run
  ... --apply
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import json

import v6_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=["jbb", "bt"])
    ap.add_argument("--keep-conditions", nargs="*", default=[],
                    help="conditions re-judged with the validated protocol (never touched)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    keep = set(args.keep_conditions)
    n = 0
    for cell in C.discover_cells(args.models, args.datasets):
        p = C.V6_ROOT / "judge" / cell.model / cell.dataset / cell.condition / cell.seed / "judge_pathway.jsonl"
        if cell.condition in keep or not p.exists():
            continue
        with open(p) as f:
            first = json.loads(f.readline() or "{}")
        if first.get("judge_kind_source") == "single_label_merge":
            continue                                   # already validated-protocol labels
        dst = p.with_name("judge_pathway__30b_multilabel.jsonl")
        if dst.exists():
            print(f"  [skip] {cell.key}: {dst.name} already exists")
            continue
        print(f"  {'rename' if args.apply else 'would rename'} {cell.key}/judge_pathway.jsonl -> {dst.name}")
        if args.apply:
            p.replace(dst)
        n += 1
    print(f"[quarantine] {'renamed' if args.apply else 'would rename'} {n} legacy pathway files")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
