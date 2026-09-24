#!/usr/bin/env python3
"""Remove v6 judge outputs whose inputs changed since they were judged.

Judge rows written before input hashing existed (every row before v6.1) carry
no ``input_sha256``, so the judge runners cannot tell whether they are stale.
All of them were judged on the **v6.0 parse of the original v5 completion**.
That reference is re-derived here deterministically -- v6.0 is exactly
``parse_completion(v5_completion, trace_prefilled=False)`` -- and compared with
the CURRENT parsed file (v6.1 parser, continued completions). A legacy row is
pruned iff a field its stage consumed differs. Rows that carry
``input_sha256`` are left alone: the runners' hash-aware resume handles them.

Because the reference is recomputed from the immutable v5 files, this is
idempotent and does not depend on ``parse_changes.json`` (which a repeated
re-parse overwrites; it is kept only as an informational report).

Pruned rows are moved to ``<file>.pruned_<tag>.jsonl`` beside the original;
nothing is deleted.

Usage:
    .venv/bin/python scripts/preprocessing/invalidate_v6_rows.py --models olmo3_7b_think qwen3_8b
    .venv/bin/python scripts/preprocessing/invalidate_v6_rows.py --models olmo3_7b_think --apply
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import sys
from collections import Counter
from pathlib import Path

import v6_common as C  # noqa: E402
from safety_cot_heads.direction_a_v6.parsing import parse_completion  # noqa: E402

#: parsed field -> judge output files that consumed it
CONSUMERS = {
    "answer_text": ("coherence_answer.jsonl", "judge_answer_safety.jsonl"),
    "trace_text": ("judge_cot_only.jsonl", "judge_pathway.jsonl",
                   "judge_safety_reasoning_trace.jsonl"),
    "trace_kind": ("judge_cot_only.jsonl", "judge_cot_only__prefix.jsonl",
                   "judge_pathway.jsonl", "judge_safety_reasoning_trace.jsonl"),
    "prose_prefix_text": ("judge_cot_only__prefix.jsonl",),
}
FIELDS = tuple(CONSUMERS)


def _v60_reference(cell: C.Cell) -> dict[str, dict]:
    """What every legacy judge row saw: the v6.0 parse of the v5 completion."""
    src = sorted(cell.gen_dir().glob("completions*.jsonl"))
    out = {}
    for r in (C.read_jsonl(src[0]) if src else []):
        p = parse_completion(r.get("completion"), trace_prefilled=False).to_dict()
        out[str(r.get("id"))] = {f: p.get(f) for f in FIELDS}
    return out


def changed_fields(cell: C.Cell) -> dict[str, list[str]]:
    ref = _v60_reference(cell)
    cur = {str(r.get("id")): r for r in
           C.read_jsonl(cell.v6_parsed_dir() / "parsed_completions.jsonl")}
    out = {}
    for rid, now in cur.items():
        old = ref.get(rid)
        if old is None:
            continue
        diff = [f for f in FIELDS if (old.get(f) or "") != (now.get(f) or "")]
        if diff:
            out[rid] = diff
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--apply", action="store_true", help="prune (default: dry run)")
    ap.add_argument("--tag", default=None, help="backup suffix (default: UTC date)")
    args = ap.parse_args()
    tag = args.tag or ("invalidate_" + C.utcnow_iso()[:10])

    totals = Counter()
    for cell in C.discover_cells(args.models, args.datasets):
        changed = changed_fields(cell)
        if not changed:
            continue
        totals["rows_changed"] += len(changed)
        by_file: dict[str, set[str]] = {}
        for rid, fields in changed.items():
            for f in fields:
                for name in CONSUMERS[f]:
                    by_file.setdefault(name, set()).add(rid)
        jdir = C.V6_ROOT / "judge" / cell.model / cell.dataset / cell.condition / cell.seed
        for name, ids in sorted(by_file.items()):
            path = jdir / name
            if not path.exists():
                continue
            # only legacy rows (no input hash); hashed rows are resumed by hash
            legacy = {str(r.get("id")) for r in C.read_jsonl(path)
                      if r.get("input_sha256") is None
                      and (str(r.get("id")) in ids
                           or (r.get("parent_id") is not None and str(r["parent_id"]) in ids
                               and "::" in str(r.get("id"))))}
            if not legacy:
                continue
            n = C.prune_rows(path, legacy, tag) if args.apply else len(legacy)
            totals[name] += n
            print(f"  {cell.key:55s} {name:40s} {'pruned' if args.apply else 'would prune'} {n}")
    print(f"[invalidate] {'applied' if args.apply else 'DRY RUN'}: " +
          (", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "nothing to do"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
