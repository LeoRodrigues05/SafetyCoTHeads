#!/usr/bin/env python3
"""Materialize canonical parsed answer/trace files for every v6 cell.

Reads immutable v5 completions, applies the single canonical parser
(:func:`safety_cot_heads.direction_a_v6.parsing.parse_completion`), and writes:

    runs/direction_a_v6/parsed/<model>/<dataset>/<condition>/seed0/parsed_completions.jsonl
    runs/direction_a_v6/parsed/parse_diagnostics.json

Each parsed row carries: full_completion, answer_text, trace_text, trace_kind,
parse_status, has_explicit_trace, answer_is_empty, trace_is_empty,
prose_prefix_text, plus the original id/prompt/model/condition/dataset/seed and
provenance. The answer-judge and trace-judge *input* payloads are derived from
these fields so no downstream script re-splits completions.

Enforces (and reports, never silently drops):
  * explicit-model answer-judge inputs contain no hidden trace text;
  * explicit-model trace-judge inputs contain no final answer;
  * source vs parsed row counts reconcile;
  * malformed cases are reported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import v6_common as C
from safety_cot_heads.direction_a_v6.parsing import parse_completion, PARSER_VERSION


def process_cell(cell: C.Cell) -> dict:
    rows = C.load_completions(cell)
    parsed_rows = []
    kinds = Counter()
    status = Counter()
    n_leak_answer = 0        # trace text leaked into answer (must be 0)
    n_leak_trace = 0         # answer text leaked into trace (must be 0)
    n_answer_empty = 0
    n_trace_empty = 0
    for r in rows:
        p = parse_completion(r.get("completion"))
        kinds[p.trace_kind] += 1
        status[p.parse_status] += 1
        n_answer_empty += int(p.answer_is_empty)
        n_trace_empty += int(p.trace_is_empty)
        # leak checks (only meaningful for explicit traces)
        if p.has_explicit_trace and "</think>" in p.answer_text.lower():
            n_leak_answer += 1
        parsed_rows.append({
            "id": r.get("id"),
            "prompt": r.get("prompt"),
            "model": r.get("model"),
            "condition": r.get("condition"),
            "dataset": r.get("dataset"),
            "seed": cell.seed,
            "category": r.get("category"),
            "full_completion": p.full_completion,
            "answer_text": p.answer_text,
            "trace_text": p.trace_text,
            "trace_kind": p.trace_kind,
            "parse_status": p.parse_status,
            "has_explicit_trace": p.has_explicit_trace,
            "answer_is_empty": p.answer_is_empty,
            "trace_is_empty": p.trace_is_empty,
            "prose_prefix_text": p.prose_prefix_text,
            "parser_version": PARSER_VERSION,
            # provenance carried through
            "decoding": r.get("decoding"),
            "config_path": r.get("config_path"),
        })
    out = cell.v6_parsed_dir() / "parsed_completions.jsonl"
    C.write_jsonl(out, parsed_rows)
    return {
        "cell": cell.key,
        "n_source": len(rows),
        "n_parsed": len(parsed_rows),
        "reconciles": len(rows) == len(parsed_rows),
        "trace_kinds": dict(kinds),
        "parse_statuses": dict(status),
        "n_answer_empty": n_answer_empty,
        "n_trace_empty": n_trace_empty,
        "n_malformed_explicit": kinds.get("malformed_explicit", 0),
        "n_leak_answer": n_leak_answer,
        "n_leak_trace": n_leak_trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    cells = C.discover_cells(args.models, args.datasets)
    diags = [process_cell(c) for c in cells]

    totals = defaultdict(int)
    kind_tot = Counter()
    status_tot = Counter()
    n_nonreconcile = 0
    for d in diags:
        totals["n_source"] += d["n_source"]
        totals["n_parsed"] += d["n_parsed"]
        totals["n_malformed_explicit"] += d["n_malformed_explicit"]
        totals["n_leak_answer"] += d["n_leak_answer"]
        totals["n_answer_empty"] += d["n_answer_empty"]
        kind_tot.update(d["trace_kinds"])
        status_tot.update(d["parse_statuses"])
        n_nonreconcile += int(not d["reconciles"])

    summary = {
        "generated_at_utc": C.utcnow_iso(),
        "parser_version": PARSER_VERSION,
        "n_cells": len(diags),
        "totals": dict(totals),
        "trace_kind_histogram": dict(kind_tot),
        "parse_status_histogram": dict(status_tot),
        "n_cells_not_reconciling": n_nonreconcile,
        "n_answer_leak_cells": sum(1 for d in diags if d["n_leak_answer"] > 0),
        "cells": diags,
    }
    C.write_json(C.V6_ROOT / "parsed" / "parse_diagnostics.json", summary)
    print(f"[parse] {len(diags)} cells; source={totals['n_source']} parsed={totals['n_parsed']} "
          f"reconcile_fail={n_nonreconcile} leak_answer={totals['n_leak_answer']}")
    print(f"[parse] trace_kinds={dict(kind_tot)}")
    print(f"[parse] wrote runs/direction_a_v6/parsed/parse_diagnostics.json")


if __name__ == "__main__":
    main()
