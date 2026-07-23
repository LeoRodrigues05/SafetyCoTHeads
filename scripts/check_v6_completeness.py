#!/usr/bin/env python3
"""Completeness checker for the Direction A v6 rerun.

Reconciles what SHOULD exist (expected cells from the paper scope + discovered
baselines) against what does exist on disk, and reports:

  * expected vs discovered cells;
  * missing source generation files;
  * duplicate / mismatched prompt IDs;
  * incomplete judge outputs (v5 reused + v6 produced);
  * parser failures;
  * missing metric blocks;
  * overlap / gaps between the two GPU shards;
  * cells excluded from explicit-trace metrics (prose / exploratory);
  * cells requiring generation repair.

Read-only. Exit code is non-zero if any *blocking* problem is found (missing
source generation, duplicate ids, parser reconcile failure, shard overlap/gap),
so the runner's `check` stage can gate the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import v6_common as C
from safety_cot_heads.direction_a_v6.sharding import assign_shards, verify_partition


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--stage", default="all",
                    help="which judge stage's outputs to require: parse|answer|monitor|"
                         "pathway|safety-reasoning|all (default all reports whatever exists)")
    args = ap.parse_args()

    scope = C.load_paper_scope()
    explicit = set(scope["explicit_trace_models"])
    primary = set(scope["primary"])
    cells = C.discover_cells(args.models, args.datasets)

    report: dict = {"generated_at_utc": C.utcnow_iso(),
                    "n_discovered_cells": len(cells), "blocking": [], "warnings": [],
                    "sections": {}}

    # --- source generation presence & id integrity ------------------------
    missing_src, dup_ids, empty_completion = [], [], []
    for c in cells:
        p = C.completions_path(c)
        if p is None:
            missing_src.append(c.key)
            continue
        rows = C.load_completions(c)
        ids = [str(r.get("id")) for r in rows]
        if len(ids) != len(set(ids)):
            dup_ids.append(c.key)
        if any(not (r.get("completion") or "").strip() for r in rows):
            empty_completion.append(c.key)
    report["sections"]["source_generation"] = {
        "missing": missing_src, "duplicate_prompt_ids": dup_ids,
        "cells_with_empty_completions": empty_completion}
    if missing_src:
        report["blocking"].append(f"{len(missing_src)} cells missing source generation")
    if dup_ids:
        report["blocking"].append(f"{len(dup_ids)} cells with duplicate prompt ids")

    # --- audit / repair reconciliation ------------------------------------
    audit = _load_json(C.V6_ROOT / "audit" / "generation_audit.json")
    repair = _load_json(C.V6_ROOT / "audit" / "generation_repair_manifest.json")
    report["sections"]["audit"] = {
        "audit_present": audit is not None,
        "n_cells_needing_repair": (repair or {}).get("n_repairs", 0),
        "repairs": (repair or {}).get("repairs", []),
    }
    if (repair or {}).get("n_repairs", 0):
        report["warnings"].append(f"{repair['n_repairs']} cells flagged for generation repair")

    # --- parser diagnostics ----------------------------------------------
    pdiag = _load_json(C.V6_ROOT / "parsed" / "parse_diagnostics.json")
    if pdiag is None:
        report["warnings"].append("parse_diagnostics.json missing — run `parse` stage")
    else:
        nonrec = pdiag.get("n_cells_not_reconciling", 0)
        leak = pdiag.get("n_answer_leak_cells", 0)
        report["sections"]["parser"] = {
            "n_cells": pdiag.get("n_cells"),
            "n_cells_not_reconciling": nonrec,
            "n_answer_leak_cells": leak,
            "trace_kind_histogram": pdiag.get("trace_kind_histogram"),
            "n_malformed_explicit": pdiag.get("totals", {}).get("n_malformed_explicit"),
        }
        if nonrec:
            report["blocking"].append(f"{nonrec} cells: parsed rows do not reconcile with source")
        if leak:
            report["blocking"].append(f"{leak} cells: trace leaked into answer input")

    # --- judge output completeness (v5 reused) ----------------------------
    incomplete_answer, incomplete_cot, missing_summary = [], [], []
    for c in cells:
        jd = c.judge_dir()
        n_src = len(C.load_completions(c))
        judged = sorted(jd.glob("judged_*.jsonl"))
        if not judged:
            incomplete_answer.append(c.key)
        cot = jd / "judge_cot_only.jsonl"
        # only explicit-trace models are required to carry monitor labels
        if c.model in explicit and not cot.exists():
            incomplete_cot.append(c.key)
        if not (jd / "summary.json").exists():
            missing_summary.append(c.key)
    report["sections"]["judge_outputs_v5"] = {
        "cells_without_answer_judge": len(incomplete_answer),
        "cells_without_cot_only_judge(explicit)": len(incomplete_cot),
        "cells_without_summary_block": len(missing_summary),
        "note": "missing v5 judge outputs are expected for un-judged cells and are the "
                "work items for the B200 answer/monitor/pathway/safety-reasoning stages.",
    }

    # --- explicit-trace exclusions ---------------------------------------
    excluded = [c.key for c in cells if c.model not in explicit or c.model not in primary]
    report["sections"]["explicit_trace_scope"] = {
        "primary_explicit_models": sorted(explicit & primary),
        "n_cells_excluded_from_primary_explicit": len(excluded),
        "reason": "prose-only (llama) and exploratory (r1) cells are excluded from the "
                  "primary explicit-trace monitorability aggregate.",
    }

    # --- shard partition integrity ---------------------------------------
    task_ids = [c.key for c in cells]
    shards = assign_shards(task_ids, 2)
    part = verify_partition(task_ids, shards)
    report["sections"]["sharding"] = {
        "n_tasks": part["n_tasks"], "shard0": len(shards[0]), "shard1": len(shards[1]),
        "ok": part["ok"], "overlap": part["overlap"][:20], "missing": part["missing"][:20]}
    if not part["ok"]:
        report["blocking"].append("shard partition invalid (overlap or gap)")

    report["ok"] = not report["blocking"]
    C.write_json(C.V6_ROOT / "audit" / "completeness_report.json", report)

    print(f"[check] discovered={len(cells)} shard0={len(shards[0])} shard1={len(shards[1])} "
          f"blocking={len(report['blocking'])} warnings={len(report['warnings'])}")
    for b in report["blocking"]:
        print(f"  BLOCKING: {b}")
    for w in report["warnings"]:
        print(f"  warning:  {w}")
    print(f"[check] wrote runs/direction_a_v6/audit/completeness_report.json")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
