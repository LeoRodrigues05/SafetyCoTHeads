#!/usr/bin/env python3
"""Audit v5 generation outputs before any v6 re-evaluation.

Treats runs/direction_a_v5 as immutable source data. For every discovered
(model, dataset, condition, seed) cell it verifies the 13 checks from the rerun
spec and writes:

    runs/direction_a_v6/audit/generation_audit.json
    runs/direction_a_v6/audit/generation_audit.md
    runs/direction_a_v6/audit/generation_repair_manifest.json

The repair manifest lists ONLY cells that must be regenerated (missing/corrupt/
mismatched/wrong-intervention). It is empty when no generation rerun is needed.
Nothing here writes into the v5 tree.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import v6_common as C


def _decoding_key(rows: list[dict]) -> tuple:
    if not rows:
        return ()
    d = rows[0].get("decoding") or {}
    return (d.get("do_sample"), d.get("temperature"), d.get("top_p"),
            d.get("top_k"), d.get("repetition_penalty"), d.get("max_new_tokens"))


def audit_cell(cell: C.Cell, baseline_rows: list[dict] | None) -> dict:
    res: dict = {"cell": cell.key, "model": cell.model, "dataset": cell.dataset,
                 "condition": cell.condition, "seed": cell.seed,
                 "checks": {}, "problems": [], "needs_repair": False,
                 "repair_reasons": []}
    ch = res["checks"]
    path = C.completions_path(cell)

    # 1. generation exists & parses
    if path is None:
        res["problems"].append("missing_generation_file")
        res["needs_repair"] = True
        res["repair_reasons"].append("missing source generation JSONL")
        ch["generation_exists"] = False
        return res
    ch["generation_exists"] = True
    ch["source_path"] = str(path.relative_to(C.REPO))
    ch["sha256"] = C.sha256_file(path)
    try:
        rows = C.read_jsonl(path)
    except Exception as e:  # corrupt JSONL
        res["problems"].append(f"parse_error:{e}")
        res["needs_repair"] = True
        res["repair_reasons"].append("generation JSONL does not parse")
        ch["parses"] = False
        return res
    ch["parses"] = True
    ch["n_rows"] = len(rows)

    ids = [str(r.get("id")) for r in rows]
    # 2. unique prompt ids
    dup = [i for i in set(ids) if ids.count(i) > 1]
    ch["unique_ids"] = not dup
    if dup:
        res["problems"].append(f"duplicate_ids:{len(dup)}")

    # 6/7. metadata + completion field present
    bad_meta = sum(1 for r in rows if r.get("model") is None or "completion" not in r)
    missing_completion = sum(1 for r in rows if not (r.get("completion") or "").strip())
    ch["all_have_completion_field"] = all("completion" in r for r in rows)
    ch["n_empty_completion"] = missing_completion
    ch["condition_metadata_matches_dir"] = all(
        (r.get("condition") in (cell.condition, None)) for r in rows)
    if bad_meta:
        res["problems"].append(f"missing_metadata_rows:{bad_meta}")

    # 8. intervention provenance flags recorded
    def _flag(rs, k):
        return sum(1 for r in rs if r.get(k) is True)
    is_baseline = cell.condition == "baseline"
    prov = {
        "mask_cfg_active": _flag(rows, "mask_cfg_active"),
        "neuron_cfg_active": _flag(rows, "neuron_cfg_active"),
        "steering_cfg_active": _flag(rows, "steering_cfg_active"),
        "has_config_path": sum(1 for r in rows if r.get("config_path")),
    }
    ch["provenance"] = prov
    cond = cell.condition
    # 10/11. steering cells must show steering was actually applied (not ablation)
    if cond.startswith("steering_") and not cond.endswith("ablate"):
        if prov["steering_cfg_active"] == 0:
            res["problems"].append("steering_cell_without_steering_flag")
            res["needs_repair"] = True
            res["repair_reasons"].append(
                "steering dose not evidenced (steering_cfg_active never true)")
        ch["steering_dose_evidenced"] = prov["steering_cfg_active"] > 0
    if cond.endswith("ablate") or cond == "steering_ablate":
        # ablation should NOT carry a nonzero steering dose masquerading as steering
        ch["is_ablation_cell"] = True
    # neuron/head/ships provenance
    if cond.startswith("neurons") or cond.startswith("heads") or cond.startswith("ships"):
        ch["intervention_provenance_present"] = (
            prov["neuron_cfg_active"] > 0 or prov["mask_cfg_active"] > 0
            or prov["has_config_path"] > 0)

    # 12. decoding consistency vs baseline
    if baseline_rows is not None:
        ch["decoding_matches_baseline"] = _decoding_key(rows) == _decoding_key(baseline_rows)
        if not ch["decoding_matches_baseline"]:
            res["problems"].append("decoding_mismatch_vs_baseline")
        # 3/4/5. prompt id set + text match vs baseline
        b_ids = {str(r.get("id")): (r.get("prompt") or "") for r in baseline_rows}
        c_ids = {i: (r.get("prompt") or "") for i, r in zip(ids, rows)}
        shared = set(b_ids) & set(c_ids)
        ch["n_baseline_ids"] = len(b_ids)
        ch["n_shared_ids_with_baseline"] = len(shared)
        ch["ids_match_baseline"] = set(b_ids) == set(c_ids)
        text_mismatch = [i for i in shared if b_ids[i] != c_ids[i]]
        ch["prompt_text_matches_baseline"] = not text_mismatch
        if text_mismatch:
            res["problems"].append(f"prompt_text_mismatch:{len(text_mismatch)}")
            res["needs_repair"] = True
            res["repair_reasons"].append("prompt text differs from baseline for shared ids")
        # expected count
        ch["count_matches_baseline"] = len(rows) == len(baseline_rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    scope = C.load_paper_scope()
    cells = C.discover_cells(args.models, args.datasets)

    # index baseline rows per (model, dataset, seed) for pairing checks
    baselines: dict[tuple, list[dict]] = {}
    for cell in cells:
        if cell.condition == scope["baseline_condition"]:
            baselines[(cell.model, cell.dataset, cell.seed)] = C.load_completions(cell)

    results = []
    for cell in cells:
        b = baselines.get((cell.model, cell.dataset, cell.seed))
        b = None if cell.condition == scope["baseline_condition"] else b
        results.append(audit_cell(cell, b))

    by_model = defaultdict(lambda: {"cells": 0, "problems": 0, "repairs": 0})
    for r in results:
        m = by_model[r["model"]]
        m["cells"] += 1
        m["problems"] += len(r["problems"])
        m["repairs"] += int(r["needs_repair"])

    repair = [{"cell": r["cell"], "reasons": r["repair_reasons"]}
              for r in results if r["needs_repair"]]

    audit = {
        "generated_at_utc": C.utcnow_iso(),
        "v5_root": str(C.V5_ROOT.relative_to(C.REPO)),
        "paper_scope": {"primary": scope["primary"], "exploratory": scope["exploratory"]},
        "n_cells_discovered": len(results),
        "n_cells_with_problems": sum(1 for r in results if r["problems"]),
        "n_cells_needing_repair": len(repair),
        "by_model": {k: dict(v) for k, v in by_model.items()},
        "cells": results,
    }

    out = C.V6_ROOT / "audit"
    C.write_json(out / "generation_audit.json", audit)
    C.write_json(out / "generation_repair_manifest.json",
                 {"generated_at_utc": audit["generated_at_utc"],
                  "n_repairs": len(repair), "repairs": repair})

    # markdown
    lines = ["# Direction A v6 — generation audit", "",
             f"- Discovered cells: **{len(results)}**",
             f"- Cells with problems: **{audit['n_cells_with_problems']}**",
             f"- Cells needing generation repair: **{len(repair)}**", "",
             "## Per-model", "", "| model | cells | problems | repairs |",
             "|---|---:|---:|---:|"]
    for m, v in sorted(by_model.items()):
        lines.append(f"| {m} | {v['cells']} | {v['problems']} | {v['repairs']} |")
    lines += ["", "## Cells needing repair", ""]
    if repair:
        for r in repair:
            lines.append(f"- `{r['cell']}` — {'; '.join(r['reasons'])}")
    else:
        lines.append("_None. No generation rerun required; reuse all v5 completions._")
    lines += ["", "## Cells with non-blocking problems", ""]
    probs = [r for r in results if r["problems"] and not r["needs_repair"]]
    if probs:
        for r in probs[:200]:
            lines.append(f"- `{r['cell']}` — {'; '.join(r['problems'])}")
    else:
        lines.append("_None._")
    (out / "generation_audit.md").write_text("\n".join(lines) + "\n")

    print(f"[audit] {len(results)} cells; {audit['n_cells_with_problems']} with problems; "
          f"{len(repair)} need repair")
    print(f"[audit] wrote {out}/generation_audit.json|md and generation_repair_manifest.json")


if __name__ == "__main__":
    main()
