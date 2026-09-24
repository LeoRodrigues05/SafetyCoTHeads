#!/usr/bin/env python3
"""Export the corrected v6 bundle in the legacy ``composite_cells.csv`` schema.

The paper-figure scripts (``make_narrative_figs``, ``make_composite_insight_reports``,
``plot_pathway_radar``) were written against ``runs/direction_a_v5/composite_cells.csv``.
Rather than port each of them, this writes the same 15-column schema from
``runs/direction_a_v6/reports/cell_metrics.json`` so those scripts can be pointed
at the corrected numbers with a one-line path change.

Cells whose axes are undefined (no coherence-gate-passing response, or no paired
trace/answer) are written with empty fields rather than zeros, so a consumer that
coerces to float will fail loudly instead of silently plotting a fabricated 0.

Usage:
    .venv/bin/python scripts/analysis/export_v6_composite_cells.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs" / "direction_a_v6" / "reports" / "cell_metrics.json"
DST = ROOT / "runs" / "direction_a_v6" / "reports" / "composite_cells.csv"

#: the ten primary conditions of the paper grid (the ONLY cells pooled into
#: family means)
FAMILY = {
    "steering_a0.5": "Steering", "steering_a1.0": "Steering", "steering_a1.5": "Steering",
    "steering_ablate": "Directional ablation",
    "ships_top3": "SHIPS (heads)", "ships_top5": "SHIPS (heads)", "ships_top8": "SHIPS (heads)",
    "neurons_top256": "Neuron", "neurons_top512": "Neuron", "neurons_top1024": "Neuron",
}


def control_family(cond: str):
    """(role, family) for non-primary conditions, written to control_cells.csv."""
    if cond.startswith("rand_heads_"):
        return "random_control", "SHIPS (heads)"
    if cond.startswith("rand_neurons_"):
        return "random_control", "Neuron"
    if cond.startswith("rand_dir_ablate"):
        return "random_control", "Directional ablation"
    if cond.startswith("rand_dir_"):
        return "random_control", "Steering"
    if cond == "steering_ablate_all":
        return "replication", "Directional ablation"
    if cond.startswith("steering_rel_"):
        return "relative_dose", "Steering"
    if cond in ("steering_a0.75", "steering_a1.25"):
        return "dose_refinement", "Steering"
    if cond == "covert_prompt":
        return "monitor_positive_control", "Prompt (control)"
    return None, None
COLS = ["model", "dataset", "condition", "family", "P", "Q", "S", "covert", "raw_hac",
        "clean_rate", "gap", "sfs", "sfs_product", "sfs_covert", "sr_rate"]


def num(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""


def main() -> int:
    payload = json.loads(SRC.read_text())
    rows = payload["rows"]
    sr = {}
    rp = ROOT / "runs" / "direction_a_v6" / "reports" / "reasoning_metrics.json"
    if rp.exists():
        for r in json.loads(rp.read_text())["rows"]:
            sr[(r["model"], r["dataset"], r["condition"])] = r.get("safety_reasoning_rate")

    out, controls = [], []
    for r in rows:
        fam = FAMILY.get(r["condition"])
        if fam is None:
            role, cfam = control_family(r["condition"])
            if role is not None:
                controls.append({
                    "model": r["model"], "dataset": r["dataset"], "condition": r["condition"],
                    "role": role, "family": cfam,
                    "P": num(r.get("P")), "Q": num(r.get("Q")), "S": num(r.get("S_v6")),
                    "sfs": num(r.get("SFS")), "raw_hac": num(r.get("hac")),
                    "clean_rate": num(r.get("clean_rate")),
                    "n_clean": r.get("n_clean"), "n_harmful_clean": r.get("n_harmful_clean"),
                })
            continue
        U, O = r.get("U_covert"), r.get("O_overwarn")
        gap = (U - O) if isinstance(U, (int, float)) and isinstance(O, (int, float)) else None
        sfs = r.get("SFS")
        out.append({
            "model": r["model"], "dataset": r["dataset"], "condition": r["condition"],
            "family": fam,
            "P": num(r.get("P")), "Q": num(r.get("Q")), "S": num(r.get("S_v6")),
            "covert": num(U), "raw_hac": num(r.get("hac")),
            "clean_rate": num(r.get("clean_rate")), "gap": num(gap),
            "sfs": num(sfs),
            "sfs_product": num(sfs ** 3 if isinstance(sfs, (int, float)) else None),
            "sfs_covert": num(sfs),
            "sr_rate": num(sr.get((r["model"], r["dataset"], r["condition"]))),
        })

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    if controls:
        cdst = DST.with_name("control_cells.csv")
        with cdst.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(controls[0].keys()))
            w.writeheader()
            w.writerows(controls)
        print(f"[export] wrote {cdst.relative_to(ROOT)}: {len(controls)} non-primary cells "
              f"(never pooled into primary family means)")
    defined = sum(1 for r in out if r["sfs"])
    print(f"[export] wrote {DST.relative_to(ROOT)}: {len(out)} cells "
          f"({defined} with a defined SFS), answer_source={payload['answer_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
