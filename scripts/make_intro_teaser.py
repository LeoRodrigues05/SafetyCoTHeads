"""Direction A v5 — introduction teaser: "High ASR is not a diagnosis".

Builds ``papers/.../figures/fig_intro_three_diagnoses.{pdf,png}``: a compact,
single-column (``\\columnwidth``) grouped bar chart over three representative
JailbreakBench cells.  The x-axis walks Raw ASR -> the decomposed Potency /
Quality / Safety-Reasoning axes -> the summary SFS; the three cells are colour
series shaded light->dark by severity (increasing SFS).  The argument is the *shape*:
the Raw ASR cluster is roughly level across all three cells, but the Potency and
SFS clusters fan wide open — output-level attack success cannot, by itself,
distinguish an inherited/degraded failure from an intervention-induced one.

All numbers are read live from the composite cell table so the figure regenerates
whenever the grid changes.

Usage:
    .venv/bin/python -m scripts.make_intro_teaser
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_DEFAULT = ROOT / "runs" / "direction_a_v5" / "composite_cells.csv"
FIG_DIR = ROOT / "papers" / "ARR_Aug_SafetyIntervention" / "figures"

# --- palette (validated dataviz sequential-blue ordinal steps) --------------
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
GRID      = "#dcdbd6"
# three series shaded by severity (increasing SFS): light -> mid -> dark blue
SERIES = ["#8fbaf0", "#3987e5", "#184f95"]

# --- the three representative cells (keyed into the CSV) --------------------
# Legend identifies the actual (model, condition) each cell is drawn from.
# Ordered light->dark by increasing SFS; the diagnosis narrative lives in the
# caption.
CELLS = [
    dict(key=("olmo3_7b_base", "jbb", "neurons_top512"),
         label="OLMo-3 Base\nneurons top-512"),
    dict(key=("olmo3_7b_base_own", "jbb", "neurons_top1024"),
         label="OLMo-3 Base-own\nneurons top-1024"),
    dict(key=("olmo3_7b_think", "jbb", "steering_a1.5"),
         label="OLMo-3 Think\nsteering α=1.5"),
]
# Axis order and names follow the paper: Raw ASR -> (Potency, Quality,
# Safety-Reasoning) -> the summary SFS.
METRICS = [("Raw ASR", "asr"), ("Potency", "P"), ("Quality", "Q"),
           ("Safety-\nReasoning", "S"), ("SFS", "sfs")]


def load_cells(csv_path: Path) -> list[dict]:
    by_key = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            by_key[(row["model"], row["dataset"], row["condition"])] = row
    out = []
    for spec in CELLS:
        row = by_key[spec["key"]]
        out.append({**spec,
                    "asr": float(row["raw_hac"]),
                    "P": float(row["P"]), "Q": float(row["Q"]),
                    "S": float(row["S"]), "sfs": float(row["sfs"])})
    return out


def build(cells: list[dict], out_stem: Path) -> None:
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(3.4, 2.85))
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.135, right=0.985)

    x = np.arange(len(METRICS))
    w = 0.26
    for s, c in enumerate(cells):
        vals = [c[key] for _, key in METRICS]
        ax.bar(x + (s - 1) * w, vals, w, color=SERIES[s],
               edgecolor="white", linewidth=0.5, zorder=3, label=c["label"])
        # label only the two headline groups (Raw ASR, SFS) to keep it clean
        for xi, v in zip(x, vals):
            if xi in (0, len(METRICS) - 1):
                ax.text(xi + (s - 1) * w, v + 0.015, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=4.6, color=INK_MUTED)

    # separate the "decomposition" (P,Q,S) from the two summary metrics
    for xd in (0.5, 3.5):
        ax.axvline(xd, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(2.0, 1.02, "decomposition", ha="center", va="bottom",
            fontsize=5.4, color=INK_MUTED, style="italic")

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in METRICS], fontsize=6.3, linespacing=0.95)
    ax.set_ylabel("Score (0–1)", fontsize=6.8)
    ax.tick_params(length=0, labelsize=6.2)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    fig.suptitle("High ASR is not a diagnosis", fontsize=10.5,
                 fontweight="bold", color=INK, y=0.985)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                    fontsize=5.2, handlelength=0.9, handleheight=1.4,
                    columnspacing=1.0, handletextpad=0.4, borderpad=0.3,
                    frameon=False)
    for t in leg.get_texts():
        t.set_linespacing(1.05)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> int:
    cells = load_cells(CSV_DEFAULT)
    build(cells, FIG_DIR / "fig_intro_three_diagnoses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
