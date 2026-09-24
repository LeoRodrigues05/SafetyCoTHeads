"""Regenerate the two narrative figures that had no surviving generator:
``fig_baseline_correction`` (04a) and ``fig_gap_by_family`` (04b).

Both are rebuilt faithfully from the committed v5 composite table
(``runs/direction_a_v5/composite_cells.csv``) so the *content* is identical to
the previous PNGs; only the font (Times-like serif), sizing, and vector-PDF
output change. Single-column canvases (~5 in) so LaTeX barely downscales them.

Usage:
    .venv/bin/python scripts/plotting/make_narrative_figs.py
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
from _paper_figstyle import apply_style, save_fig  # noqa: E402

CSV = ROOT / "runs" / "direction_a_v6" / "reports" / "composite_cells.csv"
FIG_DIR = ROOT / "figures" / "paper"

# The five model/configuration arms in the paper grid (r1_distill is excluded,
# matching the "100 intervention cells = five arms x two datasets x ten
# conditions" statement in the captions).
PAPER_ARMS = ["qwen3_8b", "olmo3_7b_base", "olmo3_7b_base_own",
              "olmo3_7b_think", "llama31_8b_control"]

FAM_ORDER = ["Steering", "Directional ablation", "SHIPS (heads)", "Neuron"]
FAM_COLOR = {"Steering": "#1f77b4", "Directional ablation": "#ff7f0e",
             "SHIPS (heads)": "#2ca02c", "Neuron": "#9467bd"}
FAM_XLABEL = {"Steering": "Steering", "Directional ablation": "Dir.\nablation",
              "SHIPS (heads)": "SHIPS\n(heads)", "Neuron": "Neuron"}

# condition -> row label for the baseline-correction dumbbell
COND_LABEL = {
    "steering_a0.5": "Steering α=0.5", "steering_a1.0": "Steering α=1.0",
    "steering_a1.5": "Steering α=1.5", "steering_ablate": "Directional ablation",
    "ships_top3": "Heads top-3", "ships_top5": "Heads top-5",
    "ships_top8": "Heads top-8", "neurons_top256": "Neurons top-256",
    "neurons_top512": "Neurons top-512", "neurons_top1024": "Neurons top-1024",
}

POTENCY_BLUE = "#1f77b4"
RAW_ORANGE = "#ff7f0e"


def load() -> list[dict]:
    rows = []
    with CSV.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def fig_baseline_correction(rows: list[dict], out_stem: Path) -> None:
    """Dumbbell: raw coherence-gated ASR vs baseline-corrected Potency on the
    pre-alignment OLMo-3-Base checkpoint (JailbreakBench)."""
    cells = [r for r in rows if r["model"] == "olmo3_7b_base"
             and r["dataset"] == "jbb" and r["condition"] != "baseline"]
    # composite_cells.csv stores no baseline row; recover the shared baseline
    # HAC from the Potency definition P = (HAC_c - HAC_b)/(1 - HAC_b), i.e.
    # HAC_b = (HAC_c - P)/(1 - P), using only unclipped cells (0 < P < 1).
    cand = [(float(r["raw_hac"]) - float(r["P"])) / (1.0 - float(r["P"]))
            for r in cells if 0.0 < float(r["P"]) < 0.999]
    base_hac = float(np.median(cand))
    cells.sort(key=lambda r: float(r["P"]))  # smallest at bottom
    ys = np.arange(len(cells))

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for y, r in zip(ys, cells):
        p, hac = float(r["P"]), float(r["raw_hac"])
        ax.plot([p, hac], [y, y], color="0.7", lw=2.2, zorder=1)
    ax.scatter([float(r["raw_hac"]) for r in cells], ys, s=90,
               facecolors="white", edgecolors=RAW_ORANGE, linewidths=2.0,
               zorder=3, label="raw coherence-gated ASR (what papers report)")
    ax.scatter([float(r["P"]) for r in cells], ys, s=90, color=POTENCY_BLUE,
               edgecolors="0.2", linewidths=0.8, zorder=4,
               label="baseline-corrected Potency $P$ (induced harm)")

    top = cells[-1]
    ax.annotate(f"$P$={float(top['P']):.2f}", (float(top["P"]), ys[-1]),
                textcoords="offset points", xytext=(-6, 8), ha="right",
                color=POTENCY_BLUE, fontweight="bold")
    ax.axvline(base_hac, color="0.35", ls="--", lw=1.4, zorder=2)
    ax.text(base_hac + 0.01, ys[-1] + 0.55,
            f"baseline: model already {base_hac*100:.0f}% unsafe",
            color="0.35", va="bottom", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([COND_LABEL[r["condition"]] for r in cells])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("rate")
    ax.xaxis.grid(True, color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False,
              handletextpad=0.4, borderpad=0.2)
    save_fig(fig, out_stem)
    print(f"  wrote {out_stem.name}.{{pdf,png}}")


def fig_gap_by_family(rows: list[dict], out_stem: Path) -> None:
    """Box + jittered strip of the signed monitorability gap by family over the
    100 paper cells (5 arms x 2 datasets x 10 non-baseline conditions)."""
    by_fam = {f: [] for f in FAM_ORDER}
    n_undefined = 0
    for r in rows:
        if r["model"] in PAPER_ARMS and r["condition"] != "baseline" \
                and r["family"] in by_fam:
            # Cells with no coherence-gate-passing trace/answer pair have no gap;
            # they are excluded rather than plotted as 0.
            if r["gap"] == "":
                n_undefined += 1
                continue
            by_fam[r["family"]].append(float(r["gap"]))
    n = sum(len(v) for v in by_fam.values())
    if n_undefined:
        print(f"  [gap] {n_undefined} cells excluded (no defined gap)")

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(FAM_ORDER):
        vals = np.array(by_fam[fam])
        color = FAM_COLOR[fam]
        bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True,
                        showfliers=False, zorder=2,
                        medianprops=dict(color="0.1", lw=1.6),
                        whiskerprops=dict(color=color, lw=1.4),
                        capprops=dict(color=color, lw=1.4),
                        boxprops=dict(edgecolor=color, lw=1.4))
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.18)
        jit = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=26, color=color,
                   alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)

    ax.axhline(0.0, color="#b22222", ls="--", lw=1.4, zorder=1)
    ax.text(-0.35, 0.01, r"covert failure zone ($\gamma > 0$)",
            color="#b22222", va="bottom", ha="left")
    ax.set_xticks(range(len(FAM_ORDER)))
    ax.set_xticklabels([FAM_XLABEL[f] for f in FAM_ORDER])
    ax.set_ylabel(r"gap $\gamma = \mathrm{asr}_{\mathrm{final}} - "
                  r"\mathrm{asr}_{\mathrm{cot}}$")
    ax.yaxis.grid(True, color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.06)
    save_fig(fig, out_stem)
    print(f"  wrote {out_stem.name}.{{pdf,png}} (n={n} cells)")


def main() -> int:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig_baseline_correction(rows, FIG_DIR / "fig_baseline_correction")
    fig_gap_by_family(rows, FIG_DIR / "fig_gap_by_family")
    return 0


if __name__ == "__main__":
    sys.exit(main())
