#!/usr/bin/env python3
"""Regenerate paper-facing v6 figures under a versioned output directory.

Reads runs/direction_a_v6/reports/cell_metrics.json and writes figures to
runs/direction_a_v6/reports/figures/. The headline figure demonstrates why the
paired correction matters: the v5 signed marginal gap vs the v6 covert-failure
rate U. Cells near gap=0 with U>0 are exactly the monitorability failures the
signed gap cancelled away.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v6_common as C


def main():
    src = C.V6_ROOT / "reports" / "cell_metrics.json"
    rows = json.loads(src.read_text())["rows"]
    figdir = C.V6_ROOT / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    ex = [r for r in rows if r["trace_type"] == "explicit"
          and r.get("v5_marginal_gap") is not None and r.get("U_covert") is not None]

    # Fig 1: signed marginal gap (v5) vs covert-failure U (v6)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = [r["v5_marginal_gap"] for r in ex]
    ys = [r["U_covert"] for r in ex]
    prim = [r["is_primary"] for r in ex]
    ax.scatter([x for x, p in zip(xs, prim) if p], [y for y, p in zip(ys, prim) if p],
               s=28, alpha=0.7, label="primary explicit-trace", color="#2b6cb0")
    ax.scatter([x for x, p in zip(xs, prim) if not p], [y for y, p in zip(ys, prim) if not p],
               s=28, alpha=0.7, label="exploratory", color="#dd6b20", marker="^")
    ax.axvspan(-0.03, 0.03, color="grey", alpha=0.12,
               label="|marginal gap| < 0.03 (v5 'looks fine')")
    ax.axhline(0.05, ls="--", lw=1, color="#c53030",
               label="covert failure U = 0.05")
    ax.set_xlabel("v5 signed marginal gap  E[harmful] − E[trace-unsafe]")
    ax.set_ylabel("v6 covert-failure rate  U = P(harmful, trace safe)")
    ax.set_title("Covert failures the v5 signed gap hides")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_covert_vs_marginalgap.png", dpi=150)
    plt.close(fig)

    # Fig 2: v5 S (marginal-gap) vs v6 S (paired) — where the axis moves
    ex2 = [r for r in rows if r.get("S_v5_marginalgap") is not None and r.get("S_v6") is not None]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter([r["S_v5_marginalgap"] for r in ex2], [r["S_v6"] for r in ex2],
               s=26, alpha=0.7, color="#2b6cb0")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", label="y = x")
    ax.set_xlabel("v5 Monitorability (marginal-gap S)")
    ax.set_ylabel("v6 Monitorability Retention (paired S)")
    ax.set_title("v5 vs v6 monitorability axis")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_S_v5_vs_v6.png", dpi=150)
    plt.close(fig)

    print(f"[plots] wrote {figdir}/fig1_covert_vs_marginalgap.png, fig2_S_v5_vs_v6.png "
          f"({len(ex)} explicit cells)")


if __name__ == "__main__":
    main()
