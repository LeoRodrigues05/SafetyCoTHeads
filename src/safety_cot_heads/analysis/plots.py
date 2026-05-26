"""Plotting helpers (matplotlib + seaborn).

All functions take a JSONL/list-of-dicts input and a target path, write a
PNG, and return the path.  No interactive use.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

from ..utils import ensure_dir


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid")
    return plt


def head_grid_heatmap(scores: dict[str, float],
                       n_layers: int, n_heads: int,
                       out_path: str | Path,
                       title: str = "Per-head score") -> Path:
    """``scores`` is ``{"<layer>-<head>": value}``."""
    import numpy as np
    plt = _setup()
    grid = np.full((n_layers, n_heads), float("nan"))
    for k, v in scores.items():
        l, h = k.split("-")
        grid[int(l), int(h)] = float(v)
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.25),
                                     max(4, n_layers * 0.25)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def dose_response_plot(curves: dict[str, Iterable[dict]],
                        out_path: str | Path,
                        ylabel: str = "harmful_rate") -> Path:
    plt = _setup()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        xs = [c["k"] for c in curve]
        ys = [c["score"] for c in curve]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("# heads ablated")
    ax.set_ylabel(ylabel)
    ax.legend()
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def trajectory_flip_histogram(flip_indices: Iterable[int | None],
                               out_path: str | Path) -> Path:
    plt = _setup()
    vals = [i for i in flip_indices if i is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    if vals:
        ax.hist(vals, bins=range(0, max(vals) + 2))
    ax.set_xlabel("sentence index of first HARMFUL flip")
    ax.set_ylabel("# trajectories")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def condition_score_bar(scores_by_cond: dict[str, float],
                         out_path: str | Path,
                         ylabel: str = "mean score",
                         title: str = "") -> Path:
    """Single-axis bar chart of per-condition scalar scores (mean coherence,
    mean malicious_intent, harmful_rate, ...)."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(scores_by_cond)), 4))
    conds = list(scores_by_cond.keys())
    vals = [float(scores_by_cond[c]) if scores_by_cond[c] is not None else 0.0
            for c in conds]
    ax.bar(conds, vals, color="#4c72b0")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def per_category_grouped_bar(by_cond_by_cat: dict[str, dict[str, float]],
                              out_path: str | Path,
                              ylabel: str = "mean malicious_intent",
                              title: str = "") -> Path:
    """Grouped bar chart: one cluster per category, one bar per condition.
    ``by_cond_by_cat[condition][category] = score``.
    """
    import numpy as np
    plt = _setup()
    conds = list(by_cond_by_cat.keys())
    cats = sorted({c for d in by_cond_by_cat.values() for c in d})
    x = np.arange(len(cats))
    w = max(0.1, 0.8 / max(1, len(conds)))
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cats)), 4.5))
    for i, cond in enumerate(conds):
        vals = [float(by_cond_by_cat[cond].get(c) or 0.0) for c in cats]
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, vals, w, label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def paired_delta_bar(deltas_by_cond: dict[str, float],
                     out_path: str | Path,
                     ylabel: str = "Δ vs baseline",
                     title: str = "") -> Path:
    """Bar chart of signed deltas per condition (e.g. mean malicious_intent shift)."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(deltas_by_cond)), 4))
    conds = list(deltas_by_cond.keys())
    vals = [float(deltas_by_cond[c]) if deltas_by_cond[c] is not None else 0.0
            for c in conds]
    colors = ["#c44e52" if v > 0 else "#55a868" for v in vals]
    ax.bar(conds, vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    out = Path(out_path)
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
