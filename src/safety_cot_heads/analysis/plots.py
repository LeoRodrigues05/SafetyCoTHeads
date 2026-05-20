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
