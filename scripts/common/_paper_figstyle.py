"""Shared paper-figure style.

One place that defines the figure font (a Times-compatible serif matching the
paper's ``\\usepackage{times}``) and a consistent, legible size scheme, plus a
saver that always writes a vector PDF next to the PNG. Import and call
``apply_style()`` before creating any figure; save through ``save_fig()``.

Why a serif + PDF: the paper body is Times, so STIXGeneral keeps figures visually
consistent with it; a vector PDF keeps text crisp at any \\includegraphics scale
(the old PNGs were large canvases downscaled to column width, which is why the
text looked tiny) and avoids Type-3 fonts (``pdf.fonttype=42``), which *ACL
checkers reject.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STIXGeneral is a Times-metric-compatible serif that ships with matplotlib, so
# there is no missing-font fallback; DejaVu Serif is the safety net.
SERIF = ["STIXGeneral", "DejaVu Serif"]


def apply_style(base: float = 12.0) -> None:
    """Apply the consistent paper figure style (call before plotting)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        # embed real (Type-42/TrueType) fonts in PDF/PS, never Type-3
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # consistent, legible size scheme
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.titlesize": base + 2,
        "figure.titleweight": "bold",
        "savefig.dpi": 200,
    })


def save_fig(fig, stem, dpi: int = 200, close: bool = True):
    """Write ``stem.pdf`` and ``stem.png`` (stem may include or omit a suffix)."""
    stem = Path(stem)
    if stem.suffix in (".pdf", ".png"):
        stem = stem.with_suffix("")
    written = []
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    if close:
        plt.close(fig)
    return written
