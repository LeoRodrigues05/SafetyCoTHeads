"""Quality-head attribution.

Identical loop to :mod:`coherency` but the per-head signal is *helpfulness*
on benign instructions (Alpaca / MMLU / GSM8K).  Used to test whether a head
that improves safety also degrades general quality.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from tqdm import tqdm

from ..models import LoadedModel, num_layers_and_heads
from ..models.masks import add_head, empty_mask_cfg, fmt_head_id
from ..utils import now_iso


@dataclass
class QualityConfig:
    mask_qkv: Sequence[str] = ("q",)
    mask_type: str = "scale_mask"
    scale_factor: float = 1e-4
    layers: Optional[tuple[int, int]] = None
    heads: Optional[tuple[int, int]] = None
    top_k: int = 16
    seed: int = 0


def quality_attribution(
    lm: LoadedModel,
    cfg: QualityConfig,
    helpfulness_fn: Callable[[dict | None], float],
) -> dict:
    """``helpfulness_fn(mask_cfg)`` returns a scalar in [0, 5] (judge mean)."""
    n_layers, n_heads, _ = num_layers_and_heads(lm.model)
    base_cfg = empty_mask_cfg(
        mask_qkv=cfg.mask_qkv, mask_type=cfg.mask_type, scale_factor=cfg.scale_factor,
    )
    base = helpfulness_fn(None)
    l_lo, l_hi = cfg.layers or (0, n_layers)
    h_lo, h_hi = cfg.heads or (0, n_heads)
    scores: dict[tuple[int, int], float] = {}
    with tqdm(total=(l_hi - l_lo) * (h_hi - h_lo), desc="quality heads", leave=False) as bar:
        for layer in range(l_lo, l_hi):
            for head in range(h_lo, h_hi):
                trial = add_head(base_cfg, layer, head)
                scores[(layer, head)] = float(base - helpfulness_fn(trial))
                bar.update(1)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:cfg.top_k]
    return {
        "model": lm.name,
        "method": "quality",
        "mask_qkv": list(cfg.mask_qkv),
        "mask_type": cfg.mask_type,
        "scale_factor": cfg.scale_factor,
        "seed": cfg.seed,
        "timestamp": now_iso(),
        "baseline_helpfulness": base,
        "ranked_heads": [
            {"head_id": fmt_head_id(l, h), "layer": l, "head": h, "score": s}
            for (l, h), s in ranked
        ],
        "all_scores": {fmt_head_id(l, h): s for (l, h), s in scores.items()},
    }
