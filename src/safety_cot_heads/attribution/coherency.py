"""Coherency-head attribution.

Three scoring modes are supported via :class:`CoherencyConfig.mode`:

* ``"nll"``               — increase in NLL on benign text (WikiText-103) and
                            instruction-following text (Alpaca) when the head
                            is ablated.  Higher = more important for coherence.
* ``"judge_coherence"``   — drop in a 1-5 judge-coherence score on generated
                            CoT trajectories when the head is ablated.
* ``"pathology"``         — increase in a pathology rate (repetition rate +
                            truncation + malformed output + off-topic) on the
                            generated trajectories.
* ``"hybrid"``            — weighted sum of the three normalised signals.

This module shares the loop structure of SHIPS / Sahara but uses different
per-head scoring functions.  The judge / generation calls are delegated to
:mod:`safety_cot_heads.generation` and :mod:`safety_cot_heads.judging` so the
heavy lifting is done elsewhere.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from tqdm import tqdm

from ..models import LoadedModel, num_layers_and_heads
from ..models.masks import add_head, empty_mask_cfg, fmt_head_id
from ..utils import now_iso


# ---------------------------------------------------------------------------
# NLL scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sequence_nll(lm: LoadedModel, text: str, mask_cfg=None,
                  max_length: int = 512) -> float:
    tok = lm.tokenizer
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(lm.device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return float("nan")
    with lm.head_mask_controller.active(mask_cfg):
        out = lm.model(**enc, labels=input_ids)
    return float(out.loss.item())


def mean_nll(lm: LoadedModel, texts: Sequence[str], mask_cfg=None) -> float:
    losses = [sequence_nll(lm, t, mask_cfg=mask_cfg) for t in texts]
    losses = [l for l in losses if not math.isnan(l)]
    return float(sum(losses) / max(1, len(losses)))


# ---------------------------------------------------------------------------
# Pathology scoring
# ---------------------------------------------------------------------------
def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of length-``n`` token n-grams that are duplicates."""
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    ngrams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def pathology_score(text: str, max_tokens: int) -> float:
    rep = repetition_rate(text)
    trunc = 1.0 if len(text.split()) >= max_tokens else 0.0
    malformed = 1.0 if not text.strip() else 0.0
    return float((rep + trunc + malformed) / 3.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
@dataclass
class CoherencyConfig:
    mode: str = "nll"                           # "nll" | "judge_coherence" | "pathology" | "hybrid"
    mask_qkv: Sequence[str] = ("q",)
    mask_type: str = "scale_mask"
    scale_factor: float = 1e-4
    layers: Optional[tuple[int, int]] = None
    heads: Optional[tuple[int, int]] = None
    top_k: int = 16
    seed: int = 0
    hybrid_weights: dict[str, float] = field(
        default_factory=lambda: {"nll": 1.0, "judge_coherence": 1.0, "pathology": 1.0}
    )


def coherency_attribution(
    lm: LoadedModel,
    cfg: CoherencyConfig,
    *,
    nll_texts: Optional[Sequence[str]] = None,
    judge_fn: Optional[Callable[[dict | None], float]] = None,
    pathology_fn: Optional[Callable[[dict | None], float]] = None,
) -> dict:
    """Return a JSONL-ready row with the ranked coherency heads.

    ``judge_fn`` and ``pathology_fn`` take a ``mask_cfg`` and return a scalar
    score (higher = more important for coherence).  They are expected to
    internally drive generation+judging; this module stays generation-agnostic
    so the same code works for both LLama-2 and Mistral.
    """
    n_layers, n_heads, _ = num_layers_and_heads(lm.model)
    base_cfg = empty_mask_cfg(
        mask_qkv=cfg.mask_qkv, mask_type=cfg.mask_type, scale_factor=cfg.scale_factor,
    )

    # Baselines
    base_nll = (mean_nll(lm, nll_texts) if (cfg.mode in ("nll", "hybrid") and nll_texts)
                else float("nan"))
    base_judge = judge_fn(None) if (cfg.mode in ("judge_coherence", "hybrid") and judge_fn) else float("nan")
    base_path = pathology_fn(None) if (cfg.mode in ("pathology", "hybrid") and pathology_fn) else float("nan")

    l_lo, l_hi = cfg.layers or (0, n_layers)
    h_lo, h_hi = cfg.heads or (0, n_heads)

    scores: dict[tuple[int, int], dict] = {}
    total = (l_hi - l_lo) * (h_hi - h_lo)
    with tqdm(total=total, desc="coherency heads", leave=False) as bar:
        for layer in range(l_lo, l_hi):
            for head in range(h_lo, h_hi):
                trial_cfg = add_head(base_cfg, layer, head)
                row = {}
                if cfg.mode in ("nll", "hybrid") and nll_texts:
                    row["d_nll"] = mean_nll(lm, nll_texts, mask_cfg=trial_cfg) - base_nll
                if cfg.mode in ("judge_coherence", "hybrid") and judge_fn:
                    row["d_judge"] = base_judge - judge_fn(trial_cfg)
                if cfg.mode in ("pathology", "hybrid") and pathology_fn:
                    row["d_pathology"] = pathology_fn(trial_cfg) - base_path
                row["score"] = _combine(row, cfg)
                scores[(layer, head)] = row
                bar.update(1)

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)[:cfg.top_k]
    return {
        "model": lm.name,
        "method": "coherency",
        "mode": cfg.mode,
        "mask_qkv": list(cfg.mask_qkv),
        "mask_type": cfg.mask_type,
        "scale_factor": cfg.scale_factor,
        "seed": cfg.seed,
        "timestamp": now_iso(),
        "baselines": {"nll": base_nll, "judge_coherence": base_judge, "pathology": base_path},
        "ranked_heads": [
            {"head_id": fmt_head_id(l, h), "layer": l, "head": h, **row}
            for (l, h), row in ranked
        ],
        "all_scores": {fmt_head_id(l, h): row for (l, h), row in scores.items()},
    }


def _combine(row: dict, cfg: CoherencyConfig) -> float:
    if cfg.mode == "nll":           return float(row.get("d_nll", 0.0))
    if cfg.mode == "judge_coherence": return float(row.get("d_judge", 0.0))
    if cfg.mode == "pathology":     return float(row.get("d_pathology", 0.0))
    # hybrid: weighted sum
    w = cfg.hybrid_weights
    return float(
        w.get("nll", 1.0) * row.get("d_nll", 0.0)
        + w.get("judge_coherence", 1.0) * row.get("d_judge", 0.0)
        + w.get("pathology", 1.0) * row.get("d_pathology", 0.0)
    )
