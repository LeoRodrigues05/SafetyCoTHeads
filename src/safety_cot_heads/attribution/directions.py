"""Refusal-direction extraction (Arditi et al. 2024,
"Refusal in LLMs is Mediated by a Single Direction").

At each decoder layer ``l`` we collect the residual-stream activation
``h_l ∈ R^{hidden_size}`` at the **last prompt token** for a set of
harmful and benign prompts. The candidate refusal direction at that layer
is:

    r_l = mean_{harmful} h_l  -  mean_{benign} h_l

Following Arditi et al., the "best" layer is then chosen by held-out
validation — for Llama-3 8B this is consistently a mid-to-late layer
(roughly layer 13–14 of 32). This module saves *all* per-layer directions
to a ``.npz`` so the downstream consumer can pick.

Output: ``.npz`` with keys ``layer_00 … layer_{L-1}``, each a 1-D float32
array of size ``hidden_size``; plus a JSON sidecar with the prompt-set
metadata and the recommended layer (default = middle layer if no
validation is run).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..generation.prompts import render_chat
from ..models import LoadedModel


@dataclass
class DirectionExtractionConfig:
    batch_size: int = 4
    max_length: int = 512
    capture_last_n: int = 1
    system_prompt: Optional[str] = None


def _collect_residual_acts_hook(lm: LoadedModel):
    """Hook each decoder layer to capture its input (= residual stream
    entering that layer). Returns ``(handles, buffers)``."""
    inner = getattr(lm.model, "model", lm.model)
    layers = inner.layers
    buffers: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = []
    for li, layer in enumerate(layers):
        def make_hook(idx=li):
            def pre_hook(_module, args, kwargs):
                h = None
                if args and isinstance(args[0], torch.Tensor):
                    h = args[0]
                elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                    h = kwargs["hidden_states"]
                if h is not None:
                    buffers[idx].append(h.detach().to("cpu", dtype=torch.float32))
                return None
            return pre_hook
        handles.append(layer.register_forward_pre_hook(make_hook(), with_kwargs=True))
    return handles, buffers


@torch.no_grad()
def _mean_residuals(lm: LoadedModel, prompts: Sequence[str],
                    cfg: DirectionExtractionConfig) -> torch.Tensor:
    """Return ``(n_layers, hidden_size)`` mean residual at last token(s)."""
    tok = lm.tokenizer
    inner = getattr(lm.model, "model", lm.model)
    n_layers = len(inner.layers)
    hidden = int(lm.model.config.hidden_size)

    sums = torch.zeros(n_layers, hidden, dtype=torch.float64)
    count = 0
    handles, buffers = _collect_residual_acts_hook(lm)
    try:
        for i in tqdm(range(0, len(prompts), cfg.batch_size),
                      desc="direction.residuals"):
            batch = prompts[i:i + cfg.batch_size]
            rendered = [render_chat(tok, p, system=cfg.system_prompt) for p in batch]
            enc = tok(rendered, return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_length).to(lm.device)
            for buf in buffers:
                buf.clear()
            _ = lm.model(**enc)
            for li, captured in enumerate(buffers):
                assert len(captured) == 1, f"layer {li}: expected 1 fwd, got {len(captured)}"
                act = captured[0]  # (B, T, H)
                last_n = act[:, -cfg.capture_last_n:, :].mean(dim=1)  # (B, H)
                sums[li] += last_n.sum(dim=0).double()
            count += len(batch)
    finally:
        for h in handles:
            h.remove()
    return (sums / max(count, 1)).float()


def compute_refusal_directions(
    lm: LoadedModel,
    harmful_prompts: Sequence[str],
    benign_prompts: Sequence[str],
    cfg: Optional[DirectionExtractionConfig] = None,
) -> dict:
    """Return ``{'directions': np.ndarray(L, H), 'mean_h': ..., 'mean_b': ...}``.

    The ``directions`` array contains one row per layer = mean(harmful) -
    mean(benign), L2 norm preserved (caller normalises in the ablation hook).
    """
    cfg = cfg or DirectionExtractionConfig()
    mh = _mean_residuals(lm, harmful_prompts, cfg)
    mb = _mean_residuals(lm, benign_prompts, cfg)
    diff = (mh - mb).numpy()
    return {
        "directions": diff,
        "mean_harmful": mh.numpy(),
        "mean_benign": mb.numpy(),
        "n_harmful": len(harmful_prompts),
        "n_benign": len(benign_prompts),
        "model": lm.name,
        "default_layer": int(diff.shape[0] // 2),  # middle layer; override via val set
        "citation": (
            "Arditi et al. 2024, 'Refusal in LLMs is Mediated by a Single Direction'. "
            "Direction at each layer = mean(harmful_residual) - mean(benign_residual) "
            "captured at the last prompt token."
        ),
    }


def save_directions_npz(path: str | Path, result: dict) -> None:
    """Save the per-layer directions to a ``.npz`` indexable by layer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {}
    d = result["directions"]
    for li in range(d.shape[0]):
        arrs[f"layer_{li:02d}"] = d[li].astype(np.float32)
    np.savez(path, **arrs)
