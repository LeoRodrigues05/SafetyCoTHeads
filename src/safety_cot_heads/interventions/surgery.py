"""Head surgery — *destructive* weight modification.

Adapted from the original SafetyHeadAttribution surgery experiments which
zeroed the q/k/v columns of selected heads directly in ``q_proj.weight.data``
in place.  That implementation was destructive (no restore) and not GQA-aware.

This version:

* Snapshots the affected projection weights into a `_surgery_snapshots`
  attribute on the model so :func:`undo_surgery` can restore them.
* Is GQA-aware (k/v projections are smaller in Mistral / Llama-3 / Qwen2).
* Supports the same ``mask_qkv`` / ``mask_type`` / ``scale_factor`` semantics
  as the runtime ablation, so a surgery experiment can be reproduced from the
  exact same config.
"""

from __future__ import annotations
from typing import Iterable, Sequence

import torch

from ..models import LoadedModel
from ..models.custom_llama import _collect_attn_layers, num_layers_and_heads
from ..models.masks import parse_head_id


def _slice_for_head(weight: torch.Tensor, head_idx: int, head_dim: int) -> tuple[int, int]:
    """Return the (start, end) row indices in ``weight`` for ``head_idx``."""
    return head_idx * head_dim, (head_idx + 1) * head_dim


def apply_surgery(lm: LoadedModel,
                  heads: Iterable,
                  *,
                  mask_qkv: Sequence[str] = ("q",),
                  mask_type: str = "scale_mask",
                  scale_factor: float = 1e-4) -> int:
    """Destructively scale (or mean-replace) the selected head projections.

    Returns the number of head-edits applied.
    """
    model = lm.model
    n_layers, n_heads, n_kv = num_layers_and_heads(model)
    kv_groups = n_heads // n_kv
    head_dim = model.config.hidden_size // n_heads
    attn_layers = _collect_attn_layers(model)

    if not hasattr(model, "_surgery_snapshots"):
        model._surgery_snapshots = []

    parsed: list[tuple[int, int]] = []
    for h in heads:
        if isinstance(h, str):
            parsed.append(parse_head_id(h))
        elif isinstance(h, dict):
            parsed.append((int(h["layer"]), int(h["head"])))
        else:
            parsed.append((int(h[0]), int(h[1])))

    n_edits = 0
    for (layer, head) in parsed:
        attn = attn_layers[layer]
        for which in mask_qkv:
            if which == "q":
                proj = attn.q_proj
                idx = head
            elif which == "k":
                proj = attn.k_proj
                idx = head // kv_groups
            elif which == "v":
                proj = attn.v_proj
                idx = head // kv_groups
            else:
                raise ValueError(f"unknown mask key {which!r}")

            start, end = _slice_for_head(proj.weight, idx, head_dim)
            snapshot = proj.weight.data[start:end].detach().clone()
            model._surgery_snapshots.append((proj, start, end, snapshot))
            with torch.no_grad():
                if mask_type == "scale_mask":
                    proj.weight.data[start:end].mul_(scale_factor)
                elif mask_type == "mean_mask":
                    proj.weight.data[start:end] = proj.weight.data.mean(dim=0, keepdim=True)
                else:
                    raise ValueError(f"unknown mask_type {mask_type!r}")
            n_edits += 1
    return n_edits


def undo_surgery(lm: LoadedModel) -> int:
    """Restore weights snapshotted by :func:`apply_surgery`. Returns count."""
    model = lm.model
    snaps = getattr(model, "_surgery_snapshots", [])
    with torch.no_grad():
        for proj, start, end, snapshot in reversed(snaps):
            proj.weight.data[start:end] = snapshot.to(proj.weight.device,
                                                        dtype=proj.weight.dtype)
    model._surgery_snapshots = []
    return len(snaps)
