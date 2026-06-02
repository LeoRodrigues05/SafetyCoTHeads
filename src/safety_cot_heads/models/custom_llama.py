"""Hook-based Q/K/V head masking for Llama-family attention.

This is a **re-implementation** of the masking semantics from the original
SafetyHeadAttribution repo (``lib/utils/custommodel.py::CustomLlamaAttention``)
using ``torch`` forward hooks instead of subclassing
``LlamaAttention.forward``.  Subclassing breaks across ``transformers``
versions (and the original wrapper was pinned to a specific HF release);
hooks attach to the existing ``q_proj`` / ``k_proj`` / ``o_proj`` linear
layers and so work for any Llama / Mistral / Qwen2 style decoder with
standard naming.

Semantics preserved verbatim from the SHIPS code:

* ``mask_type='scale_mask'``: target slice is multiplied by ``scale_factor``
  (default ``0.0`` if not supplied).
* ``mask_type='mean_mask'``: target slice is replaced by the per-token mean
  across heads.
* ``mask_qkv`` for a given (layer, head) controls which of Q / K / V is masked.
* For grouped-query attention (Mistral, Llama-3), the K/V head index is
  derived as ``kv_head = head_idx // num_key_value_groups`` — matching the
  spec note that K/V masking must target the KV-head index.

The corresponding **weight-surgery** path (``mask_para=True`` in the original
repo) is exposed in :mod:`safety_cot_heads.interventions.surgery` and operates
by editing ``q/k/v_proj.weight`` on a *cloned* model (the original repo did
this destructively on the live model — fixed here).
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
from torch import nn


def _scale_or_mean_inplace(
    tensor: torch.Tensor,
    head_idx: int,
    head_dim: int,
    mask_type: str,
    scale_factor: float | None,
) -> torch.Tensor:
    """Apply scale-mask or mean-mask to the (head_idx)-th head slice of a
    tensor of shape ``(..., num_heads * head_dim)`` along the last dim.

    Returns a *new* tensor (out-of-place) so we don't mutate activations
    that may have ``requires_grad`` set.
    """
    start = head_idx * head_dim
    end = start + head_dim
    new = tensor.clone()
    if mask_type == "scale_mask":
        s = 0.0 if scale_factor is None else float(scale_factor)
        new[..., start:end] = new[..., start:end] * s
    elif mask_type == "mean_mask":
        # Mean across heads, per token.  Tensor view: (..., num_heads, head_dim).
        lead = tensor.shape[:-1]
        num_heads = tensor.shape[-1] // head_dim
        as_heads = tensor.view(*lead, num_heads, head_dim)
        mean_per_token = as_heads.mean(dim=-2, keepdim=False)   # (..., head_dim)
        new_view = new.view(*lead, num_heads, head_dim)
        new_view[..., head_idx, :] = mean_per_token
        new = new_view.view(*lead, num_heads * head_dim)
    else:
        raise ValueError(f"unknown mask_type={mask_type!r}; expected scale_mask|mean_mask")
    return new


class HeadMaskController:
    """Stateful controller registering hooks on every attention layer.

    Usage::

        ctrl = HeadMaskController.attach(model)
        with ctrl.active(mask_cfg):                  # mask_cfg as in masks.py
            outputs = model(**inputs)
        ctrl.detach()

    The same controller can be re-used with different ``mask_cfg``s.
    """

    def __init__(self, model: nn.Module, attn_layers: list[nn.Module]):
        self.model = model
        self.attn_layers = attn_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cfg: Mapping | None = None
        # Phase-window gating (Direction A v4 §4.4 / prereg §9).
        # ``_token_idx`` is the *current generated-token index* (caller-set);
        # ``_cfg['phase_window'] = (start, end)`` gates mask application to
        # generated-token indices in [start, end). When unset, masks always
        # apply (back-compat). Prompt forward pass is conventionally index -1.
        self._token_idx: int = -1
        # Cached per-layer geometry.
        self._head_dim: list[int] = []
        self._num_attention_heads: list[int] = []
        self._num_kv_heads: list[int] = []
        for attn in attn_layers:
            # Try several attribute names for cross-version robustness.
            head_dim = (
                getattr(attn, "head_dim", None)
                or getattr(getattr(attn, "config", None), "head_dim", None)
                or (attn.q_proj.out_features // getattr(attn, "num_heads", 1))
            )
            num_heads = getattr(attn, "num_heads", None) or attn.q_proj.out_features // head_dim
            num_kv = (
                getattr(attn, "num_key_value_heads", None)
                or attn.k_proj.out_features // head_dim
            )
            self._head_dim.append(int(head_dim))
            self._num_attention_heads.append(int(num_heads))
            self._num_kv_heads.append(int(num_kv))

    # ---- attach / detach ---------------------------------------------------
    @classmethod
    def attach(cls, model: nn.Module) -> "HeadMaskController":
        attn_layers = _collect_attn_layers(model)
        ctrl = cls(model, attn_layers)
        for layer_idx, attn in enumerate(attn_layers):
            ctrl._handles.append(
                attn.q_proj.register_forward_hook(ctrl._make_proj_hook(layer_idx, "q"))
            )
            ctrl._handles.append(
                attn.k_proj.register_forward_hook(ctrl._make_proj_hook(layer_idx, "k"))
            )
            # V is masked at o_proj input (equivalent to scaling ``attn_output``
            # before the output projection — matches SHIPS post-softmax masking).
            ctrl._handles.append(
                attn.o_proj.register_forward_pre_hook(ctrl._make_o_proj_pre_hook(layer_idx))
            )
        return ctrl

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---- runtime config ----------------------------------------------------
    @contextmanager
    def active(self, mask_cfg: Mapping | None):
        prev = self._cfg
        prev_idx = self._token_idx
        self._cfg = mask_cfg
        self._token_idx = -1
        try:
            yield self
        finally:
            self._cfg = prev
            self._token_idx = prev_idx

    def set_token_index(self, idx: int) -> None:
        """Caller-driven token-position notifier for ``phase_window`` gating.

        Generation loops should call this once per generated token (before the
        forward pass that produces token ``idx``). ``idx == -1`` (default)
        means "prompt / unspecified" — hooks treat this as out of every
        finite window.
        """
        self._token_idx = int(idx)

    def _phase_window_active(self) -> bool:
        """True iff masks should fire at the current ``_token_idx``."""
        if self._cfg is None:
            return False
        window = self._cfg.get("phase_window")
        if window is None:
            return True
        start, end = int(window[0]), int(window[1])
        return start <= self._token_idx < end

    # ---- hook factories ----------------------------------------------------
    def _entries_for_layer(self, layer_idx: int) -> list[tuple[int, Sequence[str]]]:
        if self._cfg is None:
            return []
        if not self._phase_window_active():
            return []
        head_mask = self._cfg.get("head_mask") or {}
        out = []
        for (li, hi), qkv in head_mask.items():
            if int(li) == layer_idx:
                out.append((int(hi), list(qkv)))
        return out

    def _make_proj_hook(self, layer_idx: int, which: str):
        head_dim = self._head_dim[layer_idx]
        num_heads = self._num_attention_heads[layer_idx]
        num_kv = self._num_kv_heads[layer_idx]
        kv_groups = max(1, num_heads // max(1, num_kv))

        def hook(_module, _inputs, output):
            cfg = self._cfg
            if cfg is None:
                return output
            entries = self._entries_for_layer(layer_idx)
            if not entries:
                return output
            mask_type = cfg.get("mask_type")
            scale = cfg.get("scale_factor")
            x = output
            for head_idx, qkv in entries:
                if which not in qkv:
                    continue
                if which == "q":
                    x = _scale_or_mean_inplace(x, head_idx, head_dim, mask_type, scale)
                else:  # 'k' — map to KV head
                    kv_idx = head_idx // kv_groups
                    if kv_idx >= num_kv:
                        continue
                    x = _scale_or_mean_inplace(x, kv_idx, head_dim, mask_type, scale)
            return x

        return hook

    def _make_o_proj_pre_hook(self, layer_idx: int):
        head_dim = self._head_dim[layer_idx]

        def pre_hook(_module, inputs):
            cfg = self._cfg
            if cfg is None:
                return None
            entries = self._entries_for_layer(layer_idx)
            if not entries:
                return None
            mask_type = cfg.get("mask_type")
            scale = cfg.get("scale_factor")
            x = inputs[0]
            for head_idx, qkv in entries:
                if "v" not in qkv:
                    continue
                x = _scale_or_mean_inplace(x, head_idx, head_dim, mask_type, scale)
            return (x,) + tuple(inputs[1:])

        return pre_hook


def _collect_attn_layers(model: nn.Module) -> list[nn.Module]:
    """Walk the model and return the per-layer self-attention modules in order."""
    # Llama / Mistral / Qwen2 style: model.model.layers[i].self_attn
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            "Could not find `.model.layers` on the model — head-masking hooks expect "
            "a Llama/Mistral/Qwen2-style decoder."
        )
    out = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError("Decoder layer is missing `.self_attn`.")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(attn, name):
                raise ValueError(f"Attention module missing `.{name}` — unsupported architecture.")
        out.append(attn)
    return out


# ---------- Convenience wrappers ------------------------------------------

def num_layers_and_heads(model: nn.Module) -> tuple[int, int, int]:
    """Return ``(num_layers, num_attention_heads, num_key_value_heads)`` from config."""
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    return n_layers, n_heads, n_kv
