"""Mask config helpers.

A ``mask_cfg`` (the dict that flows into the custom model forward) has the shape::

    {
        "head_mask":   { (layer:int, head:int): list[str]  },  # 'q'|'k'|'v'
        "mask_qkv":    list[str],                              # default for new entries
        "mask_type":   "scale_mask" | "mean_mask",
        "scale_factor": float,                                 # used iff scale_mask
    }

This mirrors the semantics in the SafetyHeadAttribution repo
(``lib/utils/custommodel.py::CustomLlamaAttention.forward``).
"""

from __future__ import annotations
from copy import deepcopy
from typing import Iterable, Mapping, Sequence

MaskQKV = Sequence[str]
HeadKey = tuple[int, int]


def empty_mask_cfg(
    mask_qkv: MaskQKV = ("q",),
    mask_type: str = "scale_mask",
    scale_factor: float = 1e-4,
) -> dict:
    return {
        "head_mask": {},
        "mask_qkv": list(mask_qkv),
        "mask_type": mask_type,
        "scale_factor": float(scale_factor),
    }


def add_head(mask_cfg: dict, layer: int, head: int, mask_qkv: MaskQKV | None = None) -> dict:
    """Return a *copy* of ``mask_cfg`` with one extra head masked (no in-place mutation)."""
    new = deepcopy(mask_cfg)
    new.setdefault("head_mask", {})
    new["head_mask"][(int(layer), int(head))] = list(mask_qkv) if mask_qkv is not None else list(mask_cfg["mask_qkv"])
    return new


def add_heads(mask_cfg: dict, heads: Iterable[HeadKey], mask_qkv: MaskQKV | None = None) -> dict:
    out = deepcopy(mask_cfg)
    out.setdefault("head_mask", {})
    qkv = list(mask_qkv) if mask_qkv is not None else list(mask_cfg["mask_qkv"])
    for layer, head in heads:
        out["head_mask"][(int(layer), int(head))] = list(qkv)
    return out


def mask_cfg_kwargs(mask_cfg: Mapping | None) -> dict:
    """Convert a ``mask_cfg`` mapping into the kwargs the custom forward accepts."""
    if mask_cfg is None:
        return {"head_mask": None, "mask_type": None, "scale_factor": None}
    return {
        "head_mask": mask_cfg.get("head_mask") or None,
        "mask_type": mask_cfg.get("mask_type"),
        "scale_factor": mask_cfg.get("scale_factor"),
    }


def parse_head_id(s: str) -> HeadKey:
    """``'12-3'`` -> ``(12, 3)``."""
    a, b = s.split("-")
    return int(a), int(b)


def fmt_head_id(layer: int, head: int) -> str:
    return f"{int(layer)}-{int(head)}"
