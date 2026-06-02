"""Circuit-level intervention helpers.

A "circuit" in this codebase is the union of three ablation surfaces, all
applied jointly during a single generation:

* a set of attention heads (head-mask, SHIPS ranking),
* a set of MLP neurons (neuron-mask, Wang et al. ranking),
* and a refusal direction projected out of the residual stream
  (Arditi et al. directional ablation).

This module just builds the three sub-cfgs from rankings on disk; the
generation script wires them through the three controllers via nested
``with`` contexts.

This pragmatic construction is in the spirit of "edge attribution
patching" (Conmy et al. 2023 ACDC; Syed et al. 2023 EAP) — we union the
top components from each independent attribution score rather than running
full ACDC, but the resulting intervention surface is a valid (greedy)
sub-circuit with respect to the harmful-vs-benign objective each attribution
was computed against.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..models import LoadedModel
from ..utils import json_load
from .ablation import build_mask_cfg
from .neuron_ablation import build_neuron_mask_cfg
from .steering import build_steering_cfg_from_file


def _load_top_heads(path: str | Path, k: int) -> list[tuple[int, int]]:
    data = json_load(path)
    if isinstance(data, dict):
        data = (data.get("dataset_ranking")
                or data.get("ranked_heads")
                or data.get("selected_heads")
                or [])
    return [(int(h["layer"]), int(h["head"])) for h in data[:k]]


def _load_top_neurons(path: str | Path, k: int) -> list[tuple[int, int]]:
    data = json_load(path)
    if isinstance(data, dict):
        data = data.get("ranked_neurons") or data.get("dataset_ranking") or []
    out = []
    for n in data[:k]:
        out.append((int(n["layer"]), int(n["neuron"])))
    return out


def build_circuit_cfgs(lm: LoadedModel,
                       *,
                       heads_path: Optional[str | Path] = None,
                       top_heads: int = 8,
                       mask_qkv: Sequence[str] = ("q",),
                       head_mask_type: str = "scale_mask",
                       head_scale_factor: float = 1e-4,
                       neurons_path: Optional[str | Path] = None,
                       top_neurons: int = 32,
                       neuron_scale_factor: float = 0.0,
                       direction_path: Optional[str | Path] = None,
                       direction_layer: Optional[int] = None,
                       steering_mode: str = "ablate") -> dict:
    """Return a dict with ``head_cfg``, ``neuron_cfg``, ``steering_cfg``
    keys (any may be ``None`` if its source is not provided)."""
    out: dict = {"head_cfg": None, "neuron_cfg": None, "steering_cfg": None}
    if heads_path is not None:
        heads = _load_top_heads(heads_path, top_heads)
        out["head_cfg"] = build_mask_cfg(
            heads, mask_qkv=mask_qkv, mask_type=head_mask_type,
            scale_factor=head_scale_factor,
        )
        out["heads"] = heads
    if neurons_path is not None:
        neurons = _load_top_neurons(neurons_path, top_neurons)
        out["neuron_cfg"] = build_neuron_mask_cfg(
            neurons, scale_factor=neuron_scale_factor,
        )
        out["neurons"] = neurons
    if direction_path is not None and direction_layer is not None:
        out["steering_cfg"] = build_steering_cfg_from_file(
            lm,
            direction_path=direction_path,
            layer=int(direction_layer),
            mode=steering_mode,
        )
    return out
