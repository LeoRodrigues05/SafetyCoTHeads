"""Neuron-ablation runtime helpers.

Build a ``neuron_mask_cfg`` dict for :class:`NeuronMaskController`. Default
hyperparameters follow Wang et al. (2024) "Finding Safety Neurons in Large
Language Models" — full ablation (``scale_factor = 0.0``).
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Iterable, Sequence

from ..models import LoadedModel


def build_neuron_mask_cfg(neurons: Iterable,
                          *,
                          mask_type: str = "scale_mask",
                          scale_factor: float = 0.0) -> dict:
    """Build a neuron-mask cfg dict.

    ``neurons`` items can be ``(layer, neuron)`` tuples, ``{"layer", "neuron"}``
    dicts, or string ids like ``"12-3072"``.

    Defaults
    --------
    * ``mask_type="scale_mask"`` + ``scale_factor=0.0``: full ablation
      (Wang et al. 2024). Use ``scale_factor=0.0`` for the ablation
      reported as "Safety-Neuron Ablation" in their Table 2.
    """
    parsed: list[tuple[int, int]] = []
    for n in neurons:
        if isinstance(n, str):
            l, k = n.split("-", 1)
            parsed.append((int(l), int(k)))
        elif isinstance(n, dict):
            parsed.append((int(n["layer"]), int(n["neuron"])))
        else:
            parsed.append((int(n[0]), int(n[1])))
    mask = {(l, k): "n" for (l, k) in parsed}
    return {
        "neuron_mask": mask,
        "mask_type": mask_type,
        "scale_factor": float(scale_factor),
    }


@contextmanager
def ablate_neurons(lm: LoadedModel, neurons: Iterable,
                   *, mask_type: str = "scale_mask",
                   scale_factor: float = 0.0):
    cfg = build_neuron_mask_cfg(neurons, mask_type=mask_type, scale_factor=scale_factor)
    with lm.neuron_mask_controller.active(cfg):
        yield cfg
