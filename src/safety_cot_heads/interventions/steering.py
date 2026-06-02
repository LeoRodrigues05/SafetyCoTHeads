"""Direction-steering runtime helpers (Arditi et al. 2024; Turner et al. 2023).

Build a ``steering_cfg`` dict for :class:`SteeringController`. Two default
protocols are exposed:

* :func:`build_directional_ablation_cfg` — Arditi et al. (2024) "Refusal is
  Mediated by a Single Direction". Project the refusal direction out of the
  residual stream at *every* layer. This is the recommended default for
  reducing refusal behaviour.
* :func:`build_activation_addition_cfg` — Turner et al. (2023) "Activation
  Addition" / Zou et al. (2023) RepE. Add ``alpha * v`` at a single chosen
  layer (often ≈40 % depth).
"""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from ..models import LoadedModel
from ..models.custom_llama import num_layers_and_heads


def _load_direction(path: str | Path, layer: int) -> torch.Tensor:
    """Load a per-layer direction from a ``.npz`` produced by
    :func:`safety_cot_heads.attribution.directions.compute_refusal_directions`."""
    arr = np.load(path)
    key = f"layer_{int(layer):02d}"
    if key not in arr.files:
        raise KeyError(f"layer {layer} not in {path}; have {arr.files[:5]}…")
    return torch.from_numpy(arr[key]).float()


def build_directional_ablation_cfg(*,
                                   direction: torch.Tensor | np.ndarray,
                                   n_layers: int,
                                   layers: Optional[Sequence[int]] = None) -> dict:
    """Arditi et al. 2024 default: ablate the refusal direction at every layer."""
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(np.asarray(direction)).float()
    layers = list(range(n_layers)) if layers is None else list(layers)
    return {
        "mode": "ablate",
        "direction": direction,
        "layers": layers,
        "alpha": 1.0,
    }


def build_activation_addition_cfg(*,
                                  direction: torch.Tensor | np.ndarray,
                                  layer: int,
                                  alpha: float = 1.0) -> dict:
    """Turner et al. 2023 default: add ``alpha * v`` at one chosen layer."""
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(np.asarray(direction)).float()
    return {
        "mode": "add",
        "direction": direction,
        "layers": [int(layer)],
        "alpha": float(alpha),
    }


def build_steering_cfg_from_file(lm: LoadedModel,
                                 *,
                                 direction_path: str | Path,
                                 layer: int,
                                 mode: str = "ablate",
                                 alpha: float = 1.0,
                                 layers: Optional[Sequence[int]] = None) -> dict:
    """Convenience: load a direction from disk and dispatch on ``mode``."""
    v = _load_direction(direction_path, layer)
    if mode == "ablate":
        n_layers, _, _ = num_layers_and_heads(lm.model)
        return build_directional_ablation_cfg(direction=v, n_layers=n_layers, layers=layers)
    if mode == "add":
        return build_activation_addition_cfg(direction=v, layer=layer, alpha=alpha)
    raise ValueError(f"unknown steering mode {mode!r}; expected add|ablate")


@contextmanager
def steer(lm: LoadedModel, steering_cfg: dict):
    with lm.steering_controller.active(steering_cfg):
        yield steering_cfg
