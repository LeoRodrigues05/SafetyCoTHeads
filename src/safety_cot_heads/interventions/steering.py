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
                                   layers: Optional[Sequence[int]] = None,
                                   mode: str = "ablate") -> dict:
    """Project the refusal direction out at every layer.

    ``mode="ablate"`` (every existing ``steering_ablate`` cell) cleans only the
    layer inputs; ``mode="ablate_all"`` cleans every residual write, which is
    the Arditi et al. (2024) operation (see ``SteeringController``).
    """
    if mode not in ("ablate", "ablate_all"):
        raise ValueError(f"ablation mode must be ablate|ablate_all, got {mode!r}")
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(np.asarray(direction)).float()
    layers = list(range(n_layers)) if layers is None else list(layers)
    return {
        "mode": mode,
        "direction": direction,
        "layers": layers,
        "alpha": 1.0,
    }


DOSE_MODES = ("absolute", "relative")


def build_activation_addition_cfg(*,
                                  direction: torch.Tensor | np.ndarray,
                                  layer: int,
                                  alpha: float = 1.0,
                                  dose_mode: str = "absolute") -> dict:
    """Turner et al. 2023 default: add ``alpha * v_hat`` at one chosen layer.

    The controller unit-normalises the direction, so with
    ``dose_mode="absolute"`` (every existing v5/v6 cell) ``alpha`` is the raw
    L2 size of the perturbation, identical across models even though their
    residual streams and harmful-benign separations differ by ~8x.
    ``dose_mode="relative"`` rescales to ``alpha * ||r_l||``: ``alpha=-1`` then
    subtracts exactly the harmful-minus-benign mean difference at that layer
    (Arditi et al.'s activation-addition scale), which is comparable across
    models.
    """
    if dose_mode not in DOSE_MODES:
        raise ValueError(f"dose_mode must be one of {DOSE_MODES}, got {dose_mode!r}")
    if not isinstance(direction, torch.Tensor):
        direction = torch.as_tensor(np.asarray(direction)).float()
    norm = float(direction.norm())
    alpha_abs = float(alpha) * (norm if dose_mode == "relative" else 1.0)
    return {
        "mode": "add",
        "direction": direction,
        "layers": [int(layer)],
        "alpha": alpha_abs,
        "alpha_requested": float(alpha),
        "dose_mode": dose_mode,
        "direction_norm": norm,
    }


def random_direction_like(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Seeded isotropic Gaussian direction with the same L2 norm as ``v``.

    Control for the refusal direction: same layer, same magnitude, same dose
    ladder, but no harmful-benign information.
    """
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(v.shape, generator=g, dtype=torch.float32)
    return r / (r.norm() + 1e-8) * float(v.norm())


def build_steering_cfg_from_file(lm: LoadedModel,
                                 *,
                                 direction_path: str | Path,
                                 layer: int,
                                 mode: Optional[str] = None,
                                 alpha: float = 1.0,
                                 layers: Optional[Sequence[int]] = None,
                                 dose_mode: str = "absolute",
                                 random_direction_seed: Optional[int] = None) -> dict:
    """Convenience: load a direction from disk and dispatch on ``mode``.

    ``mode`` must be given explicitly (``"add"`` or ``"ablate"``). A silent
    default is dangerous: ``"ablate"`` ignores ``alpha`` entirely, so an
    activation-addition dose sweep whose config omits ``mode`` would collapse
    to identical directional-ablation runs across all doses.
    """
    if mode is None:
        raise ValueError(
            "steering 'mode' must be set explicitly to 'add' or 'ablate'; "
            "refusing to default (ablate silently discards the alpha dose)"
        )
    v = _load_direction(direction_path, layer)
    extra = {}
    if random_direction_seed is not None:
        true_v = v
        v = random_direction_like(true_v, int(random_direction_seed))
        extra = {"random_direction_seed": int(random_direction_seed),
                 "cos_to_refusal_direction": float(
                     torch.nn.functional.cosine_similarity(v, true_v, dim=0))}
    if mode in ("ablate", "ablate_all"):
        n_layers, _, _ = num_layers_and_heads(lm.model)
        out = build_directional_ablation_cfg(direction=v, n_layers=n_layers,
                                             layers=layers, mode=mode)
    elif mode == "add":
        out = build_activation_addition_cfg(direction=v, layer=layer, alpha=alpha,
                                            dose_mode=dose_mode)
    else:
        raise ValueError(f"unknown steering mode {mode!r}; expected add|ablate|ablate_all")
    out.update(extra)
    return out


@contextmanager
def steer(lm: LoadedModel, steering_cfg: dict):
    with lm.steering_controller.active(steering_cfg):
        yield steering_cfg
