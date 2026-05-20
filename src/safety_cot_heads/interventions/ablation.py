"""Head ablation — the runtime side of the SHIPS / Sahara mask config.

Given a list of heads to ablate (and a mask spec: ``mask_qkv``, ``mask_type``,
``scale_factor``) this module produces a ``mask_cfg`` dict and an
``ablate_heads(lm, heads, ...)`` context manager that activates the mask on
the :class:`HeadMaskController` attached to ``lm``.

Nothing here mutates model weights — it is the *runtime* hook-based ablation.
For destructive weight surgery see :mod:`safety_cot_heads.interventions.surgery`.
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Iterable, Sequence

from ..models import LoadedModel
from ..models.masks import add_heads, empty_mask_cfg, parse_head_id


def build_mask_cfg(heads: Iterable,
                   *,
                   mask_qkv: Sequence[str] = ("q",),
                   mask_type: str = "scale_mask",
                   scale_factor: float = 1e-4) -> dict:
    """Build a SHIPS-shaped mask_cfg dict.

    ``heads`` can be a list of ``(layer, head)`` tuples or string ids like
    ``"12-3"``.
    """
    parsed: list[tuple[int, int]] = []
    for h in heads:
        if isinstance(h, str):
            parsed.append(parse_head_id(h))
        elif isinstance(h, dict):
            parsed.append((int(h["layer"]), int(h["head"])))
        else:
            parsed.append((int(h[0]), int(h[1])))
    cfg = empty_mask_cfg(mask_qkv=mask_qkv, mask_type=mask_type, scale_factor=scale_factor)
    return add_heads(cfg, parsed)


@contextmanager
def ablate_heads(lm: LoadedModel,
                  heads: Iterable,
                  *,
                  mask_qkv: Sequence[str] = ("q",),
                  mask_type: str = "scale_mask",
                  scale_factor: float = 1e-4):
    """Context manager: activates the ablation for the duration of the block."""
    cfg = build_mask_cfg(heads, mask_qkv=mask_qkv, mask_type=mask_type,
                          scale_factor=scale_factor)
    with lm.head_mask_controller.active(cfg):
        yield cfg
