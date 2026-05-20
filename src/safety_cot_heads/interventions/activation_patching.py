"""Activation patching — stub.

Future work: swap the residual-stream contribution of a target attention head
on a *harmful* prompt with its contribution on a *paired benign* prompt
(same prefix, different last query) to test the causal role of safety heads.

This file is intentionally a stub — the full implementation will come in a
later phase (see :doc:`ExperimentTracker.md`, "Future" section).
"""

from __future__ import annotations
from typing import Any


def patch_head_activation(*args: Any, **kwargs: Any):
    """Not implemented yet."""
    raise NotImplementedError(
        "activation patching is part of the post-MVP phase; see ExperimentTracker.md."
    )
