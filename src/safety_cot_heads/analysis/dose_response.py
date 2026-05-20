"""Dose-response: harmful_rate vs number of ablated heads."""

from __future__ import annotations
from typing import Callable, Iterable, Sequence


def dose_curve(ranked_heads: Sequence[dict],
                doses: Iterable[int],
                evaluate: Callable[[list[dict]], float]) -> list[dict]:
    """For each dose ``k``, take the top-k heads and call ``evaluate(heads)``.

    Returns ``[{"k": int, "score": float}, ...]``.  ``evaluate`` is expected
    to drive generation + judging under the given ablation set.
    """
    curve = []
    for k in doses:
        chosen = list(ranked_heads[:k])
        curve.append({"k": k, "score": float(evaluate(chosen))})
    return curve
