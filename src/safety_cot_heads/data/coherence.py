"""Coherence-eval prompts.

For the coherency-head module we use a small mixed set of benign prompts that
should elicit *long-form* coherent answers (so that the per-head ablation can
visibly degrade the trajectory).  We default to the first ``n`` Alpaca
instructions; users may swap in their own list via ``load_coherence_prompts``.
"""
from __future__ import annotations
from typing import Optional, Sequence


def load_coherence_prompts(n: int = 32,
                            extra: Optional[Sequence[str]] = None) -> list[dict]:
    from .benign import load_alpaca
    rows = load_alpaca(n=n)
    if extra:
        for i, p in enumerate(extra):
            rows.append({"id": f"coh-extra-{i:05d}", "prompt": p,
                         "dataset": "coherence-extra"})
    return rows
