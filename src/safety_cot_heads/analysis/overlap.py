"""Overlap analysis between attribution rankings (safety vs coherency vs quality)."""

from __future__ import annotations
from typing import Iterable


def head_set(rows: Iterable[dict], top_k: int | None = None) -> set[tuple[int, int]]:
    """Extract ``{(layer, head)}`` from a ranked_heads list (or list of attribution rows)."""
    out: set[tuple[int, int]] = set()
    iter_rows = list(rows)
    if iter_rows and "ranked_heads" in iter_rows[0]:
        # an attribution row → take the top_k of each
        for r in iter_rows:
            ranked = r["ranked_heads"]
            if top_k:
                ranked = ranked[:top_k]
            for h in ranked:
                out.add((int(h["layer"]), int(h["head"])))
    else:
        # already a list of head dicts
        ranked = iter_rows
        if top_k:
            ranked = ranked[:top_k]
        for h in ranked:
            out.add((int(h["layer"]), int(h["head"])))
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def overlap_report(named: dict[str, set]) -> dict:
    out = {"sizes": {k: len(v) for k, v in named.items()},
           "intersections": {}, "jaccard": {}}
    names = list(named.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            inter = named[n1] & named[n2]
            out["intersections"][f"{n1}∩{n2}"] = sorted(inter)
            out["jaccard"][f"{n1}∩{n2}"] = jaccard(named[n1], named[n2])
    return out


def safety_excluding(safety: set, coherency: set) -> set:
    return safety - coherency
