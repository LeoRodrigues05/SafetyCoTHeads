"""Random / layer-matched / activation-matched head baselines.

Used to build the control conditions:

* ``random_head_ablation``      — uniform random over the full (layer, head) grid
* ``layer_matched_random_head_ablation`` — same per-layer count distribution as
  a reference safety-head set, but random head indices within each layer.
* ``activation_magnitude_matched`` — match the activation magnitude profile of
  the safety heads (used as a stricter sanity check in the trajectory paper).
"""
from __future__ import annotations
import random
from collections import Counter
from typing import Sequence


def uniform_random(n_layers: int, n_heads: int, k: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    rng.shuffle(pool)
    return pool[:k]


def layer_matched(reference_heads: Sequence[tuple[int, int]],
                  n_heads_per_layer: int,
                  seed: int = 0,
                  exclude_reference: bool = False) -> list[tuple[int, int]]:
    """Random heads with the reference set's per-layer counts.

    ``exclude_reference=True`` never re-draws a reference head, so the control
    shares no target with the discovered set (used by the random-target
    controls; the default keeps the historical behaviour).
    """
    rng = random.Random(seed)
    ref = set(map(tuple, reference_heads))
    by_layer = Counter(l for (l, _) in reference_heads)
    out: list[tuple[int, int]] = []
    for layer, count in by_layer.items():
        avail = [h for h in range(n_heads_per_layer)
                 if not (exclude_reference and (layer, h) in ref)]
        rng.shuffle(avail)
        out.extend((layer, h) for h in avail[:count])
    return out


def layer_matched_neurons(reference_neurons: Sequence[tuple[int, int]],
                          intermediate_size: int,
                          seed: int = 0,
                          exclude_reference: bool = True) -> list[tuple[int, int]]:
    """Random MLP neurons with the reference set's per-layer counts.

    The discovered top-k neurons are concentrated in the last layers (raw
    activation scale grows with depth), so a uniform random draw would be an
    unfair control; matching the per-layer histogram isolates *which* neurons
    were chosen from *where* they sit.
    """
    rng = random.Random(seed)
    ref = set(map(tuple, reference_neurons))
    by_layer = Counter(l for (l, _) in reference_neurons)
    out: list[tuple[int, int]] = []
    for layer in sorted(by_layer):
        count = by_layer[layer]
        avail = [n for n in range(intermediate_size)
                 if not (exclude_reference and (layer, n) in ref)]
        out.extend((layer, n) for n in rng.sample(avail, count))
    return out


def activation_magnitude_matched(activations,                  # dict[(l,h)->float]
                                  reference_heads: Sequence[tuple[int, int]],
                                  seed: int = 0,
                                  tolerance: float = 0.1) -> list[tuple[int, int]]:
    """For each reference head pick a random head whose mean |activation| is
    within ``tolerance`` (relative) and that is not itself a reference head."""
    rng = random.Random(seed)
    ref_set = set(reference_heads)
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for h in reference_heads:
        target = activations[h]
        lo, hi = target * (1 - tolerance), target * (1 + tolerance)
        candidates = [k for k, v in activations.items()
                      if lo <= v <= hi and k not in ref_set and k not in used]
        if not candidates:
            candidates = [k for k in activations if k not in ref_set and k not in used]
        pick = rng.choice(candidates)
        chosen.append(pick)
        used.add(pick)
    return chosen
