"""Random / layer-matched head selection tests."""
from __future__ import annotations
from collections import Counter

from safety_cot_heads.attribution import layer_matched, uniform_random


def test_uniform_random_size_and_uniqueness():
    heads = uniform_random(n_layers=32, n_heads=32, k=10, seed=0)
    assert len(heads) == 10
    assert len(set(heads)) == 10
    for l, h in heads:
        assert 0 <= l < 32 and 0 <= h < 32


def test_uniform_random_deterministic():
    a = uniform_random(8, 8, 5, seed=42)
    b = uniform_random(8, 8, 5, seed=42)
    assert a == b


def test_layer_matched_preserves_per_layer_distribution():
    ref = [(3, 0), (3, 1), (5, 0), (10, 4), (10, 5), (10, 7)]
    out = layer_matched(ref, n_heads_per_layer=32, seed=0)
    ref_layer_counts = Counter(l for (l, _) in ref)
    out_layer_counts = Counter(l for (l, _) in out)
    assert ref_layer_counts == out_layer_counts
    # All within range
    for l, h in out:
        assert 0 <= h < 32
