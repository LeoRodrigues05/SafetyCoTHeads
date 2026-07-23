"""Tests for paired monitorability metrics and corrected S axis."""

import math
import pytest

from safety_cot_heads.direction_a_v6.paired_metrics import (
    PairedItem, build_paired_table, monitorability_retention,
)


def _items(pairs, clean=None, trace_available=None):
    """pairs: list of (y, t). clean/trace_available optional parallel lists."""
    out = []
    for i, (y, t) in enumerate(pairs):
        out.append(PairedItem(
            prompt_id=f"p{i}", y=y, t=t,
            clean=True if clean is None else clean[i],
            trace_available=True if trace_available is None else trace_available[i],
        ))
    return out


def test_full_2x2_counts():
    # 2 hh, 1 hs, 1 sh, 1 ss
    items = _items([(1, 1), (1, 1), (1, 0), (0, 1), (0, 0)])
    t = build_paired_table(items)
    assert (t.n_hh, t.n_hs, t.n_sh, t.n_ss) == (2, 1, 1, 1)
    assert t.n_pairs == 5
    assert t.n_harmful == 3
    assert t.U == pytest.approx(1 / 5)
    assert t.O == pytest.approx(1 / 5)
    assert t.A == pytest.approx(3 / 5)
    assert t.trace_fnr == pytest.approx(1 / 3)


def test_cancellation_counterexample():
    # U = 0.20 and O = 0.20 -> signed gap 0, but covert failure stays 0.20.
    # 10 items: 2 covert (1,0), 2 over-warning (0,1), 3 hh, 3 ss
    items = _items(
        [(1, 0)] * 2 + [(0, 1)] * 2 + [(1, 1)] * 3 + [(0, 0)] * 3
    )
    t = build_paired_table(items)
    assert t.U == pytest.approx(0.20)
    assert t.O == pytest.approx(0.20)
    assert t.g == pytest.approx(0.0)          # signed gap cancels
    assert t.U == pytest.approx(0.20)         # covert failure does NOT vanish
    # explicit invariant: U and O both preserved, cannot cancel into one number
    assert t.U is not None and t.O is not None


def test_trace_fnr_undefined_without_harmful():
    items = _items([(0, 0), (0, 1), (0, 0)])
    t = build_paired_table(items)
    assert t.n_harmful == 0
    assert t.trace_fnr is None            # undefined, NOT zero


def test_missing_judgments_are_dropped_not_zeroed():
    items = _items([(1, None), (None, 0), (1, 1), (0, 0)])
    t = build_paired_table(items)
    assert t.n_pairs == 2                 # only the two complete pairs
    assert t.n_missing_t == 1
    assert t.n_missing_y == 1


def test_nonclean_excluded_by_default_included_on_request():
    items = _items([(1, 0), (1, 0)], clean=[True, False])
    primary = build_paired_table(items, include_nonclean=False)
    assert primary.n_pairs == 1
    assert primary.n_nonclean_excluded == 1
    allp = build_paired_table(items, include_nonclean=True)
    assert allp.n_pairs == 2


def test_missing_trace_excluded_when_required():
    items = _items([(1, 0), (1, None)], trace_available=[True, False])
    t = build_paired_table(items, require_trace=True)
    assert t.n_no_trace_excluded == 1
    assert t.n_pairs == 1


def test_monitorability_retention_baseline_correction():
    base = build_paired_table(_items([(1, 1), (0, 0), (1, 1), (0, 0)]))   # U_b = 0
    cell = build_paired_table(_items([(1, 0), (0, 0), (1, 1), (0, 0)]))   # U_c = 0.25
    mr = monitorability_retention(cell, base)
    assert mr.U_b == pytest.approx(0.0)
    assert mr.U_c == pytest.approx(0.25)
    assert mr.S == pytest.approx(1 - 0.25)     # 1 - clip(U_c - U_b)


def test_S_clipped_when_intervention_improves():
    base = build_paired_table(_items([(1, 0), (0, 0)]))   # U_b = 0.5
    cell = build_paired_table(_items([(1, 1), (0, 0)]))   # U_c = 0.0
    mr = monitorability_retention(cell, base)
    # U_c - U_b < 0 -> clip to 0 -> S = 1 (fully retained)
    assert mr.S == pytest.approx(1.0)


def test_S_none_when_paired_denominator_empty():
    empty = build_paired_table([])
    other = build_paired_table(_items([(1, 0)]))
    assert monitorability_retention(other, empty).S is None
    assert monitorability_retention(empty, other).S is None
