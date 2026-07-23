"""Composite (P, Q, S) metric — orientation flag (Experiment 5, reverse/defence-side)."""
from __future__ import annotations

import math

import pytest

from safety_cot_heads.analysis.composite import CellSignals, axis_scores, sfs


def _signals(hac, clean=1.0, gap=0.0, model="m", dataset="jbb", condition="c"):
    return CellSignals(model=model, dataset=dataset, condition=condition,
                        hac=hac, clean=clean, gap=gap, n_clean=10, n_coh=10, n_gap=10)


def test_default_orientation_is_suppress():
    baseline = _signals(hac=0.1, condition="baseline")
    cell = _signals(hac=0.6, condition="steering_a1.0")
    a = axis_scores(cell, baseline)
    assert a.orientation == "suppress"
    assert math.isclose(a.P, (0.6 - 0.1) / (1 - 0.1), abs_tol=1e-6)


def test_suppress_orientation_explicit_matches_default():
    baseline = _signals(hac=0.1, condition="baseline")
    cell = _signals(hac=0.6, condition="steering_a1.0")
    a_default = axis_scores(cell, baseline)
    a_explicit = axis_scores(cell, baseline, orientation="suppress")
    assert a_default.P == a_explicit.P


def test_defend_orientation_rewards_harm_removal():
    # A defence reduces harm below an already-jailbroken baseline.
    baseline = _signals(hac=0.8, condition="baseline")
    cell = _signals(hac=0.2, condition="steering_defend_a1.0")
    a = axis_scores(cell, baseline, orientation="defend")
    assert a.orientation == "defend"
    assert math.isclose(a.P, (0.8 - 0.2) / 0.8, abs_tol=1e-6)
    assert a.P > 0.7


def test_defend_orientation_clips_at_zero_when_defence_increases_harm():
    baseline = _signals(hac=0.2, condition="baseline")
    cell = _signals(hac=0.8, condition="steering_defend_a1.0")  # got worse
    a = axis_scores(cell, baseline, orientation="defend")
    assert a.P == 0.0


def test_baseline_vs_baseline_is_p_zero_in_both_orientations():
    baseline = _signals(hac=0.4, condition="baseline")
    same = _signals(hac=0.4, condition="baseline")
    a_suppress = axis_scores(same, baseline, orientation="suppress")
    a_defend = axis_scores(same, baseline, orientation="defend")
    assert a_suppress.P == 0.0
    assert a_defend.P == 0.0


def test_q_and_s_unchanged_across_orientation():
    baseline = _signals(hac=0.5, clean=0.9, gap=0.1, condition="baseline")
    cell = _signals(hac=0.2, clean=0.8, gap=0.05, condition="steering_defend_a1.0")
    a_suppress = axis_scores(cell, baseline, orientation="suppress")
    a_defend = axis_scores(cell, baseline, orientation="defend")
    assert a_suppress.Q == a_defend.Q
    assert a_suppress.S == a_defend.S
    assert a_suppress.P != a_defend.P


def test_unknown_orientation_raises():
    baseline = _signals(hac=0.1)
    cell = _signals(hac=0.5)
    with pytest.raises(ValueError):
        axis_scores(cell, baseline, orientation="sideways")


def test_incomplete_cell_returns_none_under_defend_too():
    baseline = _signals(hac=0.1)
    incomplete = CellSignals(model="m", dataset="jbb", condition="c",
                              hac=None, clean=1.0, gap=0.0)
    assert axis_scores(incomplete, baseline, orientation="defend") is None


def test_sfs_still_collapses_to_zero_on_defend_axes():
    baseline = _signals(hac=0.9, clean=0.9, gap=0.0, condition="baseline")
    cell = _signals(hac=0.1, clean=0.0, gap=0.0, condition="steering_defend_a1.0")
    a = axis_scores(cell, baseline, orientation="defend")
    assert a.P > 0.0
    assert sfs(a) == 0.0  # Q == 0 (destroyed coherence) -> geometric mean is 0
