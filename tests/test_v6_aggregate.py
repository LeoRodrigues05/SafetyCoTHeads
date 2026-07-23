"""Tests for corrected P/Q/S/SFS aggregation."""

import pytest

from safety_cot_heads.direction_a_v6.aggregate import (
    AnswerSignals, potency, quality, compute_cell_axes,
)
from safety_cot_heads.direction_a_v6.paired_metrics import PairedItem, build_paired_table


def _ans(hac, clean, **kw):
    return AnswerSignals(model="m", dataset="d", condition=kw.get("cond", "c"),
                         hac=hac, clean_rate=clean, **{k: v for k, v in kw.items() if k != "cond"})


def test_potency_baseline_correction():
    base = _ans(0.2, 1.0)
    cell = _ans(0.6, 1.0)
    # (0.6 - 0.2) / (1 - 0.2) = 0.5
    assert potency(cell, base) == pytest.approx(0.5, abs=1e-6)


def test_potency_clipped_nonnegative():
    base = _ans(0.6, 1.0)
    cell = _ans(0.2, 1.0)   # intervention reduces harm -> clip to 0
    assert potency(cell, base) == pytest.approx(0.0)


def test_quality_baseline_correction():
    base = _ans(0.0, 0.8)
    cell = _ans(0.0, 0.4)
    assert quality(cell, base) == pytest.approx(0.5, abs=1e-6)


def test_quality_clipped_at_one():
    base = _ans(0.0, 0.5)
    cell = _ans(0.0, 0.9)   # improved coherence -> clip to 1
    assert quality(cell, base) == pytest.approx(1.0)


def test_sfs_only_when_all_axes_present():
    base_a = _ans(0.2, 1.0)
    cell_a = _ans(0.6, 1.0)
    pt_base = build_paired_table([PairedItem("p0", 1, 1), PairedItem("p1", 0, 0)])
    pt_cell = build_paired_table([PairedItem("p0", 1, 0), PairedItem("p1", 0, 0)])
    axes = compute_cell_axes(cell_a, base_a, pt_cell, pt_base)
    assert axes.P is not None and axes.Q is not None and axes.S is not None
    assert axes.sfs == pytest.approx((axes.P * axes.Q * axes.S) ** (1 / 3))
    # backward-compat product equals sfs**3
    assert axes.pqs_product == pytest.approx(axes.sfs ** 3, abs=1e-9)


def test_sfs_none_when_S_missing():
    base_a = _ans(0.2, 1.0)
    cell_a = _ans(0.6, 1.0)
    axes = compute_cell_axes(cell_a, base_a, paired_cell=None, paired_baseline=None)
    assert axes.S is None
    assert axes.sfs is None           # never silently set to 1
    assert axes.axes_available["SFS"] is False


def test_denominators_exposed():
    base_a = _ans(0.2, 1.0)
    cell_a = _ans(0.6, 1.0, n_generated=100, n_clean=90, n_harmful_clean=54)
    axes = compute_cell_axes(cell_a, base_a)
    assert axes.denominators["n_generated"] == 100
    assert axes.denominators["n_clean"] == 90
