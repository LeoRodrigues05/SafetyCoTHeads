"""Unit tests for template-anchoring residualisation (P1.5).

The GPU-dependent ``compute_head_template_anchoring`` is not covered here;
those checks live in the smoke job invoked by the SLURM pipeline (it asserts
``len(anchoring) == n_layers * n_heads`` and that values lie in ``[0, 1]``).

These tests pin the pure-Python residualiser, which is the piece that
ranking-residualised SHIPS comparisons (§4.3) depend on.
"""
from __future__ import annotations

import math

from safety_cot_heads.attribution.template_anchoring import (
    residualize_on_template_anchoring,
)


def _check_close(a: float, b: float, tol: float = 1e-9) -> None:
    assert math.isclose(a, b, abs_tol=tol), f"{a} !~= {b}"


def test_residual_zero_when_score_proportional_to_anchoring() -> None:
    """If score == 2 * rho_tpl exactly, all residuals must be ~0."""
    anchoring = {(0, 0): 0.1, (0, 1): 0.5, (0, 2): 0.9}
    ranking = [
        {"layer": 0, "head": 0, "score": 0.2},
        {"layer": 0, "head": 1, "score": 1.0},
        {"layer": 0, "head": 2, "score": 1.8},
    ]
    out = residualize_on_template_anchoring(ranking, anchoring)
    for row in out:
        _check_close(row["score_resid"], 0.0, tol=1e-9)
        assert row["rho_tpl"] is not None
        assert row["rank_resid"] in {1, 2, 3}


def test_residual_preserves_score_when_anchoring_uncorrelated() -> None:
    """Constant anchoring → OLS slope 0 → residual = score - mean(score)."""
    anchoring = {(0, h): 0.5 for h in range(4)}
    scores = [1.0, 2.0, 3.0, 4.0]
    ranking = [{"layer": 0, "head": h, "score": s} for h, s in enumerate(scores)]
    out = residualize_on_template_anchoring(ranking, anchoring)
    mean = sum(scores) / len(scores)
    for row, s in zip(out, scores):
        _check_close(row["score_resid"], s - mean)
    # Ranking is preserved (highest score → rank 1)
    sorted_by_resid = sorted(out, key=lambda r: r["rank_resid"])
    assert [r["head"] for r in sorted_by_resid] == [3, 2, 1, 0]


def test_residual_handles_missing_anchoring() -> None:
    """Heads with no rho_tpl get score_resid=None and rank_resid=None."""
    anchoring = {(0, 0): 0.2, (0, 1): 0.8}  # head 2 missing
    ranking = [
        {"layer": 0, "head": 0, "score": 1.0},
        {"layer": 0, "head": 1, "score": 2.0},
        {"layer": 0, "head": 2, "score": 5.0},
    ]
    out = residualize_on_template_anchoring(ranking, anchoring)
    assert out[2]["rho_tpl"] is None
    assert out[2]["score_resid"] is None
    assert out[2]["rank_resid"] is None
    # The two heads with anchoring still get residuals (zero, since with n=2
    # the line is perfect).
    _check_close(out[0]["score_resid"], 0.0)
    _check_close(out[1]["score_resid"], 0.0)


def test_residual_returns_copy_does_not_mutate_input() -> None:
    anchoring = {(0, 0): 0.3, (0, 1): 0.6}
    ranking = [
        {"layer": 0, "head": 0, "score": 1.0},
        {"layer": 0, "head": 1, "score": 2.0},
    ]
    snapshot = [dict(r) for r in ranking]
    _ = residualize_on_template_anchoring(ranking, anchoring)
    assert ranking == snapshot


def test_residual_negative_correlation_inverts_ranking() -> None:
    """Score decreases with anchoring → residualised line absorbs the trend."""
    anchoring = {(0, 0): 0.1, (0, 1): 0.5, (0, 2): 0.9}
    ranking = [
        {"layer": 0, "head": 0, "score": 0.9},  # low anchoring, high score
        {"layer": 0, "head": 1, "score": 0.5},
        {"layer": 0, "head": 2, "score": 0.1},
    ]
    out = residualize_on_template_anchoring(ranking, anchoring)
    # Perfect negative correlation → all residuals ~0 (within FP noise);
    # tie-break order is implementation-defined and not part of the contract.
    for r in out:
        _check_close(r["score_resid"], 0.0, tol=1e-9)
        assert r["rank_resid"] in {1, 2, 3}
    assert sorted(r["rank_resid"] for r in out) == [1, 2, 3]
