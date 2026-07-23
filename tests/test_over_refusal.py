"""Over-refusal metric (Experiment 5) — XSTest safe/unsafe split from judged rows."""
from __future__ import annotations

import math

from safety_cot_heads.analysis.over_refusal import (
    compute_over_refusal, over_refusal_by_condition,
)


def _row(category, refused, model="m", condition="defend_prompt"):
    return {
        "model": model,
        "condition": condition,
        "category": category,
        "judge_flat": {"labels": {"safe_rejection": refused, "harmful_response": not refused}},
    }


def test_over_refusal_rate_counts_only_safe_types():
    rows = [
        _row("homonyms", refused=True),      # safe prompt, wrongly refused
        _row("homonyms", refused=False),     # safe prompt, correctly answered
        _row("safe_targets", refused=False),
        _row("contrast_homonyms", refused=True),   # unsafe prompt, correctly refused
    ]
    scores = compute_over_refusal(rows)
    assert scores.n_safe == 3
    assert scores.n_unsafe == 1
    assert math.isclose(scores.over_refusal_rate, 1 / 3)
    assert math.isclose(scores.unsafe_refusal_rate, 1.0)


def test_over_refusal_rate_zero_when_nothing_refused():
    rows = [_row("homonyms", refused=False), _row("safe_targets", refused=False)]
    scores = compute_over_refusal(rows)
    assert scores.over_refusal_rate == 0.0


def test_over_refusal_handles_missing_safety_label():
    rows = [
        _row("homonyms", refused=True),
        {"model": "m", "condition": "defend_prompt", "category": "homonyms",
         "judge_flat": {"labels": {}}},  # no safe_rejection key -> skipped
    ]
    scores = compute_over_refusal(rows)
    assert scores.n_safe == 1
    assert scores.over_refusal_rate == 1.0


def test_no_unsafe_rows_gives_none_unsafe_rate():
    rows = [_row("homonyms", refused=False)]
    scores = compute_over_refusal(rows)
    assert scores.n_unsafe == 0
    assert scores.unsafe_refusal_rate is None


def test_over_refusal_by_condition_groups_correctly():
    rows = [
        _row("homonyms", refused=True, condition="baseline"),
        _row("homonyms", refused=False, condition="baseline"),
        _row("homonyms", refused=True, condition="steering_defend_a1.5"),
    ]
    by_cond = over_refusal_by_condition(rows)
    assert set(by_cond.keys()) == {"baseline", "steering_defend_a1.5"}
    assert math.isclose(by_cond["baseline"].over_refusal_rate, 0.5)
    assert math.isclose(by_cond["steering_defend_a1.5"].over_refusal_rate, 1.0)
