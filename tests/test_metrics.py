"""Metric edge cases."""
from __future__ import annotations

import math

from safety_cot_heads.analysis import asr, harmful_rate, refusal_rate, wilson_ci
from safety_cot_heads.judging import aggregate_safety


def _row(harmful=False, safe=False, cond="baseline"):
    return {
        "condition": cond,
        "judge_flat": {
            "labels": {
                "safe_rejection": safe,
                "harmful_response": harmful,
                "reasoning_about_safety": False,
                "adding_intention": False,
                "changing_subject": False,
            },
            "confidences": {},
        },
    }


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_ci_full_and_empty_proportions():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = wilson_ci(100, 100)
    assert math.isclose(hi, 1.0) and 0.9 < lo <= 1.0


def test_wilson_ci_brackets_p():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_harmful_refusal_rates_simple():
    rows = [_row(harmful=True)] * 3 + [_row(safe=True)] * 7
    assert math.isclose(harmful_rate(rows), 0.3)
    assert math.isclose(refusal_rate(rows), 0.7)
    assert asr(rows) == harmful_rate(rows)


def test_aggregate_safety_counts():
    rows = [_row(harmful=True, cond="ablate")] * 4 + [_row(safe=True, cond="baseline")] * 6
    summary = aggregate_safety(rows)
    assert summary["ablate"]["harmful_rate"] == 1.0
    assert summary["ablate"]["n"] == 4
    assert summary["baseline"]["refusal_rate"] == 1.0
    assert summary["baseline"]["n"] == 6


def test_harmful_rate_handles_unparsed():
    rows = [_row(harmful=True), {"judge_flat": None}, _row(harmful=False)]
    assert math.isclose(harmful_rate(rows), 0.5)         # only parsed rows count
