"""Judge output parsing tests."""
from __future__ import annotations

from safety_cot_heads.judging import normalize_safety_labels, parse_judge_json


_VALID = (
    '{"behavior_assignment": {"safe_rejection": true, "harmful_response": false,'
    ' "reasoning_about_safety": false, "adding_intention": false, "changing_subject": false},'
    ' "behavior_assignment_certainty": {"safe_rejection": 0.9, "harmful_response": 0.1,'
    ' "reasoning_about_safety": 0.1, "adding_intention": 0.05, "changing_subject": 0.0}}'
)


def test_parse_clean_json():
    r = parse_judge_json(_VALID)
    assert r["parse_status"] == "ok"
    assert r["data"]["behavior_assignment"]["safe_rejection"] is True


def test_parse_fenced_json():
    r = parse_judge_json("```json\n" + _VALID + "\n```")
    assert r["parse_status"] in ("ok", "recovered")
    assert r["data"]["behavior_assignment"]["safe_rejection"] is True


def test_parse_with_trailing_commentary():
    r = parse_judge_json("Sure! Here:\n" + _VALID + "\nHope this helps.")
    assert r["parse_status"] == "recovered"
    assert r["data"] is not None


def test_parse_empty_fails():
    r = parse_judge_json("")
    assert r["parse_status"] == "failed"
    assert r["data"] is None


def test_parse_garbage_fails():
    r = parse_judge_json("not json at all { broken")
    assert r["parse_status"] == "failed"


def test_normalize_accepts_spaced_keys():
    raw = (
        '{"behavior assignment": {"safe_rejection": "TRUE", "harmful_response": "false",'
        ' "reasoning_about_safety": false, "adding_intention": false, "changing_subject": false}}'
    )
    r = parse_judge_json(raw)
    assert r["parse_status"] == "ok"
    flat = normalize_safety_labels(r["data"])
    assert flat is not None
    assert flat["labels"]["safe_rejection"] is True
    assert flat["labels"]["harmful_response"] is False
