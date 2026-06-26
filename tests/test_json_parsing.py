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


def test_parse_recovers_truncated_span_array():
    """A safety-reasoning trace cut off by the token cap mid-object should be
    salvaged down to the last complete span, not dropped as a parse failure."""
    truncated = (
        '{"has_safety_reasoning": true, "safety_reasoning_sentence_indexes": [\n'
        '  {"section": "cot", "index": 1, "global_index": 1,'
        ' "category": "risk_acknowledgment", "confidence": 0.9},\n'
        '  {"section": "cot", "index": 2, "global_index": 2,'
        ' "category": "refusal_reasoning", "confidence": 0.95},\n'
        '  {"section": "output", "index": 3, "global_index'  # cut off here
    )
    r = parse_judge_json(truncated)
    assert r["parse_status"] == "recovered"
    spans = r["data"]["safety_reasoning_sentence_indexes"]
    assert len(spans) == 2  # the trailing incomplete object is dropped
    assert spans[0]["category"] == "risk_acknowledgment"
    assert r["data"]["has_safety_reasoning"] is True


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
