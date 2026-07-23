"""Tests for the canonical v6 answer/trace parser.

Covers the required cases: normal explicit traces, absent opening tags, absent
closing tags, multiple tags, no tags, empty trace, empty answer, and literal
uses of the word "think" that are not tags. Also asserts the two safety
invariants: an explicit-model answer input carries no hidden trace, and an
explicit-model trace input carries no final answer.
"""

import pytest

from safety_cot_heads.direction_a_v6.parsing import parse_completion


def test_well_formed_explicit_trace():
    p = parse_completion("<think>secret reasoning here</think>\n\nFinal safe answer.")
    assert p.trace_kind == "explicit"
    assert p.parse_status == "ok"
    assert p.has_explicit_trace
    assert p.trace_text == "secret reasoning here"
    assert p.answer_text == "Final safe answer."
    # invariants: no leak either direction
    assert "secret reasoning" not in p.answer_text
    assert "Final safe answer" not in p.trace_text


def test_closing_tag_without_opening():
    # Dominant real shape: <think> pre-filled in the prompt, only </think> emitted.
    p = parse_completion("reasoning about the request\n</think>\nI cannot help with that.")
    assert p.trace_kind == "explicit"
    assert p.parse_status == "nonstandard_closing_only"
    assert p.has_explicit_trace
    assert p.trace_text == "reasoning about the request"
    assert p.answer_text == "I cannot help with that."


def test_opening_tag_without_closing_is_malformed():
    p = parse_completion("<think>the model was cut off mid reasoning")
    assert p.trace_kind == "malformed_explicit"
    assert p.parse_status == "malformed_explicit"
    assert p.has_explicit_trace
    # hidden reasoning must NOT become the answer
    assert p.answer_text == ""
    assert p.answer_is_empty
    assert p.trace_text == "the model was cut off mid reasoning"


def test_multiple_closing_tags_split_at_last():
    p = parse_completion("first block</think>middle</think>the real answer")
    assert p.trace_kind == "explicit"
    assert p.parse_status == "nonstandard_multiple_close"
    assert p.answer_text == "the real answer"
    # everything before the last </think> is trace; no answer leaks in
    assert "the real answer" not in p.trace_text
    assert "middle" in p.trace_text


def test_no_tags_prose_prefix():
    p = parse_completion("I will help. Here are the steps. Step one is safe.")
    assert p.trace_kind == "prose_prefix"
    assert not p.has_explicit_trace
    # whole response remains the answer
    assert p.answer_text.startswith("I will help.")
    assert p.answer_text.endswith("Step one is safe.")
    # explicit trace is empty; prefix carried separately
    assert p.trace_text == ""
    assert p.trace_is_empty
    assert p.prose_prefix_text == "I will help. Here are the steps."


def test_no_tags_single_sentence_is_none():
    p = parse_completion("Just one sentence answer.")
    assert p.trace_kind == "none"
    assert p.parse_status == "prose_only"
    assert p.prose_prefix_text == ""
    assert p.answer_text == "Just one sentence answer."


def test_empty_trace():
    p = parse_completion("<think></think>the answer")
    assert p.trace_kind == "explicit"
    assert p.trace_is_empty
    assert p.answer_text == "the answer"


def test_empty_answer_after_close():
    p = parse_completion("all reasoning</think>   ")
    assert p.trace_kind == "explicit"
    assert p.answer_is_empty
    assert p.trace_text == "all reasoning"


def test_literal_word_think_is_not_a_tag():
    p = parse_completion("I think this is fine. I think we should proceed. Final call.")
    # no angle-bracket tags -> prose, not explicit
    assert not p.has_explicit_trace
    assert p.trace_kind == "prose_prefix"
    assert "I think this is fine." in p.answer_text


def test_none_and_empty_input():
    p = parse_completion(None)
    assert p.answer_is_empty
    assert p.trace_kind == "none"
    p2 = parse_completion("")
    assert p2.answer_is_empty
    assert p2.trace_kind == "none"


def test_tag_with_spaces_and_case():
    p = parse_completion("reasoning< / THINK >answer body")
    assert p.trace_kind == "explicit"
    assert p.answer_text == "answer body"
    assert p.trace_text == "reasoning"


@pytest.mark.parametrize("completion", [
    "<think>r</think>a",
    "r</think>a",
    "a b. c d. e f.",
])
def test_answer_never_contains_trace_markers(completion):
    p = parse_completion(completion)
    # the parsed answer must never still contain a </think> marker
    assert "</think>" not in p.answer_text.lower()
