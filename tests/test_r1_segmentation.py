"""R1-Distill segmenter edge-case tests (P1.4).

Pins behaviour of :func:`segment_completion` and :func:`build_prefix_rows`
on the malformed/edge-case ``<think>`` patterns that show up in real
DeepSeek-R1-Distill-Llama-8B output (chat-template-prefilled open tag,
mid-stream close-tag truncation, nested-looking content, no close tag at all).

These tests are intentionally CPU-only — they exercise pure parsing logic
in :mod:`safety_cot_heads.direction_a.segmentation`.
"""
from __future__ import annotations

from safety_cot_heads.direction_a.segmentation import (
    Segments,
    build_prefix_rows,
    segment_completion,
)


def test_prose_completion_marks_kind_prose() -> None:
    seg = segment_completion("Hello world. This is a refusal. I will not help.")
    assert seg.kind == "prose"
    assert seg.think_body == ""
    assert seg.answer_body.startswith("Hello world")
    assert len(seg.answer_sentences) == 3
    # Prefixes are cumulative.
    assert [p["idx"] for p in seg.prefixes] == [0, 1, 2]
    assert all(p["kind"] == "prose" for p in seg.prefixes)
    assert seg.prefixes[-1]["is_answer"] is True
    assert seg.prefixes[0]["text"] == "Hello world."
    assert seg.prefixes[2]["text"].endswith("I will not help.")


def test_r1_full_think_block_then_answer() -> None:
    text = "<think>First step. Second step.</think>Final answer here."
    seg = segment_completion(text)
    assert seg.kind == "r1"
    assert seg.think_body == "First step. Second step."
    assert seg.answer_body == "Final answer here."
    assert len(seg.think_sentences) == 2
    # 2 think prefixes + 1 final-answer prefix that spans the whole completion
    assert len(seg.prefixes) == 3
    assert [p["kind"] for p in seg.prefixes] == ["think", "think", "answer"]
    assert seg.prefixes[-1]["is_answer"] is True
    assert seg.prefixes[-1]["text"] == text   # answer prefix is whole completion


def test_r1_missing_open_tag_chat_template_prefilled() -> None:
    """Some R1 chat templates pre-fill <think> in the *prompt* — completion
    starts mid-think and only contains </think>."""
    text = "First step. Second step.</think>Done."
    seg = segment_completion(text)
    assert seg.kind == "r1"
    assert seg.think_body == "First step. Second step."
    assert seg.answer_body == "Done."
    assert len(seg.think_sentences) == 2
    assert seg.prefixes[-1]["is_answer"] is True


def test_r1_no_close_tag_treated_as_prose() -> None:
    """Generation truncated before </think> → falls back to prose path
    (we cannot judge a half-open think block sensibly)."""
    text = "<think>Reasoning step one. Reasoning step two without close tag."
    seg = segment_completion(text)
    assert seg.kind == "prose"
    # The open tag is preserved in the prose stream; we don't try to recover.
    assert "<think>" in seg.full_completion
    assert seg.think_body == ""


def test_r1_empty_answer_body_still_emits_answer_prefix() -> None:
    """Model closes </think> but produces no answer text. We still emit
    a final answer-prefix so the trajectory pipeline has the same shape."""
    text = "<think>Step one. Step two.</think>"
    seg = segment_completion(text)
    assert seg.kind == "r1"
    assert seg.answer_body == ""
    assert seg.prefixes[-1]["kind"] == "answer"
    assert seg.prefixes[-1]["is_answer"] is True


def test_r1_case_insensitive_think_tags() -> None:
    text = "<THINK>Step one.</Think>Answer."
    seg = segment_completion(text)
    assert seg.kind == "r1"
    assert seg.think_body == "Step one."
    assert seg.answer_body == "Answer."


def test_empty_completion_does_not_crash() -> None:
    seg = segment_completion("")
    assert seg.kind == "prose"
    assert seg.prefixes == []


def test_build_prefix_rows_carries_metadata() -> None:
    gen_rows = [
        {
            "id": "g_001",
            "prompt": "Question?",
            "completion": "<think>Thought one. Thought two.</think>Final answer.",
            "dataset": "jbb",
            "category": "harm",
            "condition": "baseline",
            "model": "r1distill",
            "seed": 0,
        }
    ]
    rows = build_prefix_rows(gen_rows)
    # 2 think prefixes + 1 final-answer prefix
    assert len(rows) == 3
    for r in rows:
        assert r["parent_id"] == "g_001"
        assert r["prompt"] == "Question?"
        assert r["traj_segments_kind"] == "r1"
        assert r["traj_n_think_sentences"] == 2
        assert r["dataset"] == "jbb"
        assert r["seed"] == 0
    # IDs are namespaced and ordered.
    assert rows[0]["id"] == "g_001::p000::think"
    assert rows[1]["id"] == "g_001::p001::think"
    assert rows[2]["id"] == "g_001::p002::answer"
    assert rows[0]["traj_is_answer"] is False
    assert rows[-1]["traj_is_answer"] is True


def test_build_prefix_rows_prose_input() -> None:
    gen_rows = [
        {
            "id": "g_002",
            "prompt": "Q",
            "completion": "Sentence A. Sentence B.",
            "model": "llama31",
            "seed": 1,
        }
    ]
    rows = build_prefix_rows(gen_rows)
    assert len(rows) == 2
    assert all(r["traj_segments_kind"] == "prose" for r in rows)
    assert all(r["traj_prefix_kind"] == "prose" for r in rows)
    assert rows[-1]["traj_is_answer"] is True
