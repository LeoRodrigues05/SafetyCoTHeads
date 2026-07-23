"""Tests for the Finding-2 trace-judge input builders (P0.4)."""

from safety_cot_heads.direction_a_v6.trace_inputs import (
    build_pathway_prefix_rows, build_indexed_sr_row, trace_sentences,
)


def _row(trace, has_explicit=True, ans="final answer here."):
    return {"id": "jbb-1", "prompt": "q", "model": "m", "dataset": "jbb",
            "condition": "baseline", "seed": "seed0", "category": None,
            "trace_text": trace, "answer_text": ans,
            "has_explicit_trace": has_explicit}


def test_pathway_prefixes_are_cumulative_and_within_trace():
    r = _row("First sentence. Second one. Third bit.")
    rows = build_pathway_prefix_rows(r)
    assert len(rows) == 3
    assert rows[0]["completion"] == "First sentence."
    assert rows[1]["completion"] == "First sentence. Second one."
    assert rows[2]["completion"].endswith("Third bit.")
    # ids + trajectory metadata match the v5 aggregation convention
    assert rows[0]["id"] == "jbb-1::p000::think"
    assert all(x["parent_id"] == "jbb-1" for x in rows)
    assert [x["traj_prefix_idx"] for x in rows] == [0, 1, 2]
    assert all(x["traj_prefix_kind"] == "think" for x in rows)
    # the final answer must never leak into a prefix
    assert all("final answer here" not in x["completion"] for x in rows)


def test_pathway_empty_or_nonexplicit_yields_nothing():
    assert build_pathway_prefix_rows(_row("", has_explicit=True)) == []
    assert build_pathway_prefix_rows(_row("Some trace.", has_explicit=False)) == []


def test_indexed_sr_format_and_segment_count():
    r = _row("Alpha reasoning. Beta reasoning.")
    x = build_indexed_sr_row(r)
    assert x is not None
    assert x["n_trace_segments"] == 2
    assert x["n_output_sentences"] == 0        # trace-only: no output section
    assert "[global=0 section=cot index=0] Alpha reasoning." in x["completion"]
    assert "[global=1 section=cot index=1] Beta reasoning." in x["completion"]
    # answer must not appear in the indexed trace
    assert "final answer here" not in x["completion"]
    # every declared segment index is present and in range
    idxs = [e["global_index"] for e in x["trace_segments"]]
    assert idxs == [0, 1]


def test_indexed_sr_nonexplicit_is_none():
    assert build_indexed_sr_row(_row("x.", has_explicit=False)) is None
    assert build_indexed_sr_row(_row("", has_explicit=True)) is None


def test_three_sentence_trace_gives_three_prefixes_one_sr_all_in_range():
    r = _row("One. Two. Three.")
    prefixes = build_pathway_prefix_rows(r)
    sr = build_indexed_sr_row(r)
    assert len(prefixes) == 3
    assert sr["n_trace_segments"] == 3
    assert all(0 <= e["index"] < 3 for e in sr["trace_segments"])
