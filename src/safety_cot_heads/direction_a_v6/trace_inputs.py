"""Finding-2 trace-judge input builders (pathway prefixes + indexed SR).

These fix P0.4: the pathway judge prompt scores a *cumulative reasoning prefix*
and the safety-reasoning judge prompt expects *indexed sentences*, but the first
v6 runner fed both a single raw ``trace_text`` blob. This module builds the
structures those prompts actually require, operating on the **canonical
``trace_text`` only** (never the final answer), so the v6 trace-only principle
is preserved.

* :func:`build_pathway_prefix_rows` — one judge input per cumulative prefix
  ``s_1``, ``s_1 s_2``, ..., over the trace's sentences. IDs and trajectory
  fields match the v5 pathway aggregation convention
  (``<parent>::p<idx>::think``) so downstream trajectory metrics join.
* :func:`build_indexed_sr_row` — a single judge input per completion whose text
  is the indexed-sentence format ``[global=k section=cot index=k] <sentence>``
  the SR prompt parses, plus ``n_trace_segments`` for the position/extent
  normalisation.

The final answer is deliberately never appended: pathway prefixes stay within
the reasoning trace, and the SR index covers only trace (``cot``) sentences.
"""

from __future__ import annotations

from typing import Optional

from ..analysis.trajectory import split_sentences


def trace_sentences(trace_text: Optional[str]) -> list[str]:
    """Sentence-segment a canonical trace (empty/None -> [])."""
    return split_sentences(trace_text or "")


def build_pathway_prefix_rows(parsed_row: dict) -> list[dict]:
    """Expand one explicit-trace completion into cumulative-prefix pathway inputs.

    Each returned row matches the judge driver contract (``id``, ``prompt``,
    ``completion``) and carries the trajectory metadata the v5 pathway
    aggregation consumes. Non-explicit or empty traces yield ``[]`` (they have
    no reasoning trajectory to score).
    """
    if not parsed_row.get("has_explicit_trace"):
        return []
    sents = trace_sentences(parsed_row.get("trace_text"))
    if not sents:
        return []
    parent = str(parsed_row.get("id"))
    base = {
        "parent_id": parent,
        "prompt": parsed_row.get("prompt") or "",
        "model": parsed_row.get("model"),
        "dataset": parsed_row.get("dataset"),
        "condition": parsed_row.get("condition"),
        "seed": parsed_row.get("seed"),
        "category": parsed_row.get("category"),
        "traj_segments_kind": "r1",
        "traj_n_think_sentences": len(sents),
        "traj_n_prose_sentences": 0,
    }
    rows = []
    cum = ""
    for k, s in enumerate(sents):
        cum = f"{cum} {s}".strip() if cum else s
        rows.append({
            "id": f"{parent}::p{k:03d}::think",
            "completion": cum,
            "traj_prefix_idx": k,
            "traj_prefix_kind": "think",
            "traj_is_answer": False,
            **base,
        })
    return rows


def build_indexed_sr_row(parsed_row: dict) -> Optional[dict]:
    """Build the single indexed-sentence SR judge input for one completion.

    Text is ``[global=k section=cot index=k] <sentence>`` over the trace's
    sentences (trace-only: no ``output`` section). Returns ``None`` for
    non-explicit / empty traces.
    """
    if not parsed_row.get("has_explicit_trace"):
        return None
    sents = trace_sentences(parsed_row.get("trace_text"))
    if not sents:
        return None
    entries = [
        {"global_index": k, "section": "cot", "index": k, "text": s}
        for k, s in enumerate(sents)
    ]
    indexed = "\n".join(
        "[global={global_index} section={section} index={index}] {text}".format(**e)
        for e in entries
    )
    parent = str(parsed_row.get("id"))
    return {
        "id": parent,
        "parent_id": parent,
        "prompt": parsed_row.get("prompt") or "",
        "completion": indexed,
        "model": parsed_row.get("model"),
        "dataset": parsed_row.get("dataset"),
        "condition": parsed_row.get("condition"),
        "seed": parsed_row.get("seed"),
        "category": parsed_row.get("category"),
        "n_cot_sentences": len(sents),
        "n_output_sentences": 0,
        "n_trace_segments": len(entries),
        "trace_segments": entries,
    }
