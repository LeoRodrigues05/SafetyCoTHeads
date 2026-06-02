"""Segmentation + cumulative-prefix construction for trajectory analysis.

Two completion shapes are supported:

1. **Prose CoT** (Llama-3.1-8B-Instruct and similar instruction-tuned models).
   The completion is split into sentences with the regex/spaCy fallback
   inherited from :mod:`safety_cot_heads.analysis.trajectory` and cumulative
   prefixes ``s_1``, ``s_1 + s_2``, ..., ``s_1 + ... + s_N`` are constructed.

2. **R1-Distill** (and any other ``<think>...</think>`` model). The completion
   has structure ``<think>{think_body}</think>{answer_body}``. We segment
   ``think_body`` into sentences, build cumulative prefixes **within the
   think block**, and add a final "full completion" prefix
   (``<think>{think_body}</think>{answer_body}``) so the answer-level judge
   call is recorded too. ``think_idx`` / ``is_answer_prefix`` fields tag each
   prefix.

The returned :class:`Segments` carries enough structure for
:mod:`trajectory_metrics` to compute both prose and R1 metric variants
without re-parsing the completion.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional
import re

from ..analysis.trajectory import split_sentences


# Open / close think tags. We tolerate the open tag being absent (some R1
# chat templates pre-fill it in the prompt rather than the generation).
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


@dataclass
class Segments:
    """Segmented completion + cumulative-prefix metadata.

    Attributes
    ----------
    kind
        Either ``"prose"`` (no think block detected) or ``"r1"``.
    full_completion
        The original completion string, unmodified.
    think_body
        Text between ``<think>`` and ``</think>``. Empty for prose.
    answer_body
        Text after ``</think>``, or the entire completion for prose.
    think_sentences
        Sentence segmentation of ``think_body`` (empty for prose).
    answer_sentences
        Sentence segmentation of ``answer_body`` — used to count the prose-CoT
        sentences in instruction-tuned outputs and also exposed for R1
        downstream analysis. For prose this is the segmentation of the whole
        completion.
    prefixes
        Ordered list of ``{idx, kind, text, is_answer}`` entries describing
        which cumulative texts get judged. For prose, ``kind="prose"`` and
        each ``text`` is the cumulative prefix of sentences up to ``idx``.
        For R1, ``kind="think"`` prefixes cover the think block in order, and
        a single final ``kind="answer"`` prefix carries the entire completion
        (so the judge sees both the reasoning and the final answer).
    """

    kind: str
    full_completion: str
    think_body: str
    answer_body: str
    think_sentences: list[str] = field(default_factory=list)
    answer_sentences: list[str] = field(default_factory=list)
    prefixes: list[dict] = field(default_factory=list)


def _split_think(completion: str) -> tuple[Optional[str], str]:
    """Return ``(think_body, answer_body)`` or ``(None, completion)`` for prose."""
    text = completion or ""
    close = _THINK_CLOSE_RE.search(text)
    if close is None:
        return None, text
    end_pos = close.end()
    answer_body = text[end_pos:].lstrip()
    open_m = _THINK_OPEN_RE.search(text, 0, close.start())
    think_body = text[open_m.end():close.start()] if open_m else text[:close.start()]
    return think_body.strip(), answer_body


def segment_completion(completion: str) -> Segments:
    """Segment a completion into prose-CoT or R1 ``<think>`` form."""
    think_body, answer_body = _split_think(completion or "")
    if think_body is None:
        sents = split_sentences(answer_body)
        cum = ""
        prefixes: list[dict] = []
        for i, s in enumerate(sents):
            cum = f"{cum} {s}".strip() if cum else s
            prefixes.append({"idx": i, "kind": "prose", "text": cum,
                             "is_answer": (i == len(sents) - 1)})
        return Segments(
            kind="prose",
            full_completion=completion or "",
            think_body="",
            answer_body=answer_body,
            think_sentences=[],
            answer_sentences=sents,
            prefixes=prefixes,
        )

    think_sents = split_sentences(think_body)
    ans_sents = split_sentences(answer_body)
    prefixes = []
    cum = ""
    for i, s in enumerate(think_sents):
        cum = f"{cum} {s}".strip() if cum else s
        prefixes.append({"idx": i, "kind": "think", "text": cum, "is_answer": False})
    # final full-completion prefix: judge sees the entire <think>...</think>+answer
    prefixes.append({
        "idx": len(think_sents),
        "kind": "answer",
        "text": completion or "",
        "is_answer": True,
    })
    return Segments(
        kind="r1",
        full_completion=completion or "",
        think_body=think_body,
        answer_body=answer_body,
        think_sentences=think_sents,
        answer_sentences=ans_sents,
        prefixes=prefixes,
    )


def build_prefix_rows(generation_rows: Iterable[dict]) -> list[dict]:
    """Expand generation rows into one row per cumulative prefix.

    The returned rows match the shape consumed by
    :func:`safety_cot_heads.judging.judge_rows` (``id``, ``prompt``,
    ``completion``) but carry trajectory-analysis metadata
    (``traj_*`` fields) so the per-prefix labels can be re-assembled.

    Each returned row has::

        {
          "id": "<gen_id>::p<idx>::<kind>",
          "parent_id": "<gen_id>",
          "prompt": <original user prompt>,
          "completion": <cumulative prefix text>,        # what judge sees
          "dataset": ..., "category": ..., "condition": ..., "model": ...,
          "seed": ...,
          "traj_prefix_idx": <int>,
          "traj_prefix_kind": "prose" | "think" | "answer",
          "traj_is_answer": <bool>,
          "traj_segments_kind": "prose" | "r1",
          "traj_n_prose_sentences": <int>,
          "traj_n_think_sentences": <int>,
        }
    """
    out: list[dict] = []
    for row in generation_rows:
        seg = segment_completion(row.get("completion", ""))
        parent = row["id"]
        base = {
            "parent_id": parent,
            "prompt": row.get("prompt"),
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "condition": row.get("condition"),
            "model": row.get("model"),
            "seed": row.get("seed"),
            "traj_segments_kind": seg.kind,
            "traj_n_prose_sentences": len(seg.answer_sentences),
            "traj_n_think_sentences": len(seg.think_sentences),
        }
        for p in seg.prefixes:
            out.append({
                "id": f"{parent}::p{p['idx']:03d}::{p['kind']}",
                "completion": p["text"],
                "traj_prefix_idx": p["idx"],
                "traj_prefix_kind": p["kind"],
                "traj_is_answer": p["is_answer"],
                **base,
            })
    return out
