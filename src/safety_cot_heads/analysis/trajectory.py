"""Sentence-level CoT trajectory analysis.

For each generated completion we split on sentence boundaries (regex
fallback, optional spaCy if available) and compute a per-sentence safety
profile by querying the same judge prompt against the *partial* trajectory
``response[:sentence_i_end]``.

This is the implementation hook for the "Where do you Lie?" trajectory
analyses — given the per-sentence labels we report the first sentence where
``HARMFUL_RESPONSE`` flips to True, the fraction of the trajectory after
that flip, etc.
"""

from __future__ import annotations
import re
from typing import Optional


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        import spacy                                           # optional
        if not hasattr(split_sentences, "_nlp"):
            split_sentences._nlp = spacy.blank("en")
            split_sentences._nlp.add_pipe("sentencizer")
        return [s.text.strip() for s in split_sentences._nlp(text).sents
                if s.text.strip()]
    except Exception:
        pass
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def trajectory_for_row(row: dict) -> list[dict]:
    """Return a per-sentence list of ``{idx, sentence, cumulative_text}``.
    Judging is left to the caller (see :func:`safety_cot_heads.judging.judge_rows`)."""
    sents = split_sentences(row.get("completion", ""))
    out = []
    cum = ""
    for i, s in enumerate(sents):
        cum = (cum + " " + s).strip() if cum else s
        out.append({
            "id": row["id"], "sent_idx": i, "sentence": s,
            "cumulative_text": cum,
            "is_partial": i < len(sents) - 1,
            "prompt": row["prompt"],
        })
    return out


def first_flip_to_harmful(judge_traj_rows: list[dict]) -> Optional[int]:
    """Given judge rows for a single trajectory in order, return the index of
    the first sentence whose cumulative judgement is HARMFUL_RESPONSE=True,
    or None if it never flips."""
    for r in judge_traj_rows:
        flat = r.get("judge_flat")
        if flat and flat["labels"].get("harmful_response"):
            return r["sent_idx"]
    return None
