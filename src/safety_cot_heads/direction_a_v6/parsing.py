"""Canonical answer/trace parser for the Direction A v6 corrected evaluation.

This is the *single* source of truth for splitting a model completion into its
final answer and its reasoning trace. Every downstream v6 evaluator (answer
safety, answer coherence, trace monitor, trace safety-reasoning, pathway) MUST
consume :func:`parse_completion` output rather than re-implementing tag rules.
The v5 code split completions in several places (``direction_a.segmentation``,
the judge drivers) with subtly different rules; v6 funnels everything here.

Design decisions (documented so they are auditable, not silent):

* **Well-formed explicit trace** ``<think>reasoning</think>answer`` ->
  ``trace_text`` is the reasoning inside the tags, ``answer_text`` is the text
  after ``</think>``. ``trace_kind="explicit"``, ``parse_status="ok"``.

* **Closing tag without opening tag** ``reasoning</think>answer``. This is the
  *dominant* real shape in this project: several chat templates pre-fill the
  literal ``<think>`` token into the rendered prompt, so the generation begins
  inside the reasoning block and only the closing tag is emitted. This is an
  intentionally supported form: text before ``</think>`` is the trace, text
  after is the answer. ``trace_kind="explicit"``,
  ``parse_status="nonstandard_closing_only"``.

* **Opening tag without closing tag** ``<think>reasoning`` (truncated). Hidden
  reasoning must NOT be treated as the final answer. We keep the reasoning as
  ``trace_text``, leave ``answer_text`` empty, mark
  ``trace_kind="malformed_explicit"``, and surface the item in diagnostics.

* **Multiple closing tags** — deterministic policy: split at the *last*
  ``</think>``. Everything before it is trace, everything after is answer. This
  guarantees no trace content leaks into ``answer_text`` even when the model
  emits several reasoning blocks. ``parse_status="nonstandard_multiple_close"``.

* **No explicit trace tags** — the whole completion is the answer. No explicit
  trace exists. For a prose model we may additionally construct an
  *early-response prefix* (all sentences except the last) as a monitorability
  *sensitivity* signal only; it is exposed as ``prose_prefix_text`` and
  ``trace_kind="prose_prefix"`` (``has_explicit_trace`` stays False). It is
  never labelled chain-of-thought. If the response is too short to form a
  prefix, ``trace_kind="none"``.

The literal word "think" appearing in prose (e.g. "I think that...") is not a
tag: only the angle-bracket forms ``<think>`` / ``</think>`` are recognised.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Optional

from ..analysis.trajectory import split_sentences

# Angle-bracket think tags only. Case-insensitive; tolerant of attributes/space
# inside the tag (``<think >``) but NOT of the bare word "think".
_THINK_OPEN_RE = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

PARSER_VERSION = "v6.0"

TRACE_KINDS = ("explicit", "prose_prefix", "none", "malformed_explicit")
PARSE_STATUSES = (
    "ok",                          # well-formed explicit OR clean no-tag answer
    "nonstandard_closing_only",    # </think> present, <think> absent (pre-filled)
    "nonstandard_multiple_close",  # more than one </think>
    "malformed_explicit",          # <think> present, </think> absent
    "prose_prefix",                # no tags, prose-prefix sensitivity available
    "prose_only",                  # no tags, response too short for a prefix
)


@dataclass
class ParsedCompletion:
    """Canonical parse of one model completion.

    Fields mirror the task's required parser contract. ``prose_prefix_text`` is
    an *extra* field carrying the heuristic early-response prefix for prose
    models; it is only meaningful when ``trace_kind == "prose_prefix"``.
    """

    full_completion: str
    answer_text: str
    trace_text: str
    trace_kind: str          # one of TRACE_KINDS
    parse_status: str        # one of PARSE_STATUSES
    has_explicit_trace: bool
    answer_is_empty: bool
    trace_is_empty: bool
    prose_prefix_text: str = ""
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _build_prose_prefix(answer_body: str) -> tuple[str, bool]:
    """Return (prefix_text, has_prefix) for a prose completion.

    The prefix is every sentence except the last one. Returns ``("", False)``
    when there are fewer than two sentences (no meaningful early-response
    signal to monitor).
    """
    sents = split_sentences(answer_body or "")
    if len(sents) <= 1:
        return "", False
    return " ".join(sents[:-1]).strip(), True


def parse_completion(completion: Optional[str]) -> ParsedCompletion:
    """Split a raw completion string into canonical answer / trace parts.

    Never raises: malformed inputs are reported via ``parse_status`` /
    ``trace_kind`` rather than by discarding the row.
    """
    text = completion or ""

    close_matches = list(_THINK_CLOSE_RE.finditer(text))
    open_match = _THINK_OPEN_RE.search(text)

    # --- Case A: at least one closing tag -> explicit trace -----------------
    if close_matches:
        # Deterministic policy: split at the LAST closing tag so no trace
        # content can leak past it into the answer.
        last_close = close_matches[-1]
        answer_text = text[last_close.end():].lstrip()
        # Trace is everything before the last close; strip a leading <think>
        # if the model actually emitted one.
        pre = text[: last_close.start()]
        first_open = _THINK_OPEN_RE.search(pre)
        trace_text = pre[first_open.end():] if first_open else pre
        trace_text = trace_text.strip()

        if len(close_matches) > 1:
            status = "nonstandard_multiple_close"
        elif open_match is None:
            status = "nonstandard_closing_only"
        else:
            status = "ok"
        return ParsedCompletion(
            full_completion=text,
            answer_text=answer_text,
            trace_text=trace_text,
            trace_kind="explicit",
            parse_status=status,
            has_explicit_trace=True,
            answer_is_empty=(len(answer_text.strip()) == 0),
            trace_is_empty=(len(trace_text.strip()) == 0),
            prose_prefix_text="",
        )

    # --- Case B: opening tag but no closing tag -> malformed ----------------
    if open_match is not None:
        trace_text = text[open_match.end():].strip()
        # Hidden reasoning must NOT become the answer.
        return ParsedCompletion(
            full_completion=text,
            answer_text="",
            trace_text=trace_text,
            trace_kind="malformed_explicit",
            parse_status="malformed_explicit",
            has_explicit_trace=True,
            answer_is_empty=True,
            trace_is_empty=(len(trace_text.strip()) == 0),
            prose_prefix_text="",
        )

    # --- Case C: no tags at all -> whole completion is the answer -----------
    answer_text = text.strip()
    prefix_text, has_prefix = _build_prose_prefix(answer_text)
    if has_prefix:
        return ParsedCompletion(
            full_completion=text,
            answer_text=answer_text,
            trace_text="",              # NO explicit trace; prefix is separate
            trace_kind="prose_prefix",
            parse_status="prose_prefix",
            has_explicit_trace=False,
            answer_is_empty=(len(answer_text) == 0),
            trace_is_empty=True,        # explicit trace is empty by definition
            prose_prefix_text=prefix_text,
        )
    return ParsedCompletion(
        full_completion=text,
        answer_text=answer_text,
        trace_text="",
        trace_kind="none",
        parse_status="prose_only",
        has_explicit_trace=False,
        answer_is_empty=(len(answer_text) == 0),
        trace_is_empty=True,
        prose_prefix_text="",
    )


def parse_row(row: dict) -> dict:
    """Parse a generation JSONL row, returning it enriched with parse fields.

    Preserves all original keys (``id``, ``prompt``, ``model``, ``condition``,
    ``dataset``, ``seed``, ``decoding``, provenance flags, ...) and adds a
    ``parsed`` sub-dict plus top-level convenience mirrors used by the judge
    input builders. The original ``completion`` is retained verbatim.
    """
    parsed = parse_completion(row.get("completion"))
    out = dict(row)
    out["parsed"] = parsed.to_dict()
    return out
