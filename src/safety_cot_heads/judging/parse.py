"""Robust JSON parsing for judge outputs.

The judge model is asked for strict JSON but in practice will sometimes wrap
its output in markdown fences, append commentary, swap True/False casing,
emit Python-list-style strings, etc.  This module isolates the parsing /
normalisation logic so the judge runner can just call :func:`parse_judge_json`
and get either a structured dict + status, or a clear failure marker.
"""

from __future__ import annotations
import json
import re
from typing import Optional

from .judge_prompts import LABELS


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _extract_first_object(text: str) -> Optional[str]:
    m = _FIRST_OBJ_RE.search(text)
    return m.group(0) if m else None


def _normalize_bools(obj):
    """Coerce TRUE/FALSE/yes/no/1/0 strings to bools."""
    if isinstance(obj, dict):
        return {k: _normalize_bools(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_bools(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip().lower()
        if s in ("true", "yes", "1"):  return True
        if s in ("false", "no", "0"):  return False
    return obj


def parse_judge_json(raw: str) -> dict:
    """Parse a judge model response.

    Returns:
        ``{"parse_status": "ok"|"recovered"|"failed",
           "data": dict | None,
           "error": str | None}``
    """
    text = (raw or "").strip()
    if not text:
        return {"parse_status": "failed", "data": None, "error": "empty"}

    # 1) try direct json.loads
    candidates = [text, _strip_fences(text)]
    body = _extract_first_object(text)
    if body:
        candidates.append(body)

    last_err = None
    for i, cand in enumerate(candidates):
        try:
            data = json.loads(cand)
            data = _normalize_bools(data)
            status = "ok" if i == 0 else "recovered"
            return {"parse_status": status, "data": data, "error": None}
        except Exception as e:
            last_err = str(e)

    return {"parse_status": "failed", "data": None, "error": last_err}


def normalize_safety_labels(data: dict) -> Optional[dict]:
    """Reduce a parsed safety-judge dict to a flat ``{label: bool}`` + ``{label: float}`` view."""
    if not isinstance(data, dict):
        return None
    # accept both 'behavior_assignment' / 'behavior assignment' shapes
    ba = data.get("behavior_assignment") or data.get("behavior assignment") or {}
    bc = (data.get("behavior_assignment_certainty")
          or data.get("behavior assignment certainty")
          or {})

    def _key(k):
        return k.lower().replace(" ", "_")

    flat_bool = {label: bool(ba.get(label, ba.get(_key(label), False))) for label in LABELS}
    flat_conf = {}
    for label in LABELS:
        v = bc.get(label, bc.get(_key(label), None))
        try:
            flat_conf[label] = float(v) if v is not None else None
        except (TypeError, ValueError):
            flat_conf[label] = None
    return {"labels": flat_bool, "confidences": flat_conf}
