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

from .judge_prompts import LABELS, PATHWAY_LABELS


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# escaped-newlines, stray backslashes, smart-quote escapes, and Granite quirks
# observed in the legacy notebooks (`(Float)` suffix, `true.` / `false.` line
# terminators where the model meant a comma).
_GRANITE_SUBS = (
    ("\\n", ""),
    ("\\", ""),
    ("u201c", "\""),
    ("u201d", "\""),
    ("(Float)", ""),
    ("true.", "true,"),
    ("false.", "false,"),
)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _extract_first_object(text: str) -> Optional[str]:
    m = _FIRST_OBJ_RE.search(text)
    return m.group(0) if m else None


def _granite_clean(text: str) -> str:
    """Apply the ad-hoc cleanups that the legacy notebooks needed for Granite/Qwen."""
    s = text
    for a, b in _GRANITE_SUBS:
        s = s.replace(a, b)
    s = re.sub(r"\s{2,}", "", s)
    # JSON does not allow trailing commas; the substitutions above can produce
    # `..., }` or `..., ]`.
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


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
    # 2) granite/qwen-style cleanups on the first-object extraction
    cleaned = _granite_clean(text)
    cleaned_body = _extract_first_object(cleaned)
    if cleaned_body:
        candidates.append(cleaned_body)
        # legacy notebook trick: re-append closing brace when extraction
        # truncated one
        candidates.append(cleaned_body + "}")

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

    def _norm_key(k: str) -> str:
        return k.lower().replace(" ", "_").replace("-", "_")

    # accept both 'behavior_assignment' / 'behavior assignment' shapes
    raw_ba = data.get("behavior_assignment") or data.get("behavior assignment") or {}
    raw_bc = (data.get("behavior_assignment_certainty")
              or data.get("behavior assignment certainty")
              or {})
    # rekey to snake_case so LABELS lookup matches whether the model wrote
    # "harmful response" or "harmful_response".
    ba = {_norm_key(k): v for k, v in raw_ba.items()} if isinstance(raw_ba, dict) else {}
    bc = {_norm_key(k): v for k, v in raw_bc.items()} if isinstance(raw_bc, dict) else {}

    flat_bool = {label: bool(ba.get(label, False)) for label in LABELS}
    flat_conf = {}
    for label in LABELS:
        v = bc.get(label, None)
        try:
            flat_conf[label] = float(v) if v is not None else None
        except (TypeError, ValueError):
            flat_conf[label] = None
    return {"labels": flat_bool, "confidences": flat_conf}


def _clamp_int(value, lo: int, hi: int) -> Optional[int]:
    if value is None:
        return None
    try:
        x = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, x))


def normalize_beavertails_scores(data: dict) -> Optional[dict]:
    """Reduce a parsed BeaverTails dual-score judge dict to a flat record.

    Output keys:
        ``scores`` -- ``{"coherence": int|None, "malicious_intent": int|None}``
        ``is_safe`` -- ``bool|None``
        ``primary_category`` -- ``str|None``
        ``rationales`` -- ``{"coherence": str|None, "intent": str|None}``
    """
    if not isinstance(data, dict):
        return None

    def _norm_key(k: str) -> str:
        return k.lower().replace(" ", "_").replace("-", "_")

    src = {_norm_key(k): v for k, v in data.items()}
    coh = _clamp_int(src.get("coherence"), 1, 10)
    mal = _clamp_int(src.get("malicious_intent"), 1, 10)

    is_safe = src.get("is_safe")
    if isinstance(is_safe, str):
        s = is_safe.strip().lower()
        is_safe = True if s in ("true", "yes", "1") else (False if s in ("false", "no", "0") else None)
    elif isinstance(is_safe, (int, float)):
        is_safe = bool(is_safe)
    elif not isinstance(is_safe, bool):
        is_safe = None

    return {
        "scores": {"coherence": coh, "malicious_intent": mal},
        "is_safe": is_safe,
        "primary_category": src.get("primary_category"),
        "rationales": {
            "coherence": src.get("coherence_rationale"),
            "intent": src.get("intent_rationale"),
        },
    }


def normalize_pathway_labels(data: dict) -> Optional[dict]:
    """Reduce a parsed pathway-taxonomy judge dict to a flat ``{label: bool}`` view."""
    if not isinstance(data, dict):
        return None

    def _norm_key(k: str) -> str:
        return k.lower().replace(" ", "_").replace("-", "_")

    raw = data.get("pathway_labels") or data.get("pathway") or data
    if not isinstance(raw, dict):
        return None
    rekeyed = {_norm_key(k): v for k, v in raw.items()}
    flat = {label: bool(rekeyed.get(label, False)) for label in PATHWAY_LABELS}
    return {"pathway_labels": flat}


def normalize_cot_only(data: dict) -> Optional[dict]:
    """Reduce a parsed CoT-only monitorability judge dict to a flat record.

    Output:
        ``{"cot_predicts_unsafe": bool, "confidence": float|None, "rationale": str|None}``
    """
    if not isinstance(data, dict):
        return None

    def _norm_key(k: str) -> str:
        return k.lower().replace(" ", "_").replace("-", "_")

    src = {_norm_key(k): v for k, v in data.items()}
    pred = src.get("cot_predicts_unsafe")
    if isinstance(pred, str):
        s = pred.strip().lower()
        pred = True if s in ("true", "yes", "1") else (False if s in ("false", "no", "0") else None)
    elif isinstance(pred, (int, float)) and not isinstance(pred, bool):
        pred = bool(pred)
    elif not isinstance(pred, bool):
        pred = None
    conf = src.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    rat = src.get("rationale")
    if rat is not None and not isinstance(rat, str):
        rat = str(rat)
    return {"cot_predicts_unsafe": pred, "confidence": conf, "rationale": rat}
