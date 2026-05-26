"""Coherence / gibberish detection.

Wraps `madhurjindal/autonlp-Gibberish-Detector-492513457` which is used in the
legacy notebook pipeline to flag broken / degenerate completions so that the
"harmful response" rate can be reported conditional on the model still producing
coherent text. The detector returns one of four classes:

    {"clean", "mild gibberish", "word salad", "noise"}

We expose ``classify_gibberish`` for a list of strings, returning a list of
``{"label", "score", "is_clean"}`` dicts. Also exposes light-weight
coherence diagnostics (n-gram repetition, compression ratio, empty rate)
that don't need a model.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence


_GIBBERISH_MODEL_ID = "madhurjindal/autonlp-Gibberish-Detector-492513457"


@dataclass
class CoherenceConfig:
    model_id: str = _GIBBERISH_MODEL_ID
    batch_size: int = 16
    max_length: int = 512
    device: str | int | None = None  # None -> cuda if available


def _resolve_device(device):
    if device is not None:
        return device
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def classify_gibberish(texts: Sequence[str],
                       cfg: CoherenceConfig | None = None) -> list[dict]:
    """Return per-row ``{label, score, is_clean}`` for each completion.

    Empty strings are forced to ``noise`` without invoking the model.
    """
    cfg = cfg or CoherenceConfig()
    from transformers import pipeline

    out: list[dict] = []
    # mask empties; they'd otherwise crash the pipeline
    nonempty_idx = [i for i, t in enumerate(texts) if (t or "").strip()]
    nonempty = [texts[i][: cfg.max_length * 4] for i in nonempty_idx]
    preds: list[dict] = []
    if nonempty:
        clf = pipeline(
            "text-classification",
            model=cfg.model_id,
            tokenizer=cfg.model_id,
            device=_resolve_device(cfg.device),
            truncation=True,
            max_length=cfg.max_length,
        )
        preds = clf(nonempty, batch_size=cfg.batch_size)

    pred_by_idx = dict(zip(nonempty_idx, preds))
    for i in range(len(texts)):
        if i in pred_by_idx:
            p = pred_by_idx[i]
            lab = str(p.get("label", "")).lower()
            out.append({
                "label": lab,
                "score": float(p.get("score", 0.0)),
                "is_clean": lab == "clean",
            })
        else:
            out.append({"label": "noise", "score": 1.0, "is_clean": False})
    return out


# --- model-free diagnostics ---------------------------------------------------


def ngram_repeat_fraction(text: str, n: int = 3) -> float:
    """Fraction of n-grams that are repeats. 0 = no repeats, 1 = all repeats."""
    toks = (text or "").split()
    if len(toks) <= n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def compression_ratio(text: str) -> float:
    """zlib(text)/len(text). Low values => high repetition / low entropy."""
    if not text:
        return 1.0
    b = text.encode("utf-8", errors="ignore")
    return len(zlib.compress(b)) / max(1, len(b))


def coherence_diagnostics(texts: Iterable[str]) -> list[dict]:
    out = []
    for t in texts:
        t = t or ""
        out.append({
            "len_chars": len(t),
            "len_words": len(t.split()),
            "is_empty": not t.strip(),
            "starts_alpha": bool(t.strip()[:1].isalpha()) if t.strip() else False,
            "repeat3": ngram_repeat_fraction(t, 3),
            "compression_ratio": compression_ratio(t),
        })
    return out
