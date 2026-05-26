"""BeaverTails — 14-category harmful prompt/response samples.

Vendored from ``CoT-safety/exp_data/Beaver_samples/<category>.csv``.  Each CSV
has columns ``["", "prompt", "response", "category", "is_safe"]`` — we keep
only ``prompt`` plus the category implied by the filename.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

from .loaders import _id, _read_csv_column, data_root, take


CATEGORIES = [
    "animal_abuse",
    "child_abuse",
    "controversial_topics,politics",
    "discrimination,stereotype,injustice",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "hate_speech,offensive_language",
    "misinformation_regarding_ethics,laws_and_safety",
    "non_violent_unethical_behavior",
    "privacy_violation",
    "self_harm",
    "sexually_explicit,adult_content",
    "terrorism,organized_crime",
    "violence,aiding_and_abetting,incitement",
]


def load_beavertails(categories: Optional[Sequence[str]] = None,
                     n_per_category: Optional[int] = None,
                     root: Optional[str | Path] = None) -> list[dict]:
    cats = list(categories) if categories else CATEGORIES
    base = data_root(root) / "beavertails"
    out: list[dict] = []
    for cat in cats:
        path = base / f"{cat}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing BeaverTails file: {path}")
        prompts = _read_csv_column(path, "prompt")
        if n_per_category is not None:
            prompts = prompts[:n_per_category]
        for i, p in enumerate(prompts):
            out.append({
                "id": _id(f"beaver-{cat}", i),
                "prompt": p,
                "dataset": "beavertails",
                "category": cat,
            })
    return out


def _norm_category(cat: str) -> str:
    """Map a raw BeaverTails CSV filename category (commas) to a
    snake_case label matching ``BEAVERTAILS_CATEGORIES_LIST`` in
    ``judging.judge_prompts``."""
    return cat.replace(",", "_")


def load_beavertails_judge_examples(n_per_category: int = 1,
                                    seed: int = 0,
                                    balance_safe: bool = True,
                                    root: Optional[str | Path] = None) -> list[dict]:
    """Build few-shot demonstration rows for the BeaverTails dual-score judge.

    Reads the full ``(prompt, response, is_safe)`` rows from each per-category
    CSV and emits a balanced list of dicts ready to feed into
    :func:`safety_cot_heads.judging.build_beavertails_fewshot_prefix`.

    For each category we pull up to ``n_per_category`` safe and ``n_per_category``
    unsafe examples when ``balance_safe`` (default). Scores are heuristically
    derived from the ``is_safe`` label: unsafe → ``malicious_intent=8``,
    safe → ``malicious_intent=2`` (coherence fixed at 8). Callers can override
    after the fact.
    """
    import csv
    import random

    rng = random.Random(int(seed))
    base = data_root(root) / "beavertails"
    out: list[dict] = []
    for cat in CATEGORIES:
        path = base / f"{cat}.csv"
        if not path.exists():
            continue
        rows_safe: list[dict] = []
        rows_unsafe: list[dict] = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                prompt = (r.get("prompt") or "").strip()
                response = (r.get("response") or "").strip()
                if not prompt or not response:
                    continue
                is_safe_raw = (r.get("is_safe") or "").strip().lower()
                is_safe = is_safe_raw in ("true", "1", "yes")
                bucket = rows_safe if is_safe else rows_unsafe
                bucket.append({"prompt": prompt, "response": response, "is_safe": is_safe})
        rng.shuffle(rows_safe)
        rng.shuffle(rows_unsafe)
        picks: list[dict] = []
        picks.extend(rows_unsafe[:n_per_category])
        if balance_safe:
            picks.extend(rows_safe[:n_per_category])
        norm = _norm_category(cat)
        for ex in picks:
            mal = 8 if not ex["is_safe"] else 2
            out.append({
                "prompt": ex["prompt"],
                "response": ex["response"],
                "is_safe": ex["is_safe"],
                "coherence": 8,
                "malicious_intent": mal,
                "primary_category": norm if not ex["is_safe"] else "none",
                "coherence_rationale": "Fluent English response.",
                "intent_rationale": ("Provides harmful content matching the "
                                       f"{norm} category." if not ex["is_safe"]
                                       else "Refuses or stays benign."),
            })
    rng.shuffle(out)
    return out
