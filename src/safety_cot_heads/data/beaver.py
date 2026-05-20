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
