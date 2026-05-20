"""Dataset loaders for the safety_cot_heads project.

All loaders return a list of dicts with at minimum the keys
``{"id", "prompt"}`` (plus optional category / label metadata).  This is the
single in-memory schema used by attribution, generation and judging modules.

Raw files live under ``<repo>/data/raw/`` and are vendored from the upstream
``SafetyHeadAttribution/exp_data/`` and ``CoT-safety/exp_data/Beaver_samples/``
directories — see ``MIGRATION.md``.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

# The package layout is:  <root>/safety_cot_heads/data/raw/...
# This file lives at:     <root>/safety_cot_heads/src/safety_cot_heads/data/loaders.py
_DATA_ROOT_DEFAULT = Path(__file__).resolve().parents[3] / "data" / "raw"


def data_root(override: str | Path | None = None) -> Path:
    return Path(override) if override else _DATA_ROOT_DEFAULT


# ---------------------------------------------------------------------------
def _read_csv_column(path: Path, column: str) -> list[str]:
    import pandas as pd
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"column {column!r} not in {path} (have: {list(df.columns)})")
    return [str(v) for v in df[column].dropna().tolist()]


def _id(prefix: str, i: int) -> str:
    return f"{prefix}-{i:05d}"


def take(rows: list[dict], n: int | None) -> list[dict]:
    return rows if n is None else rows[:n]
