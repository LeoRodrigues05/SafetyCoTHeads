"""JailbreakBench — held-out eval split.

Vendored from ``SafetyHeadAttribution/exp_data/jailbreakbench.csv`` (column
``input``).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from .loaders import _id, _read_csv_column, data_root


def load_jailbreakbench(n: Optional[int] = None,
                         root: Optional[str | Path] = None) -> list[dict]:
    path = data_root(root) / "sha" / "jailbreakbench.csv"
    prompts = _read_csv_column(path, "input")
    if n is not None:
        prompts = prompts[:n]
    return [
        {"id": _id("jbb", i), "prompt": p, "dataset": "jailbreakbench"}
        for i, p in enumerate(prompts)
    ]
