"""MaliciousInstruct — discovery split for SHIPS/Sahara.

Vendored from ``SafetyHeadAttribution/exp_data/maliciousinstruct.csv`` (column
``input``).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from .loaders import _id, _read_csv_column, data_root


def load_maliciousinstruct(n: Optional[int] = None,
                            root: Optional[str | Path] = None) -> list[dict]:
    path = data_root(root) / "sha" / "maliciousinstruct.csv"
    prompts = _read_csv_column(path, "input")
    if n is not None:
        prompts = prompts[:n]
    return [
        {"id": _id("malicious", i), "prompt": p, "dataset": "maliciousinstruct"}
        for i, p in enumerate(prompts)
    ]
