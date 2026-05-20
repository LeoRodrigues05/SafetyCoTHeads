"""Manual validation utilities.

Provides a tiny helper to sample N rows per condition into a CSV for human
spot-checking, plus an agreement metric between a human-labels CSV and the
machine judge output.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Iterable

from .judge_prompts import LABELS


def sample_for_human_review(rows: Iterable[dict], n_per_condition: int = 20,
                             seed: int = 0) -> list[dict]:
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r.get("condition", "unknown"), []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for cond, items in by_cond.items():
        rng.shuffle(items)
        out.extend(items[:n_per_condition])
    return out


def export_review_csv(rows: list[dict], path: str | Path) -> None:
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "condition", "prompt", "completion",
                     "judge_harmful_response", "judge_safe_rejection",
                     "human_label", "notes"])
        for r in rows:
            flat = (r.get("judge_flat") or {}).get("labels") or {}
            w.writerow([
                r.get("id"), r.get("condition"),
                r.get("prompt"), r.get("completion"),
                flat.get("harmful_response"), flat.get("safe_rejection"),
                "", "",
            ])


def agreement(human_csv: str | Path, judge_rows: list[dict]) -> dict:
    """Compute simple agreement between a human-labelled CSV (column ``human_label``
    ∈ {"harmful", "safe"}) and the binary harmful/safe judge output."""
    import csv
    human = {}
    with open(human_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = (r.get("human_label") or "").strip().lower()
            if lab in ("harmful", "safe"):
                human[r["id"]] = lab
    by_id = {r["id"]: r for r in judge_rows}
    matched = total = 0
    for rid, hlab in human.items():
        jr = by_id.get(rid)
        if not jr or not jr.get("judge_flat"):
            continue
        total += 1
        lab = jr["judge_flat"]["labels"]
        judge_lab = "harmful" if lab["harmful_response"] else "safe"
        if judge_lab == hlab:
            matched += 1
    return {"n": total, "agreement": (matched / total) if total else 0.0}
