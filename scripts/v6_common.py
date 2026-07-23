"""Shared discovery + IO helpers for the Direction A v6 corrected rerun.

Everything here is read-only with respect to ``runs/direction_a_v5`` — the v5
tree is immutable source data. Writes go under ``runs/direction_a_v6``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
V5_ROOT = REPO / "runs" / "direction_a_v5"
V6_ROOT = REPO / "runs" / "direction_a_v6"
PAPER_SCOPE = REPO / "configs" / "direction_a_v6" / "paper_scope.yaml"

# Datasets whose generations live under gen/<ds>/ ; judge under judge/<ds>/.
DATASETS = ("jbb", "bt", "xstest")


def utcnow_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_paper_scope() -> dict:
    with open(PAPER_SCOPE) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Cell:
    model: str
    dataset: str
    condition: str
    seed: str = "seed0"

    @property
    def key(self) -> str:
        return f"{self.model}/{self.dataset}/{self.condition}/{self.seed}"

    def gen_dir(self) -> Path:
        return V5_ROOT / self.model / "gen" / self.dataset / self.condition / self.seed

    def judge_dir(self) -> Path:
        return V5_ROOT / self.model / "judge" / self.dataset / self.condition / self.seed

    def v6_parsed_dir(self) -> Path:
        return V6_ROOT / "parsed" / self.model / self.dataset / self.condition / self.seed


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path)) if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    from dataclasses import is_dataclass
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"not serializable: {type(o)}")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_cells(models: Optional[list[str]] = None,
                   datasets: Optional[list[str]] = None) -> list[Cell]:
    """Discover completed (model, dataset, condition, seed) generation cells on disk.

    A cell counts as discovered if its gen dir holds at least one
    ``completions*.jsonl``. This does NOT trust the experiment matrix — it walks
    the actual v5 tree.
    """
    cells: list[Cell] = []
    model_dirs = models or [p.name for p in sorted(V5_ROOT.iterdir())
                            if p.is_dir() and (p / "gen").is_dir() and not p.name.startswith("_")]
    for model in model_dirs:
        gen_root = V5_ROOT / model / "gen"
        if not gen_root.is_dir():
            continue
        for ds in (datasets or DATASETS):
            ds_dir = gen_root / ds
            if not ds_dir.is_dir():
                continue
            for cond_dir in sorted(ds_dir.iterdir()):
                if not cond_dir.is_dir():
                    continue
                for seed_dir in sorted(cond_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    if list(seed_dir.glob("completions*.jsonl")):
                        cells.append(Cell(model, ds, cond_dir.name, seed_dir.name))
    return cells


def completions_path(cell: Cell) -> Optional[Path]:
    hits = sorted(cell.gen_dir().glob("completions*.jsonl"))
    return hits[0] if hits else None


def load_completions(cell: Cell) -> list[dict]:
    p = completions_path(cell)
    return read_jsonl(p) if p else []


# --- existing-v5 judge readers (read-only) ---------------------------------

def load_final_answer_labels(cell: Cell) -> dict[str, Optional[int]]:
    """id -> harmful_response (0/1) or None, from the v5 answer-safety judge.

    NB: the v5 judged_*.jsonl was produced on the FULL completion, so this is
    the *uncorrected* answer label. v6 re-judging on parsed answer_text
    overwrites these; until then they are used only for the aggregation-only
    correction (marginal gap -> paired), clearly labelled as such.
    """
    hits = sorted(cell.judge_dir().glob("judged_*.jsonl"))
    out: dict[str, Optional[int]] = {}
    if not hits:
        return out
    for r in iter_jsonl(hits[0]):
        flat = r.get("judge_flat") or {}
        labs = flat.get("labels") or {}
        v = labs.get("harmful_response")
        out[str(r.get("id"))] = (int(bool(v)) if isinstance(v, bool) else None)
    return out


def load_cot_only_labels(cell: Cell) -> dict[str, Optional[int]]:
    """parent_id -> cot_predicts_unsafe (0/1) or None, from the v5 trace-only judge."""
    p = cell.judge_dir() / "judge_cot_only.jsonl"
    out: dict[str, Optional[int]] = {}
    if not p.exists():
        return out
    for r in iter_jsonl(p):
        flat = r.get("judge_flat") or {}
        v = flat.get("cot_predicts_unsafe")
        pid = str(r.get("parent_id") or r.get("id"))
        out[pid] = (int(bool(v)) if isinstance(v, bool) else None)
    return out


def load_coherence(cell: Cell) -> dict[str, dict]:
    """id -> coherence row (is_clean, is_empty, ...) from v5 (computed on full completion)."""
    p = cell.judge_dir() / "coherence.jsonl"
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    for r in iter_jsonl(p):
        out[str(r.get("id"))] = r
    return out
