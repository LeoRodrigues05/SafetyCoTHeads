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

REPO = Path(__file__).resolve().parents[2]
V5_ROOT = REPO / "runs" / "direction_a_v5"
# SCH_V6_ROOT redirects every v6 write (tests point it at a temp dir so they
# never touch the real run tree).
V6_ROOT = Path(os.environ["SCH_V6_ROOT"]) if os.environ.get("SCH_V6_ROOT") \
    else REPO / "runs" / "direction_a_v6"
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


def continued_gen_dir(cell: Cell) -> Path:
    """Where scripts/generation/continue_truncated_generations.py writes a cell whose
    truncated reasoning traces were continued past the original token cap. The
    v5 file is never modified; this copy supersedes it when present."""
    return V6_ROOT / "gen_continued" / cell.model / cell.dataset / cell.condition / cell.seed


def completions_path(cell: Cell) -> Optional[Path]:
    hits = sorted(continued_gen_dir(cell).glob("completions*.jsonl"))
    if hits:
        return hits[0]
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


# --- input-hash-aware resume (roadmap P0.7) ---------------------------------
#
# Judge outputs used to be resumed by id alone, so after a parser fix the rows
# whose input text changed kept their stale labels. Every judge input row now
# carries ``input_sha256`` (hash of the exact prompt + text the judge sees) and
# every judged row copies it, so a later run re-judges exactly the rows whose
# input changed. Rows written before hashing existed carry no hash and are
# trusted; known-changed legacy rows are removed by scripts/preprocessing/invalidate_v6_rows.py.

def input_sha(prompt: Optional[str], completion: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update((prompt or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((completion or "").encode("utf-8"))
    return h.hexdigest()[:20]


def resume_todo(rows: list[dict], out_path: Path,
                version_field: Optional[str] = None,
                version_value: Optional[str] = None) -> tuple[list[dict], set[str]]:
    """Split judge-input ``rows`` into (todo, stale_ids) against ``out_path``.

    A row is done iff an output row with its id exists, that output's
    ``input_sha256`` is absent (legacy) or equal to the input's, and -- when
    ``version_field`` is given -- ``out[version_field] == version_value``.
    ``stale_ids`` are ids that have an output row which must be replaced.
    """
    existing = {str(r.get("id")): r for r in read_jsonl(out_path)} if out_path.exists() else {}
    todo, stale = [], set()
    for r in rows:
        rid = str(r["id"])
        prev = existing.get(rid)
        if prev is None:
            todo.append(r)
            continue
        ok = True
        ph, ch = prev.get("input_sha256"), r.get("input_sha256")
        if ph is not None and ch is not None and ph != ch:
            ok = False
        if version_field is not None and prev.get(version_field) != version_value:
            ok = False
        if not ok:
            todo.append(r)
            stale.add(rid)
    return todo, stale


def prune_rows(out_path: Path, ids: set[str], tag: str) -> int:
    """Remove output rows whose id is in ``ids``; removed rows are kept in a
    sibling ``<name>.pruned_<tag>.jsonl`` (nothing is ever discarded)."""
    if not ids or not out_path.exists():
        return 0
    keep, gone = [], []
    for r in read_jsonl(out_path):
        rid = str(r.get("id"))
        pid = str(r.get("parent_id")) if r.get("parent_id") is not None else None
        (gone if (rid in ids or (pid is not None and pid in ids and "::" in rid)) else keep).append(r)
    if not gone:
        return 0
    bak = out_path.with_name(out_path.stem + f".pruned_{tag}.jsonl")
    with open(bak, "a") as f:
        for r in gone:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(out_path, keep)
    return len(gone)
