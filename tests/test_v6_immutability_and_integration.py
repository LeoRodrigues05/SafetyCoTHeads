"""Source immutability, judge-input leak guards, and a 2-cell end-to-end smoke.

These tests exercise the v6 *scripts* (not just the library) against the real
v5 tree. They only ever READ the real run trees: every v6 write goes to a
temporary root (``SCH_V6_ROOT``), seeded with copies of the few real judge
files the aggregation needs. (Before 2026-09-24 they wrote into the real
runs/direction_a_v6 tree -- re-parsing live cells and, had aggregation
succeeded, overwriting the headline reports.) They are skipped automatically if
the v5 tree is absent (e.g. a clean checkout without the large run artifacts).
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
V5 = REPO / "runs" / "direction_a_v5"
REAL_V6 = REPO / "runs" / "direction_a_v6"

sys.path.insert(0, str(SCRIPTS / "common"))
import _bootstrap  # noqa: E402,F401  (src/ + every scripts/<group>/ on sys.path)

pytestmark = pytest.mark.skipif(
    not (V5 / "olmo3_7b_think" / "gen" / "jbb").is_dir(),
    reason="v5 run tree not present",
)

#: judge files the aggregation reads, copied (never moved) into the temp root
_SEED_FILES = ("coherence_answer.jsonl", "judge_answer_safety.jsonl", "judge_cot_only.jsonl",
               "judge_cot_only__prefix.jsonl")
_SEED_CONDS = ("baseline", "ships_top3", "steering_a1.0", "steering_ablate")


@pytest.fixture(scope="module")
def V6(tmp_path_factory):
    root = tmp_path_factory.mktemp("v6_root")
    for model in ("olmo3_7b_think", "llama31_8b_control"):
        for cond in _SEED_CONDS:
            src = REAL_V6 / "judge" / model / "jbb" / cond / "seed0"
            dst = root / "judge" / model / "jbb" / cond / "seed0"
            for name in _SEED_FILES:
                if (src / name).exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / name, dst / name)
    old = os.environ.get("SCH_V6_ROOT")
    os.environ["SCH_V6_ROOT"] = str(root)
    import v6_common as C
    saved = C.V6_ROOT
    C.V6_ROOT = root                                  # in-process callers
    yield root
    C.V6_ROOT = saved
    if old is None:
        os.environ.pop("SCH_V6_ROOT", None)
    else:
        os.environ["SCH_V6_ROOT"] = old


def _hash_tree(root: Path, sample_limit=40):
    """Hash a deterministic sample of v5 files (name+size+content digest)."""
    files = sorted(root.rglob("completions*.jsonl"))[:sample_limit]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest(), len(files)


def _hash_dir(root: Path):
    """Digest of every file (name + content) directly under ``root``."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.glob("*") if p.is_file()):
        h.update(f.name.encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def _run(mod, *args):
    assert os.environ.get("SCH_V6_ROOT"), "integration tests must write to a temp v6 root"
    env = {**os.environ, "PYTHONPATH": f"{SRC}"}
    r = subprocess.run([sys.executable, str(SCRIPTS / mod), *args],
                       cwd=str(REPO), capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{mod} failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_v5_source_unchanged_after_cpu_pipeline(V6):
    real_reports = _hash_dir(REAL_V6 / "reports")
    before, n = _hash_tree(V5)
    assert n > 0
    _run("preprocessing/parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("analysis/aggregate_v6_metrics.py", "--models", "olmo3_7b_think",
         "--datasets", "jbb", "--no-bootstrap", "--coherence-gate", "repetition-only")
    after, _ = _hash_tree(V5)
    assert before == after, "v5 source generation files were modified by v6 pipeline"
    assert _hash_dir(REAL_V6 / "reports") == real_reports, "real v6 reports were modified"


def test_answer_inputs_have_no_trace_and_trace_inputs_have_no_answer(V6):
    """Script-level guard: the judge-input builder never leaks across the split."""
    import run_v6_judge_shard as J
    import v6_common as C
    # ensure parsed files exist for the cell
    _run("preprocessing/parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    cell = C.Cell("olmo3_7b_think", "jbb", "baseline", "seed0")
    answer_rows = J.build_inputs(cell, "answer_text", "answer", prose_prefix=False)
    trace_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=False)
    assert answer_rows and trace_rows
    # answer inputs must not carry the closing think tag (no hidden trace)
    assert all("</think>" not in r["completion"].lower() for r in answer_rows)
    # trace inputs are the reasoning; they must be non-empty and explicit only
    # (malformed_explicit = a truncated trace: still reasoning, never an answer)
    assert all(r["completion"].strip() for r in trace_rows)
    assert all(r["trace_kind"] in ("explicit", "malformed_explicit") for r in trace_rows)


def test_prose_prefix_pass_is_labelled_sensitivity(V6):
    import run_v6_judge_shard as J
    import v6_common as C
    _run("preprocessing/parse_v6_completions.py", "--models", "llama31_8b_control", "--datasets", "jbb")
    cell = C.Cell("llama31_8b_control", "jbb", "baseline", "seed0")
    # without --prose-prefix, an explicit-trace stage yields nothing for prose model
    none_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=False)
    assert none_rows == []
    # with the prose-prefix pass, rows are produced and flagged as prefix
    pref_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=True)
    assert pref_rows
    assert all(r["is_prefix"] for r in pref_rows)


def test_two_cell_end_to_end_smoke(V6):
    """audit -> parse -> aggregate over one model/dataset produces the reports."""
    import json
    _run("preprocessing/audit_v6_generations.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("preprocessing/parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("analysis/aggregate_v6_metrics.py", "--models", "olmo3_7b_think",
         "--datasets", "jbb", "--no-bootstrap", "--coherence-gate", "repetition-only")
    metrics = V6 / "reports" / "cell_metrics.json"
    assert metrics.exists()
    rows = json.loads(metrics.read_text())["rows"]
    assert rows, "no metric rows produced"
    # at least one cell has a full paired 2x2 and denominators exposed
    has_paired = [r for r in rows if r.get("U_covert") is not None]
    assert has_paired
    r = has_paired[0]
    for k in ("n_generated", "n_clean", "n_pairs", "n_harmful_paired"):
        assert k in r
    # prompt-id pairing invariant: paired denominator never exceeds n_generated
    assert r["n_pairs"] <= r["n_generated"]
