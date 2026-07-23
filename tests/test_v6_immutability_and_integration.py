"""Source immutability, judge-input leak guards, and a 2-cell end-to-end smoke.

These tests exercise the v6 *scripts* (not just the library) against the real
v5 tree, but only ever read v5 and write under runs/direction_a_v6. They are
skipped automatically if the v5 tree is absent (e.g. a clean checkout without
the large run artifacts).
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
V5 = REPO / "runs" / "direction_a_v5"
V6 = REPO / "runs" / "direction_a_v6"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SRC))

pytestmark = pytest.mark.skipif(
    not (V5 / "olmo3_7b_think" / "gen" / "jbb").is_dir(),
    reason="v5 run tree not present",
)


def _hash_tree(root: Path, sample_limit=40):
    """Hash a deterministic sample of v5 files (name+size+content digest)."""
    files = sorted(root.rglob("completions*.jsonl"))[:sample_limit]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest(), len(files)


def _run(mod, *args):
    env = {"PYTHONPATH": f"{SRC}:{SCRIPTS}"}
    import os
    env = {**os.environ, **env}
    r = subprocess.run([sys.executable, str(SCRIPTS / mod), *args],
                       cwd=str(REPO), capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{mod} failed:\n{r.stdout}\n{r.stderr}"
    return r


def test_v5_source_unchanged_after_cpu_pipeline():
    before, n = _hash_tree(V5)
    assert n > 0
    _run("parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("aggregate_v6_metrics.py", "--models", "olmo3_7b_think",
         "--datasets", "jbb", "--no-bootstrap")
    after, _ = _hash_tree(V5)
    assert before == after, "v5 source generation files were modified by v6 pipeline"


def test_answer_inputs_have_no_trace_and_trace_inputs_have_no_answer():
    """Script-level guard: the judge-input builder never leaks across the split."""
    import run_v6_judge_shard as J
    import v6_common as C
    # ensure parsed files exist for the cell
    _run("parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    cell = C.Cell("olmo3_7b_think", "jbb", "baseline", "seed0")
    answer_rows = J.build_inputs(cell, "answer_text", "answer", prose_prefix=False)
    trace_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=False)
    assert answer_rows and trace_rows
    # answer inputs must not carry the closing think tag (no hidden trace)
    assert all("</think>" not in r["completion"].lower() for r in answer_rows)
    # trace inputs are the reasoning; they must be non-empty and explicit only
    assert all(r["completion"].strip() for r in trace_rows)
    assert all(r["trace_kind"] == "explicit" for r in trace_rows)


def test_prose_prefix_pass_is_labelled_sensitivity():
    import run_v6_judge_shard as J
    import v6_common as C
    _run("parse_v6_completions.py", "--models", "llama31_8b_control", "--datasets", "jbb")
    cell = C.Cell("llama31_8b_control", "jbb", "baseline", "seed0")
    # without --prose-prefix, an explicit-trace stage yields nothing for prose model
    none_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=False)
    assert none_rows == []
    # with the prose-prefix pass, rows are produced and flagged as prefix
    pref_rows = J.build_inputs(cell, "trace_text", "monitor", prose_prefix=True)
    assert pref_rows
    assert all(r["is_prefix"] for r in pref_rows)


def test_two_cell_end_to_end_smoke():
    """audit -> parse -> aggregate over one model/dataset produces the reports."""
    import json
    _run("audit_v6_generations.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("parse_v6_completions.py", "--models", "olmo3_7b_think", "--datasets", "jbb")
    _run("aggregate_v6_metrics.py", "--models", "olmo3_7b_think",
         "--datasets", "jbb", "--no-bootstrap")
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
