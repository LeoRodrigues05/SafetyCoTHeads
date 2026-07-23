"""Tests for paired bootstrap reproducibility and deterministic sharding."""

import pytest

from safety_cot_heads.direction_a_v6.bootstrap import PairedAnswerCell, paired_bootstrap
from safety_cot_heads.direction_a_v6.sharding import (
    shard_of, assign_shards, verify_partition,
)


def _cell(seed_pattern):
    """Build a PairedAnswerCell from a dict pid -> (y, t, clean)."""
    by_id = {pid: {"y": y, "t": t, "clean": clean, "trace_available": True}
             for pid, (y, t, clean) in seed_pattern.items()}
    return PairedAnswerCell(by_id=by_id)


def _make_pair():
    base = _cell({f"p{i}": (0, 0, True) for i in range(20)})
    cell = _cell({f"p{i}": ((1 if i < 8 else 0), (1 if i < 4 else 0), True)
                  for i in range(20)})
    return cell, base


def test_bootstrap_is_reproducible():
    cell, base = _make_pair()
    r1 = paired_bootstrap(cell, base, n_boot=500, seed=777)
    r2 = paired_bootstrap(cell, base, n_boot=500, seed=777)
    assert r1["cis"]["U"] == r2["cis"]["U"]
    assert r1["cis"]["P"] == r2["cis"]["P"]
    assert r1["cis"]["sfs"] == r2["cis"]["sfs"]


def test_bootstrap_seed_changes_interval():
    cell, base = _make_pair()
    r1 = paired_bootstrap(cell, base, n_boot=500, seed=1)
    r2 = paired_bootstrap(cell, base, n_boot=500, seed=2)
    # different seed -> (almost surely) different bounds
    assert r1["cis"]["U"]["ci95_lo"] != r2["cis"]["U"]["ci95_lo"] or \
           r1["cis"]["U"]["ci95_hi"] != r2["cis"]["U"]["ci95_hi"]


def test_bootstrap_only_uses_shared_ids():
    base = _cell({f"p{i}": (0, 0, True) for i in range(10)})
    cell = _cell({f"p{i}": (1, 0, True) for i in range(5)})  # only p0..p4 shared
    r = paired_bootstrap(cell, base, n_boot=100, seed=5)
    assert r["n_shared_ids"] == 5


def test_bootstrap_point_estimate_present():
    cell, base = _make_pair()
    r = paired_bootstrap(cell, base, n_boot=200, seed=9)
    assert r["cis"]["U"]["point"] == pytest.approx(4 / 20)
    lo, hi = r["cis"]["U"]["ci95_lo"], r["cis"]["U"]["ci95_hi"]
    assert lo <= r["cis"]["U"]["point"] <= hi


def test_sharding_deterministic():
    tasks = [f"task-{i}" for i in range(1000)]
    a = assign_shards(tasks, 2)
    b = assign_shards(tasks, 2)
    assert [list(s) for s in a] == [list(s) for s in b]
    # stable across process: shard_of matches assignment
    for i, sh in enumerate(a):
        for t in sh:
            assert shard_of(t, 2) == i


def test_sharding_no_overlap_full_coverage():
    tasks = [f"task-{i}" for i in range(1000)]
    shards = assign_shards(tasks, 2)
    diag = verify_partition(tasks, shards)
    assert diag["ok"]
    assert diag["overlap"] == []
    assert diag["missing"] == []
    total = sum(len(s) for s in shards)
    assert total == 1000


def test_sharding_stable_under_task_set_change():
    tasks = [f"task-{i}" for i in range(500)]
    shards = assign_shards(tasks, 2)
    where = {t: (0 if t in shards[0] else 1) for t in tasks}
    # remove some tasks, add others; survivors keep their shard
    tasks2 = tasks[100:] + [f"new-{i}" for i in range(50)]
    shards2 = assign_shards(tasks2, 2)
    where2 = {t: (0 if t in shards2[0] else 1) for t in tasks2}
    for t in tasks[100:]:
        assert where[t] == where2[t]


def test_sharding_dedups_duplicates():
    tasks = ["a", "a", "b", "b", "b", "c"]
    shards = assign_shards(tasks, 2)
    diag = verify_partition(tasks, shards)
    assert diag["ok"]
    assert diag["n_tasks"] == 3
