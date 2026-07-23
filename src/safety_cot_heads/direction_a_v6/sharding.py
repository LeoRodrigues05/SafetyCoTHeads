"""Deterministic task sharding for the two-B200 runner.

A "task" is any hashable identifier string (a judge-input row id, or a
``model/dataset/condition/seed`` cell key). Sharding must be:

* **deterministic** — the same task list yields the same assignment on every
  machine/run (hash of the stable task id, not Python's salted ``hash()``);
* **non-overlapping** — every task lands on exactly one shard;
* **complete** — the union of all shards equals the input set.

We assign by ``blake2b(task_id) % n_shards`` so a task keeps its shard even if
the overall job set changes (adding/removing cells does not reshuffle survivors),
which makes resume-by-task safe.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


def _stable_bucket(task_id: str, n_shards: int) -> int:
    h = hashlib.blake2b(task_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_shards


def shard_of(task_id: str, n_shards: int) -> int:
    """Return the shard index in ``[0, n_shards)`` for ``task_id``."""
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    return _stable_bucket(str(task_id), n_shards)


def assign_shards(task_ids: Iterable[str], n_shards: int) -> list[list[str]]:
    """Partition ``task_ids`` into ``n_shards`` deterministic buckets.

    Order within a shard follows first-seen input order (stable). Duplicate ids
    are de-duplicated (first occurrence wins) so coverage checks are exact.
    """
    if n_shards <= 0:
        raise ValueError("n_shards must be positive")
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    seen: set[str] = set()
    for tid in task_ids:
        s = str(tid)
        if s in seen:
            continue
        seen.add(s)
        shards[_stable_bucket(s, n_shards)].append(s)
    return shards


def verify_partition(task_ids: Sequence[str], shards: Sequence[Sequence[str]]) -> dict:
    """Check that ``shards`` is a valid partition of the de-duplicated ``task_ids``.

    Returns a diagnostics dict: ``ok`` plus the offending sets (overlap, missing,
    extra). Used by the completeness checker and tests.
    """
    unique = list(dict.fromkeys(str(t) for t in task_ids))
    unique_set = set(unique)
    seen: set[str] = set()
    overlap: set[str] = set()
    for sh in shards:
        for t in sh:
            t = str(t)
            if t in seen:
                overlap.add(t)
            seen.add(t)
    missing = unique_set - seen
    extra = seen - unique_set
    return {
        "ok": not overlap and not missing and not extra,
        "n_tasks": len(unique_set),
        "n_assigned": len(seen),
        "overlap": sorted(overlap),
        "missing": sorted(missing),
        "extra": sorted(extra),
    }
