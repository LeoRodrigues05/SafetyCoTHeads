"""Deterministic paired prompt-level bootstrap confidence intervals for v6.

For each baseline-vs-intervention comparison we:

1. Intersect the prompt IDs present in both baseline and intervention cells.
2. Resample prompt IDs with replacement (the SAME sampled IDs are used for both
   the baseline and the intervention cell, preserving pairing).
3. Recompute every statistic (HAC, P, Q, U, O, agreement, trace FNR, S, SFS)
   on the resampled ids.
4. Report percentile 95% CIs over >= n_boot replicates (>=10000 for final runs).

The RNG is seeded (default 12345) so intervals are reproducible; the seed is
recorded in the run manifest. Statistics that are undefined on a given
replicate (e.g. trace FNR when the resample contains no harmful answers)
contribute ``nan`` and are ignored via nan-aware percentiles, with the fraction
of defined replicates reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .paired_metrics import PairedItem, build_paired_table, monitorability_retention
from .aggregate import AnswerSignals, potency, quality


@dataclass
class CI:
    point: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    n_defined: int
    n_boot: int

    def to_dict(self) -> dict:
        return {"point": self.point, "ci95_lo": self.lo, "ci95_hi": self.hi,
                "n_defined": self.n_defined, "n_boot": self.n_boot}


def _percentile_ci(samples: np.ndarray, point: Optional[float], n_boot: int) -> CI:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return CI(point=point, lo=None, hi=None, n_defined=0, n_boot=n_boot)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return CI(point=point, lo=float(lo), hi=float(hi), n_defined=int(finite.size), n_boot=n_boot)


@dataclass
class PairedAnswerCell:
    """Per-prompt records for one cell, keyed by prompt_id.

    ``y`` (final-answer harmful), ``t`` (trace unsafe), ``clean`` (coherence
    gate). ``harmful_clean`` = y among clean. Used to recompute HAC/clean/U on a
    resampled id set.
    """

    by_id: dict  # prompt_id -> {"y": int|None, "t": int|None, "clean": bool,
                 #               "answer_nonempty": bool, "trace_available": bool}


def _cell_stats_on_ids(cell: PairedAnswerCell, ids: Sequence[str]):
    """Recompute answer-level HAC/clean_rate and a PairedTable on a given id list."""
    n_clean = n_harm_clean = 0
    items = []
    for pid in ids:
        r = cell.by_id.get(pid)
        if r is None:
            continue
        clean = bool(r.get("clean"))
        y = r.get("y")
        if clean:
            n_clean += 1
            if y == 1:
                n_harm_clean += 1
        items.append(PairedItem(
            prompt_id=str(pid), y=y, t=r.get("t"),
            clean=clean, trace_available=bool(r.get("trace_available", True)),
        ))
    hac = (n_harm_clean / n_clean) if n_clean > 0 else None
    clean_rate = (n_clean / len(ids)) if ids else None
    table = build_paired_table(items, include_nonclean=False, require_trace=True)
    return hac, clean_rate, table


def paired_bootstrap(
    cell: PairedAnswerCell,
    baseline: PairedAnswerCell,
    n_boot: int = 10000,
    seed: int = 12345,
    orientation: str = "suppress",
) -> dict:
    """Return CIs for HAC, P, Q, U, O, agreement, trace FNR, S, SFS.

    Only prompt IDs present in BOTH cells are resampled (paired). The same
    resampled id vector drives baseline and intervention recomputation.
    """
    shared = sorted(set(cell.by_id) & set(baseline.by_id))
    rng = np.random.default_rng(seed)
    n = len(shared)

    keys = ["hac", "P", "Q", "U", "O", "agreement", "trace_fnr", "S", "S_fnr", "sfs", "g"]
    draws = {k: np.full(n_boot, np.nan) for k in keys}

    def _point():
        return _compose(cell, baseline, shared, orientation)

    point = _point() if n > 0 else {k: None for k in keys}

    if n > 0:
        idx_arr = np.asarray(shared, dtype=object)
        for b in range(n_boot):
            take = rng.integers(0, n, size=n)
            ids = idx_arr[take].tolist()
            vals = _compose(cell, baseline, ids, orientation)
            for k in keys:
                v = vals.get(k)
                draws[k][b] = v if v is not None else np.nan

    return {
        "n_shared_ids": n,
        "n_boot": n_boot,
        "seed": seed,
        "cis": {k: _percentile_ci(draws[k], point.get(k), n_boot).to_dict() for k in keys},
    }


def _compose(cell, baseline, ids, orientation):
    hac_c, clean_c, tab_c = _cell_stats_on_ids(cell, ids)
    hac_b, clean_b, tab_b = _cell_stats_on_ids(baseline, ids)
    a_c = AnswerSignals(model="", dataset="", condition="", hac=hac_c, clean_rate=clean_c)
    a_b = AnswerSignals(model="", dataset="", condition="", hac=hac_b, clean_rate=clean_b)
    P = potency(a_c, a_b, orientation=orientation)
    Q = quality(a_c, a_b)
    mr = monitorability_retention(tab_c, tab_b)
    S = mr.S
    sfs = None
    if P is not None and Q is not None and S is not None:
        sfs = 0.0 if min(P, Q, S) <= 0.0 else (P * Q * S) ** (1.0 / 3.0)
    return {
        "hac": hac_c, "P": P, "Q": Q,
        "U": tab_c.U, "O": tab_c.O, "agreement": tab_c.A,
        "trace_fnr": tab_c.trace_fnr, "S": S, "S_fnr": mr.S_fnr,
        "sfs": sfs, "g": tab_c.g,
    }
