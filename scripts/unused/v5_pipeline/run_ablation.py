"""Alias for run_generation with an ablation condition pre-configured.

This is a thin wrapper kept for naming parity with the spec; the real work
lives in :mod:`scripts.generation.run_generation`.  Use::

    python -m scripts.unused.v5_pipeline.run_ablation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)
import sys
from pathlib import Path


if __name__ == "__main__":
    from run_generation import main      # type: ignore
    sys.exit(main())
