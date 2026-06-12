"""Alias for run_generation with an ablation condition pre-configured.

This is a thin wrapper kept for naming parity with the spec; the real work
lives in :mod:`scripts.run_generation`.  Use::

    python -m scripts.run_ablation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from run_generation import main      # type: ignore
    sys.exit(main())
