"""Expand a model × condition × seed × dataset grid into one config per cell.

Usage:
    python -m scripts.make_experiment_matrix --matrix configs/experiments/exp03_safety_vs_random_ablation/matrix.yaml --out runs/matrix/
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mat = cfg_to_dict(load_cfg(args.matrix))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = mat["axes"]
    base = mat.get("base", {})
    keys = list(axes.keys())
    n = 0
    for combo in itertools.product(*(axes[k] for k in keys)):
        cell = dict(base)
        for k, v in zip(keys, combo):
            cell[k] = vd
        slug = "__".join(str(v).replace("/", "_") for v in combo)
        path = out_dir / f"{slug}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(cell, f, sort_keys=False)
        n += 1
    print(f"wrote {n} configs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
