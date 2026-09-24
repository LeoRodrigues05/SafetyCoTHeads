#!/usr/bin/env python3
"""Emit steering configs for the intermediate doses a0.75 and a1.25.

The v6 correction showed that steering's SFS is non-monotonic in dose: Potency
keeps rising while Coherence Retention collapses, so SFS peaks at a=1.0 and falls
at a=1.5. That claim currently rests on a three-point ladder. This adds the two
midpoints so the Potency/Quality trade-off is a five-point curve per arm.

Each new config is derived from that arm's EXISTING a1.0 config by changing only
``condition``, ``steering.alpha`` and ``output.dir`` -- model, direction file,
layer, dataset, decoding, seed and batch size are inherited verbatim, so the new
points are directly comparable to the three already in the grid.

The dose ladder is linear in alpha (a0.5 -> -4, a1.0 -> -8, a1.5 -> -12), so the
midpoints are -6 and -10.

Usage:
    .venv/bin/python scripts/configs/make_dose_refinement_configs.py [--write]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr"

#: all five primary arms (the July run covered only the first three, and the
#: Llama cells failed on HF auth), so the refined ladder can span the grid
MODELS = ["qwen3_8b", "llama31_8b_control", "olmo3_7b_think",
          "olmo3_7b_base", "olmo3_7b_base_own"]
DATASETS = ["jbb", "bt"]
#: new condition -> alpha, interpolated on the existing linear ladder
NEW_DOSES = {"steering_a0.75": -6.0, "steering_a1.25": -10.0}
BASE_COND = "steering_a1.0"
BASE_ALPHA = -8.0


def derive(text: str, model: str, ds: str, new_cond: str, alpha: float) -> str:
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("run:"):
            out.append(f"  run: {new_cond}_{model}_{ds}")
        elif s.startswith("condition:") and not line.startswith(" "):
            out.append(f"condition: {new_cond}")
        elif s.startswith("dir:") and BASE_COND in line:
            out.append(line.replace(f"/{BASE_COND}/", f"/{new_cond}/"))
        elif s.startswith("alpha:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}alpha: {alpha}")
        else:
            out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write files (default: dry run)")
    ap.add_argument("--models", nargs="*", default=None, help=f"default: {MODELS}")
    args = ap.parse_args()

    made, missing = [], []
    for model in (args.models or MODELS):
        for ds in DATASETS:
            src = CFG / model / "gen" / ds / f"{BASE_COND}.yaml"
            if not src.exists():
                missing.append(str(src.relative_to(ROOT)))
                continue
            text = src.read_text()
            if f"alpha: {BASE_ALPHA}" not in text:
                missing.append(f"{src.relative_to(ROOT)} (unexpected alpha; refusing to derive)")
                continue
            for cond, alpha in NEW_DOSES.items():
                dst = CFG / model / "gen" / ds / f"{cond}.yaml"
                body = derive(text, model, ds, cond, alpha)
                assert f"alpha: {alpha}" in body, f"alpha rewrite failed for {dst}"
                assert f"/{cond}/" in body, f"output dir rewrite failed for {dst}"
                assert f"condition: {cond}" in body, f"condition rewrite failed for {dst}"
                if args.write:
                    dst.write_text(body)
                made.append(str(dst.relative_to(ROOT)))

    for m in missing:
        print(f"  [skip] {m}")
    for m in made:
        print(f"  [{'write' if args.write else 'plan'}] {m}")
    print(f"\n{len(made)} configs {'written' if args.write else 'planned'}, {len(missing)} skipped")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
