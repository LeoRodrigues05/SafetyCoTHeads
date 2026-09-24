#!/usr/bin/env python3
"""Generate many cells of ONE model with a single model load.

Per cell this does exactly what ``run_generation.py --config <cfg>`` does (same
dataset loader, same ``build_interventions``, same ``generate`` call, same
output path and row schema); it only avoids reloading the weights for every
cell, which dominates wall time for the short cells of the control grid.
Cells whose ``completions_<condition>.jsonl`` already exists are skipped, so a
killed run resumes. All configs must name the same model / dtype / attention
implementation (checked).

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/generation/run_generation_batch.py \
      --configs configs/experiments/direction_a_v5_iso_asr/qwen3_8b/gen/*/rand_*.yaml
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import sys
import time
from pathlib import Path

from _cli import cfg_to_dict, load_cfg  # noqa: E402
from run_generation import _load_dataset, build_interventions  # noqa: E402

from safety_cot_heads.generation import DecodingConfig, generate  # noqa: E402
from safety_cot_heads.models import load_model  # noqa: E402
from safety_cot_heads.utils import ensure_dir, jsonl_write, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()

    lm, sig = None, None
    n_done = n_skip = 0
    for cfg_path in args.configs:
        cfg = load_cfg(cfg_path, [])
        out_dir = Path(cfg.output.dir)
        out_file = out_dir / f"completions_{cfg.condition}.jsonl"
        if out_file.exists():
            n_skip += 1
            continue
        this = (cfg.model.name, cfg.model.get("dtype", "auto"),
                cfg.model.get("attn_implementation"))
        if lm is None:
            lm = load_model(cfg.model.name, dtype=this[1], attn_implementation=this[2],
                            load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
                            device_map=cfg.model.get("device_map"),
                            trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
            sig = this
        elif this != sig:
            raise SystemExit(f"{cfg_path}: model settings {this} differ from loaded {sig}; "
                             "run one batch per model")
        t0 = time.time()
        set_seed(int(cfg.get("seed", 0)))
        ensure_dir(out_dir)
        rows = _load_dataset(cfg.dataset.name, cfg.dataset.get("n"),
                             categories=cfg.dataset.get("categories"),
                             n_per_category=cfg.dataset.get("n_per_category"))
        iv = build_interventions(cfg, lm, config_path=str(cfg_path))
        out_rows = generate(
            lm, rows, DecodingConfig(**(cfg.get("decoding") or {})),
            mask_cfg=iv["mask_cfg"], neuron_cfg=iv["neuron_cfg"],
            steering_cfg=iv["steering_cfg"],
            system_prompt=cfg.get("system_prompt"),
            batch_size=int(cfg.get("batch_size", 4)),
            condition_label=cfg.condition,
            chat_template_kwargs=(cfg_to_dict(cfg.get("chat_overrides"))
                                  if cfg.get("chat_overrides") is not None else None),
            extra_meta=iv["extra_meta"],
        )
        jsonl_write(out_file, out_rows)
        n_done += 1
        print(f"[gen-batch] {cfg_path}: {len(out_rows)} rows in {time.time() - t0:.0f}s",
              flush=True)
    print(f"[gen-batch] generated {n_done} cells, skipped {n_skip} existing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
