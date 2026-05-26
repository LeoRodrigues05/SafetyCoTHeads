"""Run head attribution (SHIPS or Sahara) and save ranked-heads JSONL.

Usage:
    python -m scripts.run_attribution --config configs/experiments/exp01_reproduce_ships_sahara/01-ships-discovery.yaml
    python -m scripts.run_attribution --config configs/experiments/exp01_reproduce_ships_sahara/02-sahara-discovery.yaml [--dry-run]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# allow `python scripts/run_attribution.py` without -m
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

from safety_cot_heads.attribution import (
    SHIPS, SHIPSConfig, SaharaConfig,
    aggregate_dataset_ranking, safety_head_attribution,
)
from safety_cot_heads.data import (
    load_beavertails, load_jailbreakbench, load_maliciousinstruct,
)
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _load_dataset(name: str, split_n: int | None, **kw):
    if name == "maliciousinstruct": return load_maliciousinstruct(n=split_n)
    if name == "jailbreakbench":    return load_jailbreakbench(n=split_n)
    if name == "beavertails":
        return load_beavertails(
            categories=kw.get("categories"),
            n_per_category=kw.get("n_per_category"),
        )
    raise ValueError(f"unknown attribution dataset {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(cfg.output.dir)
    ensure_dir(out_dir)

    method = cfg.method                                                       # "ships" | "sahara"
    rows = _load_dataset(
        cfg.dataset.name,
        cfg.dataset.get("n"),
        categories=cfg.dataset.get("categories"),
        n_per_category=cfg.dataset.get("n_per_category"),
    )
    log.info("loaded %d prompts from %s", len(rows), cfg.dataset.name)

    if args.dry_run:
        log.info("DRY-RUN: would load model %s and run %s on %d prompts",
                 cfg.model.name, method, len(rows))
        json_dump(out_dir / "dryrun.json",
                  {"plan": cfg_to_dict(cfg), "n_prompts": len(rows)})
        return 0

    lm = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        attn_implementation=cfg.model.get("attn_implementation"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )

    if method == "ships":
        s_cfg = SHIPSConfig(
            mask_qkv=tuple(cfg.method_args.get("mask_qkv", ["q"])),
            mask_type=cfg.method_args.get("mask_type", "scale_mask"),
            scale_factor=float(cfg.method_args.get("scale_factor", 1e-4)),
            top_k=int(cfg.method_args.get("top_k", 10)),
            seed=int(cfg.get("seed", 0)),
            prompt_template=cfg.method_args.get(
                "prompt_template", "## Query:{q}\n## Answer:"),
        )
        ships = SHIPS(lm, s_cfg)
        out_rows = ships.run([(r["id"], r["prompt"]) for r in rows])
        jsonl_write(out_dir / "ships.jsonl", out_rows)
        dataset_rank = aggregate_dataset_ranking(out_rows, top_k=int(cfg.output.get("top_k", 16)))
        json_dump(out_dir / "ships_dataset_ranking.json", {
            "model": lm.name, "method": "ships",
            "dataset": cfg.dataset.name, "n_prompts": len(rows),
            "config": cfg_to_dict(cfg),
            "dataset_ranking": dataset_rank,
        })
    elif method == "sahara":
        s_cfg = SaharaConfig(
            mask_qkv=tuple(cfg.method_args.get("mask_qkv", ["q"])),
            mask_type=cfg.method_args.get("mask_type", "scale_mask"),
            scale_factor=float(cfg.method_args.get("scale_factor", 1e-5)),
            search_step=int(cfg.method_args.get("search_step", 1)),
            prompt_template=cfg.method_args.get(
                "prompt_template", "## Query:{q}\n## Answer:"),
            seed=int(cfg.get("seed", 0)),
        )
        result = safety_head_attribution(
            lm, [r["prompt"] for r in rows], s_cfg,
            top_k=int(cfg.output.get("top_k", 16)),
        )
        json_dump(out_dir / "sahara.json", result)
        jsonl_write(out_dir / "sahara_ranked.jsonl", result["ranked_heads"])
    else:
        raise ValueError(f"unknown method {method!r}")

    log.info("wrote attribution results to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
