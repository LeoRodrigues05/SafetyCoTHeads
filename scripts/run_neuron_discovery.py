"""Discover safety-relevant MLP neurons (Wang et al. 2024).

Usage:
    python -m scripts.run_neuron_discovery \
        --config configs/experiments/direction_a_ships/16-neuron-discovery-llama31.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

from safety_cot_heads.attribution import (
    NeuronAttributionConfig, neuron_attribution,
)
from safety_cot_heads.data import (
    load_alpaca, load_beavertails, load_jailbreakbench, load_maliciousinstruct,
)
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import ensure_dir, get_logger, json_dump, set_seed

log = get_logger(__name__)


def _load_split(name: str, n: int | None, **kw):
    if name == "maliciousinstruct": return load_maliciousinstruct(n=n)
    if name == "jailbreakbench":    return load_jailbreakbench(n=n)
    if name == "alpaca":            return load_alpaca(n=n)
    if name == "beavertails":
        return load_beavertails(
            categories=kw.get("categories"),
            n_per_category=kw.get("n_per_category"),
        )
    raise ValueError(f"unknown dataset {name!r}")


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

    harm_rows = _load_split(cfg.harmful.name, cfg.harmful.get("n"),
                            categories=cfg.harmful.get("categories"),
                            n_per_category=cfg.harmful.get("n_per_category"))
    benign_rows = _load_split(cfg.benign.name, cfg.benign.get("n"))
    log.info("harmful=%d benign=%d", len(harm_rows), len(benign_rows))

    if args.dry_run:
        json_dump(out_dir / "dryrun.json",
                  {"plan": cfg_to_dict(cfg),
                   "n_harmful": len(harm_rows), "n_benign": len(benign_rows)})
        return 0

    lm = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        attn_implementation=cfg.model.get("attn_implementation"),
        device_map=cfg.model.get("device_map"),
    )

    attr_cfg = NeuronAttributionConfig(
        batch_size=int(cfg.get("batch_size", 4)),
        max_length=int(cfg.get("max_length", 512)),
        capture_last_n=int(cfg.get("capture_last_n", 1)),
        top_k_default=int(cfg.get("top_k_default", 32)),
        system_prompt=cfg.get("system_prompt"),
    )
    result = neuron_attribution(
        lm,
        harmful_prompts=[r["prompt"] for r in harm_rows],
        benign_prompts=[r["prompt"] for r in benign_rows],
        cfg=attr_cfg,
    )
    out_path = out_dir / "neuron_ranking.json"
    json_dump(out_path, result)
    log.info("wrote %s (top1=%s)", out_path, result["ranked_neurons"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
