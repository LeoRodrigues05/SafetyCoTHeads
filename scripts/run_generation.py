"""Generate completions for a (model × dataset × condition) cell.

Usage:
    python -m scripts.run_generation --config configs/experiments/exp03_safety_vs_random_ablation/03-baseline.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                   # noqa: E402

from safety_cot_heads.attribution import (
    layer_matched, uniform_random,
)
from safety_cot_heads.data import (
    load_alpaca, load_beavertails, load_jailbreakbench,
    load_maliciousinstruct,
)
from safety_cot_heads.generation import DecodingConfig, generate
from safety_cot_heads.interventions import (
    build_mask_cfg, build_neuron_mask_cfg, build_steering_cfg_from_file,
    build_circuit_cfgs,
)
from safety_cot_heads.models import load_model, num_layers_and_heads
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, json_load, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _load_dataset(name: str, n: int | None, **kw):
    if name == "maliciousinstruct": return load_maliciousinstruct(n=n)
    if name == "jailbreakbench":    return load_jailbreakbench(n=n)
    if name == "alpaca":            return load_alpaca(n=n)
    if name == "beavertails":
        return load_beavertails(
            categories=kw.get("categories"),
            n_per_category=kw.get("n_per_category"),
        )
    raise ValueError(f"unknown dataset {name!r}")


def _resolve_heads(cfg, lm) -> list[tuple[int, int]] | None:
    """Read the head set for the active condition, with sensible defaults
    for ``random_*`` conditions that need on-the-fly generation."""
    cond = cfg.condition
    if cond == "baseline":
        return None

    n_layers, n_heads, _ = num_layers_and_heads(lm.model)
    heads_cfg = cfg.get("heads") or {}
    src = heads_cfg.get("source")
    if src is None:
        # condition is not baseline but no heads block — allowed for
        # neuron-only / steering-only / circuit-only conditions.
        return None

    if src == "file":
        data = json_load(heads_cfg["path"])
        if isinstance(data, dict):
            data = (
                data.get("ranked_heads")
                or data.get("dataset_ranking")
                or data.get("selected_heads")
                or []
            )
        ranked = data[: int(heads_cfg.get("top_k", 10))]
        return [(int(h["layer"]), int(h["head"])) for h in ranked]

    if src == "uniform_random":
        return uniform_random(n_layers, n_heads,
                              k=int(heads_cfg["k"]),
                              seed=int(cfg.get("seed", 0)))
    if src == "layer_matched":
        ref = json_load(heads_cfg["reference_path"])
        ranked = (
            ref.get("ranked_heads")
            or ref.get("dataset_ranking")
            or ref.get("selected_heads")
            or []
            if isinstance(ref, dict)
            else ref
        )
        ranked = ranked[: int(heads_cfg.get("top_k", 10))]
        ref_heads = [(int(h["layer"]), int(h["head"])) for h in ranked]
        return layer_matched(ref_heads, n_heads_per_layer=n_heads,
                             seed=int(cfg.get("seed", 0)))
    if src == "explicit":
        return [(int(h["layer"]), int(h["head"])) for h in heads_cfg["heads"]]

    raise ValueError(f"unknown heads.source {src!r}")


def _resolve_neurons(cfg) -> list[tuple[int, int]] | None:
    """Read the neuron set for the active condition."""
    ncfg = cfg.get("neurons")
    if not ncfg:
        return None
    src = ncfg.get("source", "file")
    if src == "file":
        data = json_load(ncfg["path"])
        if isinstance(data, dict):
            data = data.get("ranked_neurons") or data.get("dataset_ranking") or []
        top_k = int(ncfg.get("top_k", 32))
        return [(int(n["layer"]), int(n["neuron"])) for n in data[:top_k]]
    if src == "explicit":
        return [(int(n["layer"]), int(n["neuron"])) for n in ncfg["neurons"]]
    raise ValueError(f"unknown neurons.source {src!r}")


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

    rows = _load_dataset(cfg.dataset.name, cfg.dataset.get("n"),
                          categories=cfg.dataset.get("categories"),
                          n_per_category=cfg.dataset.get("n_per_category"))
    log.info("loaded %d prompts from %s", len(rows), cfg.dataset.name)

    if args.dry_run:
        log.info("DRY-RUN: would load model %s and generate %d completions for condition=%s",
                 cfg.model.name, len(rows), cfg.condition)
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
    heads = _resolve_heads(cfg, lm)
    mask_cfg = None
    if heads is not None:
        mcfg = cfg.get("mask") or {}
        mask_cfg = build_mask_cfg(
            heads,
            mask_qkv=tuple(mcfg.get("mask_qkv", ["q"])),
            mask_type=mcfg.get("mask_type", "scale_mask"),
            scale_factor=float(mcfg.get("scale_factor", 1e-4)),
        )

    # ---- neuron ablation (Wang et al. 2024) -------------------------------
    neurons = _resolve_neurons(cfg)
    neuron_cfg = None
    if neurons is not None:
        ncfg_block = cfg.get("neurons") or {}
        neuron_cfg = build_neuron_mask_cfg(
            neurons,
            mask_type=ncfg_block.get("mask_type", "scale_mask"),
            scale_factor=float(ncfg_block.get("scale_factor", 0.0)),
        )

    # ---- direction steering (Arditi et al. 2024) --------------------------
    steering_cfg = None
    scfg_block = cfg.get("steering")
    if scfg_block:
        n_layers_total, _, _ = num_layers_and_heads(lm.model)
        layer = int(scfg_block.get("layer", n_layers_total // 2))
        steering_cfg = build_steering_cfg_from_file(
            lm,
            direction_path=scfg_block["direction_path"],
            layer=layer,
            mode=scfg_block.get("mode", "ablate"),
            alpha=float(scfg_block.get("alpha", 1.0)),
            layers=scfg_block.get("layers"),
        )

    # ---- circuit (heads ∪ neurons ∪ steering, all from disk) --------------
    ccfg_block = cfg.get("circuit")
    if ccfg_block:
        built = build_circuit_cfgs(
            lm,
            heads_path=ccfg_block.get("heads_path"),
            top_heads=int(ccfg_block.get("top_heads", 8)),
            mask_qkv=tuple(ccfg_block.get("mask_qkv", ["q"])),
            head_mask_type=ccfg_block.get("head_mask_type", "scale_mask"),
            head_scale_factor=float(ccfg_block.get("head_scale_factor", 1e-4)),
            neurons_path=ccfg_block.get("neurons_path"),
            top_neurons=int(ccfg_block.get("top_neurons", 32)),
            neuron_scale_factor=float(ccfg_block.get("neuron_scale_factor", 0.0)),
            direction_path=ccfg_block.get("direction_path"),
            direction_layer=ccfg_block.get("direction_layer"),
            steering_mode=ccfg_block.get("steering_mode", "ablate"),
        )
        if built["head_cfg"] is not None:
            mask_cfg = built["head_cfg"]
            heads = built["heads"]
        if built["neuron_cfg"] is not None:
            neuron_cfg = built["neuron_cfg"]
            neurons = built["neurons"]
        if built["steering_cfg"] is not None:
            steering_cfg = built["steering_cfg"]

    decoding = DecodingConfig(**(cfg.get("decoding") or {}))
    extra_meta = {"config_path": str(args.config)}
    if heads is not None:
        mcfg = cfg.get("mask") or {}
        extra_meta.update({
            "ablated_heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in heads
            ],
            "n_ablated_heads": len(heads),
            "mask_qkv": list(mcfg.get("mask_qkv", ["q"])),
            "mask_type": mcfg.get("mask_type", "scale_mask"),
            "scale_factor": float(mcfg.get("scale_factor", 1e-4)),
        })
    if neurons is not None:
        extra_meta.update({
            "ablated_neurons": [
                {"layer": int(l), "neuron": int(n)} for (l, n) in neurons
            ],
            "n_ablated_neurons": len(neurons),
            "neuron_scale_factor": float(neuron_cfg.get("scale_factor", 0.0)) if neuron_cfg else None,
            "neuron_method_citation": "Wang et al. 2024, Finding Safety Neurons in LLMs",
        })
    if steering_cfg is not None:
        extra_meta.update({
            "steering_mode": steering_cfg.get("mode"),
            "steering_alpha": float(steering_cfg.get("alpha", 1.0)),
            "steering_layers": list(steering_cfg.get("layers", [])),
            "steering_method_citation": (
                "Arditi et al. 2024, Refusal in LLMs is Mediated by a Single Direction"
                if steering_cfg.get("mode") == "ablate"
                else "Turner et al. 2023, Activation Addition / Zou et al. 2023 RepE"
            ),
        })

    out_rows = generate(
        lm, rows, decoding,
        mask_cfg=mask_cfg,
        neuron_cfg=neuron_cfg,
        steering_cfg=steering_cfg,
        system_prompt=cfg.get("system_prompt"),
        batch_size=int(cfg.get("batch_size", 4)),
        condition_label=cfg.condition,
        extra_meta=extra_meta,
    )
    jsonl_write(out_dir / f"completions_{cfg.condition}.jsonl", out_rows)
    log.info("wrote %d completions to %s", len(out_rows), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
