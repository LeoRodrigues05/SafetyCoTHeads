"""Direction A v5 — expand iso-ASR matrix into per-cell YAMLs.

Reads `configs/experiments/direction_a_v5_iso_asr/matrix.yaml` and writes,
per model:

  configs/experiments/direction_a_v5_iso_asr/<model_key>/
    01-ships-discovery.yaml          (unless reuse_discovery_*)
    16-neuron-discovery.yaml         (unless reuse_discovery_*)
    17-direction-extraction.yaml     (unless reuse_discovery_*)
    gen/baseline.yaml
    gen/ships_top{k}.yaml            x len(top_k)
    gen/neurons_top{k}.yaml          x len(top_k)
    gen/steering_a{alpha}.yaml       x len(alpha)
    judge.yaml

Idempotent: rewrites every file every run. Safe.

Usage:
    python -m scripts.make_v5_configs \
        --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg  # noqa: E402

import yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "configs" / "models.yaml"


def _model_name(ref: str) -> str:
    """Resolve a models.yaml key to the actual HF id."""
    m = yaml.safe_load(MODELS_YAML.read_text())
    if ref not in m:
        raise KeyError(f"model_ref={ref!r} not found in {MODELS_YAML}")
    return m[ref]["name"]


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _model_block(ref: str) -> dict:
    """Shared model: block for generation/discovery configs."""
    return {
        "name": _model_name(ref),
        "dtype": "bfloat16",
        "attn_implementation": "eager",
    }


def _runs_dir(model_key: str) -> str:
    return f"runs/direction_a_v5/{model_key}"


# ---------------------------------------------------------------------------
# discovery configs
# ---------------------------------------------------------------------------
def _ships_disc(model_key: str, ref: str) -> dict:
    return {
        "tracker": {"experiment": "direction_a_v5",
                    "run": f"ships_discovery_{model_key}",
                    "status": "runnable"},
        "seed": 0,
        "method": "ships",
        "model": _model_block(ref),
        "dataset": {"name": "maliciousinstruct", "n": 100},
        "method_args": {
            "mask_qkv": ["q"],
            "mask_type": "scale_mask",
            "scale_factor": 1.0e-4,
            "top_k": 10,
            "prompt_template": "## Query:{q}\n## Answer:",
        },
        "output": {"dir": f"{_runs_dir(model_key)}/01-ships-discovery",
                   "top_k": 16},
    }


def _neuron_disc(model_key: str, ref: str) -> dict:
    return {
        "tracker": {"experiment": "direction_a_v5",
                    "run": f"neuron_discovery_{model_key}",
                    "status": "runnable"},
        "seed": 0,
        "model": _model_block(ref),
        "harmful": {"name": "maliciousinstruct", "n": 100},
        "benign": {"name": "alpaca", "n": 100},
        "batch_size": 4,
        "max_length": 512,
        "capture_last_n": 1,
        "top_k_default": 32,
        "output": {"dir": f"{_runs_dir(model_key)}/16-neuron-discovery"},
    }


def _direction_disc(model_key: str, ref: str) -> dict:
    return {
        "tracker": {"experiment": "direction_a_v5",
                    "run": f"direction_extraction_{model_key}",
                    "status": "runnable"},
        "seed": 0,
        "model": _model_block(ref),
        "harmful": {"name": "maliciousinstruct", "n": 100},
        "benign": {"name": "alpaca", "n": 100},
        "batch_size": 4,
        "max_length": 512,
        "capture_last_n": 1,
        "output": {"dir": f"{_runs_dir(model_key)}/17-direction-extraction"},
    }


# ---------------------------------------------------------------------------
# generation configs
# ---------------------------------------------------------------------------
def _decoding(matrix: dict, model_cfg: dict) -> dict:
    d = dict(matrix["decoding"])
    d["max_new_tokens"] = int(model_cfg.get("max_new_tokens", 512))
    return d


def _gen_common(model_key: str, ref: str, model_cfg: dict, matrix: dict,
                cond: str, dset_key: str, dset: dict) -> dict:
    out = {
        "tracker": {"experiment": "direction_a_v5",
                    "run": f"{cond}_{model_key}_{dset_key}",
                    "status": "runnable"},
        "seed": 0,
        "condition": cond,
        "model": _model_block(ref),
        "dataset": dict(dset),
        "decoding": _decoding(matrix, model_cfg),
        "batch_size": int(model_cfg.get("batch_size", 4)),
        "output": {"dir": f"{_runs_dir(model_key)}/gen/{dset_key}/{cond}/seed0"},
    }
    if model_cfg.get("chat_overrides"):
        out["chat_overrides"] = dict(model_cfg["chat_overrides"])
    return out


def _ships_path(model_key: str, model_cfg: dict) -> str:
    paths = model_cfg.get("reuse_discovery_paths") or {}
    if "ships" in paths:
        return paths["ships"]
    reuse = model_cfg.get("reuse_discovery_from")
    src = reuse if reuse else model_key
    return f"{_runs_dir(src)}/01-ships-discovery/ships_dataset_ranking.json"


def _neuron_path(model_key: str, model_cfg: dict) -> str:
    paths = model_cfg.get("reuse_discovery_paths") or {}
    if "neurons" in paths:
        return paths["neurons"]
    reuse = model_cfg.get("reuse_discovery_from")
    src = reuse if reuse else model_key
    return f"{_runs_dir(src)}/16-neuron-discovery/neuron_ranking.json"


def _direction_path(model_key: str, model_cfg: dict) -> str:
    paths = model_cfg.get("reuse_discovery_paths") or {}
    if "direction" in paths:
        return paths["direction"]
    reuse = model_cfg.get("reuse_discovery_from")
    src = reuse if reuse else model_key
    return f"{_runs_dir(src)}/17-direction-extraction/refusal_directions.npz"


def _gen_baseline(model_key, ref, model_cfg, matrix, dset_key, dset) -> dict:
    cfg = _gen_common(model_key, ref, model_cfg, matrix, "baseline",
                       dset_key, dset)
    cfg["system_prompt"] = None
    return cfg


def _gen_ships(model_key, ref, model_cfg, matrix, dset_key, dset, k: int) -> dict:
    a = matrix["ablations"]["ships"]
    cfg = _gen_common(model_key, ref, model_cfg, matrix, f"ships_top{k}",
                       dset_key, dset)
    cfg["heads"] = {
        "source": "file",
        "path": _ships_path(model_key, model_cfg),
        "top_k": int(k),
    }
    cfg["mask"] = {
        "mask_qkv": list(a["mask_qkv"]),
        "mask_type": a["mask_type"],
        "scale_factor": float(a["scale_factor"]),
    }
    return cfg


def _gen_neurons(model_key, ref, model_cfg, matrix, dset_key, dset, k: int) -> dict:
    a = matrix["ablations"]["neurons"]
    cfg = _gen_common(model_key, ref, model_cfg, matrix, f"neurons_top{k}",
                       dset_key, dset)
    cfg["neurons"] = {
        "source": "file",
        "path": _neuron_path(model_key, model_cfg),
        "top_k": int(k),
        "mask_type": a["mask_type"],
        "scale_factor": float(a["scale_factor"]),
    }
    return cfg


def _gen_steering(model_key, ref, model_cfg, matrix, dset_key, dset,
                   alpha: float) -> dict:
    a = matrix["ablations"]["steering"]
    cfg = _gen_common(model_key, ref, model_cfg, matrix,
                       f"steering_a{alpha}", dset_key, dset)
    cfg["steering"] = {
        "direction_path": _direction_path(model_key, model_cfg),
        "layer": int(a["layer"]),
        "mode": a["mode"],
        "alpha": float(alpha),
    }
    return cfg


# ---------------------------------------------------------------------------
# judge config
# ---------------------------------------------------------------------------
def _judge_cfg(model_key: str, matrix: dict) -> dict:
    j = matrix["judge"]
    return {
        "tracker": {"experiment": "direction_a_v5",
                    "run": f"v5_judge_{model_key}",
                    "status": "runnable"},
        "model": {
            "name": _model_name(j["model_ref"]),
            "dtype": "bfloat16",
            "load_in_4bit": True,
            "device_map": "auto",
            "trust_remote_code": False,
        },
        "max_new_tokens": int(j["max_new_tokens"]),
        "base_temperature": float(j["base_temperature"]),
        "retry_temperature": float(j["retry_temperature"]),
        "max_retries": int(j["max_retries"]),
        "batch_size": int(j["batch_size"]),
        "use_chat_template": bool(j["use_chat_template"]),
        "seed": int(j["seed"]),
        "n_limit": j.get("n_limit"),
        "gate_safety_by_coherence": bool(j.get("gate_safety_by_coherence",
                                               True)),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def expand_one_model(model_key: str, model_cfg: dict, matrix: dict,
                     base_dir: Path) -> list[Path]:
    ref = model_cfg["model_ref"]
    out_dir = base_dir / model_key
    written: list[Path] = []

    # discovery (unless reusing)
    if not (model_cfg.get("reuse_discovery_from")
            or model_cfg.get("reuse_discovery_paths")):
        _write(out_dir / "01-ships-discovery.yaml",
               _ships_disc(model_key, ref))
        _write(out_dir / "16-neuron-discovery.yaml",
               _neuron_disc(model_key, ref))
        _write(out_dir / "17-direction-extraction.yaml",
               _direction_disc(model_key, ref))
        written += [out_dir / "01-ships-discovery.yaml",
                    out_dir / "16-neuron-discovery.yaml",
                    out_dir / "17-direction-extraction.yaml"]

    # generation: for each dataset, for each condition
    datasets = matrix.get("datasets") or {"jbb": matrix["dataset"]}
    gen_dir = out_dir / "gen"
    for dkey, dset in datasets.items():
        ddir = gen_dir / dkey
        _write(ddir / "baseline.yaml",
               _gen_baseline(model_key, ref, model_cfg, matrix, dkey, dset))
        written.append(ddir / "baseline.yaml")
        for k in matrix["ablations"]["ships"]["top_k"]:
            p = ddir / f"ships_top{k}.yaml"
            _write(p, _gen_ships(model_key, ref, model_cfg, matrix,
                                  dkey, dset, k))
            written.append(p)
        for k in matrix["ablations"]["neurons"]["top_k"]:
            p = ddir / f"neurons_top{k}.yaml"
            _write(p, _gen_neurons(model_key, ref, model_cfg, matrix,
                                    dkey, dset, k))
            written.append(p)
        for a in matrix["ablations"]["steering"]["alpha"]:
            p = ddir / f"steering_a{a}.yaml"
            _write(p, _gen_steering(model_key, ref, model_cfg, matrix,
                                     dkey, dset, a))
            written.append(p)

    # judge
    _write(out_dir / "judge.yaml", _judge_cfg(model_key, matrix))
    written.append(out_dir / "judge.yaml")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model keys to expand (default: all).")
    args = ap.parse_args()

    matrix = cfg_to_dict(load_cfg(args.matrix))
    base_dir = Path(args.matrix).parent

    keys = list(matrix["models"].keys())
    if args.models:
        keys = [k for k in keys if k in args.models]

    total = 0
    for k in keys:
        files = expand_one_model(k, matrix["models"][k], matrix, base_dir)
        print(f"[{k}] wrote {len(files)} files")
        total += len(files)
    print(f"== total {total} configs under {base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
