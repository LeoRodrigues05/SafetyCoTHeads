"""Complete missing Direction A v5 generation cells for one model key."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402

from safety_cot_heads.data import (  # noqa: E402
    load_alpaca,
    load_beavertails,
    load_jailbreakbench,
    load_maliciousinstruct,
    load_xstest,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_dataset_count(cfg) -> int:
    name = cfg.dataset.name
    if name == "maliciousinstruct":
        rows = load_maliciousinstruct(n=cfg.dataset.get("n"))
    elif name == "jailbreakbench":
        rows = load_jailbreakbench(n=cfg.dataset.get("n"))
    elif name == "alpaca":
        rows = load_alpaca(n=cfg.dataset.get("n"))
    elif name == "xstest":
        rows = load_xstest(n=cfg.dataset.get("n"))
    elif name == "beavertails":
        rows = load_beavertails(
            categories=cfg.dataset.get("categories"),
            n_per_category=cfg.dataset.get("n_per_category"),
        )
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return len(rows)


def _iter_generation_configs(model_key: str, datasets: set[str] | None):
    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / model_key
    gen_dir = cfg_dir / "gen"
    if not gen_dir.exists():
        raise FileNotFoundError(f"missing generation config dir: {gen_dir}")
    for dset_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        dkey = dset_dir.name
        if datasets and dkey not in datasets:
            continue
        for gcfg in sorted(dset_dir.glob("*.yaml")):
            yield dkey, gcfg.stem, gcfg


def _completion_file(cfg) -> Path:
    out_dir = ROOT / Path(cfg.output.dir)
    preferred = out_dir / f"completions_{cfg.condition}.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(out_dir.glob("completions_*.jsonl"))
    return matches[0] if matches else preferred


def _jsonl_count(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, "missing"
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                n += 1
    except Exception as exc:
        return n, f"invalid: {exc}"
    if n == 0:
        return 0, "zero"
    return n, "ok"


def _print_table(rows: list[dict]) -> None:
    print("dataset condition             expected present status")
    print("------- --------------------- -------- ------- --------")
    for r in rows:
        print(
            f"{r['dataset']:<7} {r['condition']:<21} "
            f"{r['expected']:>8} {r['present']:>7} {r['status']}"
        )


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key", metavar="MODEL_KEY")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Optional subset, e.g. --datasets jbb bt")
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="Forwarded to run_generation, e.g. --overrides batch_size=32")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = set(args.datasets) if args.datasets else None
    rows: list[dict] = []
    to_run: list[tuple[Path, list[str]]] = []

    for dkey, cond, gcfg in _iter_generation_configs(args.model_key, datasets):
        cfg = load_cfg(gcfg)
        expected = _load_dataset_count(cfg)
        comp = _completion_file(cfg)
        present, state = _jsonl_count(comp)
        complete = state == "ok" and present == expected
        status = "complete" if complete else state
        if state == "ok" and present != expected:
            status = "partial"
        rows.append({
            "dataset": dkey,
            "condition": cond,
            "expected": expected,
            "present": present,
            "status": status,
            "config": str(gcfg),
        })
        if not complete:
            cmd = [
                sys.executable,
                "-m",
                "scripts.run_generation",
                "--config",
                str(gcfg),
            ]
            if args.overrides:
                cmd += ["--overrides", *args.overrides]
            to_run.append((gcfg, cmd))

    _print_table(rows)
    if not to_run:
        print("generation already complete")
        return 0

    print(f"generation cells to run: {len(to_run)}")
    for _gcfg, cmd in to_run:
        print(_quote_cmd(cmd))

    if args.dry_run:
        return 0

    for _gcfg, cmd in to_run:
        subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
