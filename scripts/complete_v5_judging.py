"""Complete Direction A v5 judging cells for one model key."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402

from safety_cot_heads.direction_a import (  # noqa: E402
    build_cot_only_inputs,
    build_prefix_rows,
)
from safety_cot_heads.judging import PATHWAY_LABELS  # noqa: E402
from safety_cot_heads.judging.judge_prompts import LABELS as SAFETY_LABELS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _iter_condition_configs(model_key: str, datasets: set[str] | None):
    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / model_key
    gen_dir = cfg_dir / "gen"
    if not gen_dir.exists():
        raise FileNotFoundError(f"missing generation config dir: {gen_dir}")
    for dset_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        dkey = dset_dir.name
        if datasets and dkey not in datasets:
            continue
        for gcfg in sorted(dset_dir.glob("*.yaml")):
            yield dkey, gcfg.stem


def _completion_file(model_key: str, dkey: str, cond: str) -> Path | None:
    seed_dir = (
        ROOT / "runs" / "direction_a_v5" / model_key
        / "gen" / dkey / cond / "seed0"
    )
    preferred = seed_dir / f"completions_{cond}.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(seed_dir.glob("completions_*.jsonl"))
    return matches[0] if matches else None


def _read_jsonl(path: Path) -> tuple[list[dict], str]:
    if not path.exists():
        return [], "missing"
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception as exc:
        return rows, f"invalid: {exc}"
    return rows, "ok"


def _json_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True


def _jsonl_ok(path: Path, expected: int | None = None) -> tuple[bool, int, str]:
    rows, state = _read_jsonl(path)
    if state != "ok":
        return False, len(rows), state
    if expected is not None and len(rows) != expected:
        return False, len(rows), f"{len(rows)}/{expected}"
    return True, len(rows), "ok"


def _cell_status(model_key: str, dkey: str, cond: str, judge_cfg) -> dict:
    comp = _completion_file(model_key, dkey, cond)
    if comp is None:
        return {
            "dataset": dkey,
            "condition": cond,
            "status": "missing-completions",
            "issues": ["missing completions"],
            "completion_rows": 0,
            "prefix_rows": 0,
            "cot_rows": 0,
            "completions_path": None,
        }

    completions, comp_state = _read_jsonl(comp)
    n_limit = judge_cfg.get("n_limit")
    if n_limit is not None:
        completions = completions[: int(n_limit)]
    if comp_state != "ok" or not completions:
        return {
            "dataset": dkey,
            "condition": cond,
            "status": "bad-completions",
            "issues": [comp_state],
            "completion_rows": len(completions),
            "prefix_rows": 0,
            "cot_rows": 0,
            "completions_path": str(comp),
        }

    prefix_rows = build_prefix_rows(completions)
    cot_rows = build_cot_only_inputs(completions)
    n_comp = len(completions)
    n_prefix = len(prefix_rows)
    n_cot = len(cot_rows)

    out_dir = (
        ROOT / "runs" / "direction_a_v5" / model_key
        / "judge" / dkey / cond / "seed0"
    )
    checks: list[tuple[str, bool, str]] = []
    checks.append(("summary.json", _json_ok(out_dir / "summary.json"), "missing"))
    for name, expected in (
        ("coherence.jsonl", n_comp),
        (f"judged_{cond}.jsonl", n_comp),
        ("prefix_rows.jsonl", n_prefix),
        ("judge_pathway.jsonl", n_prefix),
        ("judge_cot_only.jsonl", n_cot),
        ("pathway_vectors.jsonl", n_comp),
    ):
        ok, _n, detail = _jsonl_ok(out_dir / name, expected)
        checks.append((name, ok, detail))
    ok, _n, detail = _jsonl_ok(out_dir / "monitorability_rows.jsonl", None)
    checks.append(("monitorability_rows.jsonl", ok, detail))

    for label in SAFETY_LABELS:
        name = f"judge_safety__{label}.jsonl"
        ok, _n, detail = _jsonl_ok(out_dir / name, n_comp)
        checks.append((name, ok, detail))
    for label in PATHWAY_LABELS:
        name = f"judge_pathway__{label}.jsonl"
        ok, _n, detail = _jsonl_ok(out_dir / name, n_prefix)
        checks.append((name, ok, detail))

    issues = [f"{name}:{detail}" for name, ok, detail in checks if not ok]
    status = "complete" if not issues else "incomplete"
    return {
        "dataset": dkey,
        "condition": cond,
        "status": status,
        "issues": issues,
        "completion_rows": n_comp,
        "prefix_rows": n_prefix,
        "cot_rows": n_cot,
        "completions_path": str(comp),
    }


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _print_table(rows: list[dict]) -> None:
    print("dataset condition             comp prefix cot status")
    print("------- --------------------- ---- ------ --- ----------")
    for r in rows:
        print(
            f"{r['dataset']:<7} {r['condition']:<21} "
            f"{r['completion_rows']:>4} {r['prefix_rows']:>6} "
            f"{r['cot_rows']:>3} {r['status']}"
        )
        if r["issues"]:
            preview = ", ".join(r["issues"][:4])
            extra = "" if len(r["issues"]) <= 4 else f", ... +{len(r['issues']) - 4}"
            print(f"        issues: {preview}{extra}")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(_quote_cmd(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def _report_commands(model_key: str, datasets: list[str]) -> list[list[str]]:
    out_base = ROOT / "runs" / "direction_a_v5" / model_key
    cmds: list[list[str]] = []
    for dkey in datasets:
        judge_out = out_base / "judge" / dkey
        cmds.append([
            sys.executable,
            "-m",
            "scripts.make_v4_jbb_report",
            "--in-base",
            str(judge_out),
            "--out",
            str(judge_out / "v5_report.md"),
            "--title",
            f"Direction A v5 - {model_key} on {dkey} (iso-ASR)",
            "--iso-anchor",
            "steering_a1.0",
        ])
    return cmds


def _run_single(model_key: str, cells: list[dict], datasets: list[str],
                *, dry_run: bool, skip_report: bool) -> None:
    if cells:
        cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / model_key
        out_base = ROOT / "runs" / "direction_a_v5" / model_key / "judge"
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_v4_jbb_judge",
            "--config",
            str(cfg_dir / "judge.yaml"),
            "--out-base",
            str(out_base),
        ]
        for cell in cells:
            spec = (
                f"tag={cell['dataset']}/{cell['condition']},"
                f"cond={cell['condition']},"
                f"completions={cell['completions_path']}"
            )
            cmd.extend(["--condition", spec])
        _run(cmd, dry_run=dry_run)
    else:
        print("judging already complete")

    if not skip_report:
        for cmd in _report_commands(model_key, datasets):
            _run(cmd, dry_run=dry_run)


def _load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _run_split(model_key: str, cells: list[dict], datasets: list[str],
               *, dry_run: bool, skip_report: bool,
               tasks_per_worker: int) -> None:
    out_base = ROOT / "runs" / "direction_a_v5" / model_key
    manifest = out_base / "judge" / "_manifests" / "query_metric_tasks.jsonl"
    filtered = out_base / "judge" / "_manifests" / "query_metric_tasks.incomplete.jsonl"

    make_cmd = [
        sys.executable,
        "-m",
        "scripts.make_v5_judge_manifest",
        "--model-key",
        model_key,
        "--out",
        str(manifest),
    ]
    _run(make_cmd, dry_run=dry_run)
    if dry_run:
        print(f"would filter {manifest} -> {filtered}")
        return

    wanted = {(c["dataset"], c["condition"]) for c in cells}
    all_tasks = _load_manifest(manifest)
    tasks = [
        t for t in all_tasks
        if (t["dataset"], t["condition"]) in wanted
    ]
    _write_manifest(filtered, tasks)
    workers = math.ceil(len(tasks) / max(1, tasks_per_worker))
    print(f"split tasks to run: {len(tasks)} ({workers} workers)")

    for task_id in range(workers):
        _run([
            sys.executable,
            "-m",
            "scripts.run_v5_query_metric_judge",
            "--manifest",
            str(filtered),
            "--task-id",
            str(task_id),
            "--tasks-per-worker",
            str(tasks_per_worker),
        ], dry_run=False)

    for dkey, cond in sorted(wanted):
        _run([
            sys.executable,
            "-m",
            "scripts.merge_v5_query_metric_judge",
            "--model-key",
            model_key,
            "--dataset",
            dkey,
            "--condition",
            cond,
        ], dry_run=False)

    if not skip_report:
        for cmd in _report_commands(model_key, datasets):
            _run(cmd, dry_run=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key", metavar="MODEL_KEY")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Optional subset, e.g. --datasets jbb bt")
    ap.add_argument("--path", choices=("single", "split"),
                    default=os.environ.get("JUDGE_PATH", "single"),
                    help="Judging path. Defaults to $JUDGE_PATH or single.")
    ap.add_argument("--tasks-per-worker", type=int, default=16,
                    help="Split-path local worker chunk size.")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = set(args.datasets) if args.datasets else None
    cfg_dir = ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr" / args.model_key
    judge_cfg = load_cfg(cfg_dir / "judge.yaml")

    rows = [
        _cell_status(args.model_key, dkey, cond, judge_cfg)
        for dkey, cond in _iter_condition_configs(args.model_key, datasets)
    ]
    _print_table(rows)

    run_cells = [r for r in rows if r["status"] != "complete"]
    bad = [r for r in run_cells if not r["completions_path"]]
    if bad:
        missing = ", ".join(f"{r['dataset']}/{r['condition']}" for r in bad)
        raise FileNotFoundError(f"cannot judge cells without completions: {missing}")

    selected_datasets = sorted({r["dataset"] for r in rows})
    if args.path == "single":
        _run_single(
            args.model_key,
            run_cells,
            selected_datasets,
            dry_run=args.dry_run,
            skip_report=args.skip_report,
        )
    else:
        if not run_cells:
            print("judging already complete")
            if not args.skip_report:
                for cmd in _report_commands(args.model_key, selected_datasets):
                    _run(cmd, dry_run=args.dry_run)
        else:
            _run_split(
                args.model_key,
                run_cells,
                selected_datasets,
                dry_run=args.dry_run,
                skip_report=args.skip_report,
                tasks_per_worker=args.tasks_per_worker,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
