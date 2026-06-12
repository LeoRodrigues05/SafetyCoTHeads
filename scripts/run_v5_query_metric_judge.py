"""Run Direction A v5 judge shards from a query+metric manifest.

Each manifest row describes one query and one metric. This script can process
one or more consecutive manifest rows in a single worker process; each row
still writes a separate shard so merges remain per-query/per-metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402

from safety_cot_heads.direction_a import (  # noqa: E402
    build_cot_only_inputs,
    build_prefix_rows,
)
from safety_cot_heads.judging import JudgeConfig, judge_rows  # noqa: E402
from safety_cot_heads.models import load_model  # noqa: E402
from safety_cot_heads.utils import (  # noqa: E402
    ensure_dir,
    jsonl_read,
    jsonl_write,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jcfg(cfg, *, kind: str, label: str | None = None,
          max_new_tokens: int | None = None) -> JudgeConfig:
    return JudgeConfig(
        kind=kind,
        label=label,
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None
                           else cfg.get("max_new_tokens", 96)),
        base_temperature=float(cfg.get("base_temperature", 0.0)),
        retry_temperature=float(cfg.get("retry_temperature", 0.3)),
        max_retries=int(cfg.get("max_retries", 2)),
        seed=int(cfg.get("seed", 0)),
        batch_size=int(cfg.get("batch_size", 8)),
        use_chat_template=bool(cfg.get("use_chat_template", True)),
    )


def _expected_input_ids(task: dict, completion: dict) -> set[str]:
    kind = task["metric_kind"]
    if kind == "safety_single":
        return {str(_safety_input(completion)["id"])}
    if kind == "pathway_single":
        return {str(r["id"]) for r in build_prefix_rows([completion])}
    if kind == "cot_only":
        cot_inputs = build_cot_only_inputs([completion])
        return {str(r["id"]) for r in cot_inputs}
    raise ValueError(f"unknown metric_kind={kind!r}")


def _existing_complete(path: Path, task: dict, completion: dict) -> bool:
    if not path.exists():
        return False
    expected_ids = _expected_input_ids(task, completion)
    try:
        rows = list(jsonl_read(path))
    except Exception as exc:
        print(f"rerun malformed shard: {path} ({exc})")
        return False
    ids = {str(r.get("id")) for r in rows}
    if len(rows) != len(expected_ids) or ids != expected_ids:
        print(
            f"rerun incomplete shard: {path} "
            f"rows={len(rows)} expected={len(expected_ids)}"
        )
        return False
    return True


def _completion_for_task(task: dict, cache: dict[str, list[dict]]) -> dict:
    p = task["completions_path"]
    if p not in cache:
        cache[p] = list(jsonl_read(p))
    rows = cache[p]
    idx = int(task["query_index"])
    if 0 <= idx < len(rows) and str(rows[idx].get("id")) == str(task["query_id"]):
        return rows[idx]
    for row in rows:
        if str(row.get("id")) == str(task["query_id"]):
            return row
    raise KeyError(f"query_id={task['query_id']!r} not found in {p}")


def _load_judge(cfg):
    return load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        attach_controllers=False,
    )


def _reattach_metadata(judged: list[dict], source_rows: list[dict]) -> list[dict]:
    by_id = {str(r["id"]): r for r in source_rows}
    passthrough = (
        "parent_id", "prompt", "completion", "seed",
        "traj_prefix_idx", "traj_prefix_kind", "traj_is_answer",
        "traj_segments_kind", "traj_n_prose_sentences",
        "traj_n_think_sentences",
    )
    out: list[dict] = []
    for row in judged:
        src = by_id.get(str(row.get("id")))
        if src:
            for k in passthrough:
                if k in src and k not in row:
                    row[k] = src[k]
        out.append(row)
    return out


def _safety_input(completion: dict) -> dict:
    return {
        "id": completion["id"],
        "prompt": completion.get("prompt") or completion.get("user_prompt") or "",
        "completion": completion.get("completion") or "",
        "condition": completion.get("condition"),
        "model": completion.get("model"),
        "dataset": completion.get("dataset"),
        "category": completion.get("category"),
        "seed": completion.get("seed"),
    }


def run_task(task: dict, cfg, judge, completion_cache: dict[str, list[dict]],
             *, force: bool = False) -> None:
    out_path = Path(task["shard_path"])
    completion = _completion_for_task(task, completion_cache)
    if _existing_complete(out_path, task, completion) and not force:
        print(f"skip complete shard: {out_path}")
        return

    kind = task["metric_kind"]
    label = task.get("metric_label")

    if kind == "safety_single":
        inputs = [_safety_input(completion)]
        judged = judge_rows(
            judge,
            inputs,
            _jcfg(cfg, kind="safety_single", label=label),
        )
    elif kind == "pathway_single":
        inputs = build_prefix_rows([completion])
        judged = judge_rows(
            judge,
            inputs,
            _jcfg(cfg, kind="pathway_single", label=label),
        )
        judged = _reattach_metadata(judged, inputs)
    elif kind == "cot_only":
        cot_inputs = build_cot_only_inputs([completion])
        inputs = [{**c, "completion": c.get("response", "")}
                  for c in cot_inputs]
        if inputs:
            judged = judge_rows(
                judge,
                inputs,
                _jcfg(cfg, kind="cot_only", max_new_tokens=128),
            )
            judged = _reattach_metadata(judged, inputs)
        else:
            judged = []
    else:
        raise ValueError(f"unknown metric_kind={kind!r}")

    ensure_dir(out_path.parent)
    jsonl_write(out_path, judged)
    print(f"wrote {len(judged)} rows -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-id", type=int, required=True,
                    help="Worker index, usually SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--tasks-per-worker", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    start = int(args.task_id) * max(1, int(args.tasks_per_worker))
    end = min(len(manifest), start + max(1, int(args.tasks_per_worker)))
    tasks = manifest[start:end]
    if not tasks:
        print(f"no tasks for worker {args.task_id}; manifest has {len(manifest)} rows")
        return 0

    model_key = tasks[0]["model_key"]
    if any(t["model_key"] != model_key for t in tasks):
        raise ValueError("a worker chunk must not span multiple model keys")

    cfg_path = (
        ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr"
        / model_key / "judge.yaml"
    )
    cfg = load_cfg(cfg_path)
    set_seed(int(cfg.get("seed", 0)))
    judge = _load_judge(cfg)
    completion_cache: dict[str, list[dict]] = {}

    print(
        f"worker={args.task_id} tasks={start}:{end} "
        f"model_key={model_key} n={len(tasks)}"
    )
    for task in tasks:
        run_task(task, cfg, judge, completion_cache, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
