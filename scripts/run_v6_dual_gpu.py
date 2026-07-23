#!/usr/bin/env python3
"""Dynamic dual-GPU judge runner — keeps BOTH B200s saturated via a shared queue.

Static N-way sharding (run_v6_judge_shard.py) assigns each GPU a fixed set of
cells, so whichever GPU draws the lighter/faster half finishes early and idles
for the rest of the stage. This runner replaces that with a **work queue**: one
persistent worker per GPU loads the judge once, then repeatedly pulls the next
unfinished cell from a shared queue and judges it. Whichever GPU is free grabs
the next cell, so load auto-balances and neither GPU idles until the queue is
nearly empty.

Correctness / safety:
  * Each cell is judged into its own per-cell output file (``judge_rows`` appends
    incrementally), so a crash loses at most the in-flight cell and re-running
    resumes (already-judged ids per cell are skipped).
  * CUDA_VISIBLE_DEVICES is set in each worker BEFORE any torch import, pinning it
    to exactly one GPU. Two workers never share a GPU.
  * Reads only runs/direction_a_v5 + runs/direction_a_v6/parsed; writes only the
    per-cell judge outputs under runs/direction_a_v6/judge.

Usage:
  python scripts/run_v6_dual_gpu.py --stage answer --gpus 2 --backend hf \
      --batch-size 96 --max-new-tokens 384
  python scripts/run_v6_dual_gpu.py --stage monitor --prose-prefix ...
  python scripts/run_v6_dual_gpu.py --stage answer --plan-only    # CPU, no model
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import v6_common as C
from run_v6_judge_shard import build_inputs, v6_judge_dir, STAGE_KIND

TRACE_STAGES = ("monitor", "pathway", "safety-reasoning")


def _out_name(stage: str, prose_prefix: bool) -> str:
    name = STAGE_KIND[stage][2]
    if prose_prefix and stage in TRACE_STAGES:
        name = name.replace(".jsonl", "__prefix.jsonl")
    return name


def _cell_todo(cell: C.Cell, stage: str, text_field: str, out_name: str,
               prose_prefix: bool):
    """Return (rows_to_judge, out_path) for a cell after resume-filtering."""
    rows = build_inputs(cell, text_field, "answer" if stage == "coherence" else stage,
                        prose_prefix)
    out_path = v6_judge_dir(cell) / out_name
    done = {str(r.get("id")) for r in C.read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if str(r["id"]) not in done]
    return todo, out_path


def _worker(gpu_id: int, task_q, done_q, args, kind: str, text_field: str, out_name: str):
    # Pin to one GPU BEFORE importing torch (spawn: fresh interpreter).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from run_v6_judge_shard import _load_judge
    from safety_cot_heads.judging import JudgeConfig, judge_rows

    judge = _load_judge(args)
    cfg = JudgeConfig(kind=kind, batch_size=args.batch_size,
                      max_new_tokens=args.max_new_tokens, base_temperature=0.0, seed=0)
    print(f"[gpu{gpu_id}] judge loaded via {args.backend}; pulling cells", flush=True)

    n_cells = n_rows = 0
    while True:
        item = task_q.get()
        if item is None:
            break
        cell = C.Cell(**item)
        todo, out_path = _cell_todo(cell, args.stage, text_field, out_name, args.prose_prefix)
        if todo:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            judge_rows(judge, todo, cfg, out_path=str(out_path))
            n_rows += len(todo)
        n_cells += 1
        done_q.put((gpu_id, cell.key, len(todo)))
    print(f"[gpu{gpu_id}] finished: {n_cells} cells, {n_rows} rows judged", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["answer", "monitor", "pathway", "safety-reasoning"])
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prose-prefix", action="store_true")
    ap.add_argument("--plan-only", action="store_true",
                    help="CPU: enumerate cells + rows-to-judge, no model, no GPU")
    args = ap.parse_args()

    scope = C.load_paper_scope()
    explicit = set(scope["explicit_trace_models"])
    prose_models = set(scope["prose_prefix_models"])
    cells = C.discover_cells(args.models, args.datasets)
    if args.stage in TRACE_STAGES:
        # explicit-trace pass: models with a real <think> trace.
        # prose-prefix sensitivity pass: prose-only models (never pooled with the
        # explicit-trace results). Filtering here avoids redundantly re-judging
        # explicit models' traces into the __prefix files.
        keep = prose_models if args.prose_prefix else explicit
        cells = [c for c in cells if c.model in keep]

    kind, tf_default, _ = STAGE_KIND[args.stage]
    text_field = "answer_text" if args.stage == "answer" else "trace_text"
    out_name = _out_name(args.stage, args.prose_prefix)

    # Order cells by descending work (rows to judge) so the big cells go first and
    # the tail is small cells — minimises end-of-stage imbalance.
    plan = []
    total_todo = 0
    for c in cells:
        todo, _ = _cell_todo(c, args.stage, text_field, out_name, args.prose_prefix)
        if todo:
            plan.append((c, len(todo)))
            total_todo += len(todo)
    plan.sort(key=lambda t: -t[1])

    print(f"[dual] stage={args.stage} backend={args.backend} gpus={args.gpus} "
          f"cells_with_work={len(plan)} rows_to_judge={total_todo} out={out_name}")
    if args.plan_only:
        for c, n in plan[:10]:
            print(f"   {c.key}: {n} rows")
        print(f"[dual] plan-only: {len(plan)} cells, {total_todo} rows (no model loaded)")
        return
    if not plan:
        print("[dual] nothing to judge (all cells already complete)")
        return

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    done_q = ctx.Queue()
    for c, _ in plan:
        task_q.put({"model": c.model, "dataset": c.dataset,
                    "condition": c.condition, "seed": c.seed})
    for _ in range(args.gpus):
        task_q.put(None)                       # one stop sentinel per worker

    workers = [ctx.Process(target=_worker,
                           args=(g, task_q, done_q, args, kind, text_field, out_name))
               for g in range(args.gpus)]
    for w in workers:
        w.start()

    # progress: consume done_q until all cells reported (or workers all dead)
    done = 0
    t0 = time.time()
    while done < len(plan):
        if not any(w.is_alive() for w in workers) and done_q.empty():
            print("[dual] all workers exited before finishing the queue", flush=True)
            break
        try:
            gpu_id, key, n = done_q.get(timeout=5)
            done += 1
            if done % 10 == 0 or done == len(plan):
                rate = done / max(1e-9, (time.time() - t0))
                print(f"[dual] {done}/{len(plan)} cells done "
                      f"({rate*60:.1f} cells/min)", flush=True)
        except Exception:
            continue

    for w in workers:
        w.join()
    codes = [w.exitcode for w in workers]
    print(f"[dual] stage={args.stage} complete; worker exit codes={codes}")
    if any(c not in (0, None) for c in codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
