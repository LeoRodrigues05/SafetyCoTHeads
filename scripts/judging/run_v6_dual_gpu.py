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
  python scripts/judging/run_v6_dual_gpu.py --stage answer --gpus 2 --backend hf \
      --batch-size 96 --max-new-tokens 384
  python scripts/judging/run_v6_dual_gpu.py --stage monitor --prose-prefix ...
  python scripts/judging/run_v6_dual_gpu.py --stage answer --plan-only    # CPU, no model
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import v6_common as C
from run_v6_judge_shard import build_inputs, v6_judge_dir, STAGE_KIND

TRACE_STAGES = ("monitor", "pathway", "pathway14b", "safety-reasoning")

PATHWAY_14B = "models/pathway_judge_14b_merged"
DEFAULT_JUDGE = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _pathway_labels():
    from safety_cot_heads.judging.judge_prompts import PATHWAY_LABELS
    return PATHWAY_LABELS


def _label_file(out_path: Path, label: str) -> Path:
    return out_path.with_name(f"judge_pathway__{label}.jsonl")


def _out_name(stage: str, prose_prefix: bool) -> str:
    name = STAGE_KIND[stage][2]
    if prose_prefix and stage in TRACE_STAGES:
        name = name.replace(".jsonl", "__prefix.jsonl")
    return name


def _cell_todo(cell: C.Cell, stage: str, text_field: str, out_name: str,
               prose_prefix: bool, prune: bool = False, answered_only: bool = False):
    """Return (rows_to_judge, out_path, all_rows) for a cell after resume-filtering."""
    rows = build_inputs(cell, text_field, "answer" if stage == "coherence" else stage,
                        prose_prefix, answered_only=answered_only)
    out_path = v6_judge_dir(cell) / out_name
    if stage == "pathway14b":
        # one todo list per label; rows already judged for every label are done
        todo_by_label = {}
        for label in _pathway_labels():
            lp = _label_file(out_path, label)
            t, stale = C.resume_todo(rows, lp)
            if stale and prune:
                C.prune_rows(lp, stale, tag=C.utcnow_iso()[:10])
            if t:
                todo_by_label[label] = t
        return todo_by_label, out_path, rows
    # P0.7: done = same id judged on the same input; rows whose input changed
    # are re-judged (their stale outputs are pruned, with a backup, first).
    todo, stale = C.resume_todo(rows, out_path)
    if stale and prune:
        C.prune_rows(out_path, stale, tag=C.utcnow_iso()[:10])
    return todo, out_path, rows


def _n_todo(todo) -> int:
    return sum(len(v) for v in todo.values()) if isinstance(todo, dict) else len(todo)


def _merge_pathway14b(out_path: Path, rows: list[dict]) -> int:
    """Merge the 12 per-label files into judge_pathway.jsonl for this cell.

    Only per-label rows judged on the CURRENT inputs are merged. A pre-existing
    judge_pathway.jsonl produced by another judge (the 30B multi-label pass) is
    preserved once as judge_pathway__30b_multilabel.jsonl.
    """
    from safety_cot_heads.judging.merge import merge_pathway_single_label
    want = {str(r["id"]): r.get("input_sha256") for r in rows}
    per_label = {}
    for label in _pathway_labels():
        lp = _label_file(out_path, label)
        per_label[label] = [x for x in C.read_jsonl(lp)
                            if str(x.get("id")) in want
                            and x.get("input_sha256") == want[str(x.get("id"))]]
    merged = merge_pathway_single_label(per_label)
    for m in merged:
        m["input_sha256"] = want.get(str(m["id"]))
        m["pathway_protocol"] = "14b_single_label"
    if out_path.exists():
        first = next(iter(C.read_jsonl(out_path)), {})
        if first.get("judge_kind_source") != "single_label_merge":
            bak = out_path.with_name("judge_pathway__30b_multilabel.jsonl")
            if not bak.exists():
                out_path.replace(bak)
    C.write_jsonl(out_path, merged)
    return len(merged)


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
        todo, out_path, rows = _cell_todo(cell, args.stage, text_field, out_name,
                                          args.prose_prefix, prune=True,
                                          answered_only=args.answered_only)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.stage == "pathway14b":
            for label, t in todo.items():
                lcfg = JudgeConfig(kind="pathway_single", label=label,
                                   batch_size=args.batch_size,
                                   max_new_tokens=args.max_new_tokens,
                                   base_temperature=0.0, retry_temperature=0.3,
                                   max_retries=2, seed=0, use_chat_template=True)
                judge_rows(judge, t, lcfg, out_path=str(_label_file(out_path, label)))
            _merge_pathway14b(out_path, rows)
        elif todo:
            judge_rows(judge, todo, cfg, out_path=str(out_path))
        n_rows += _n_todo(todo)
        n_cells += 1
        done_q.put((gpu_id, cell.key, _n_todo(todo)))
    print(f"[gpu{gpu_id}] finished: {n_cells} cells, {n_rows} rows judged", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["answer", "monitor", "pathway", "pathway14b", "safety-reasoning"],
                    help="pathway14b = validated pathway protocol (14B judge, single-label); "
                         "pathway = legacy multi-label prompt (unvalidated)")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict to these conditions (default: every cell on disk)")
    ap.add_argument("--judge-model", default=None,
                    help=f"default: {PATHWAY_14B} for pathway14b, {DEFAULT_JUDGE} otherwise")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prose-prefix", action="store_true")
    ap.add_argument("--answered-only", action="store_true",
                    help="trace stages: skip traces with no final answer (truncated/looping)")
    ap.add_argument("--plan-only", action="store_true",
                    help="CPU: enumerate cells + rows-to-judge, no model, no GPU")
    args = ap.parse_args()
    if args.judge_model is None:
        args.judge_model = PATHWAY_14B if args.stage == "pathway14b" else DEFAULT_JUDGE
    if args.stage == "pathway":
        print("[dual] WARNING: --stage pathway is the legacy multi-label protocol; the "
              "validated pathway judge is --stage pathway14b", flush=True)

    scope = C.load_paper_scope()
    explicit = set(scope["explicit_trace_models"])
    prose_models = set(scope["prose_prefix_models"])
    cells = C.discover_cells(args.models, args.datasets)
    if args.conditions:
        cells = [c for c in cells if c.condition in set(args.conditions)]
    if args.stage in TRACE_STAGES:
        # explicit-trace pass: models with a real <think> trace.
        # prose-prefix sensitivity pass: prose-only models (never pooled with the
        # explicit-trace results). Filtering here avoids redundantly re-judging
        # explicit models' traces into the __prefix files.
        keep = prose_models if args.prose_prefix else explicit
        cells = [c for c in cells if c.model in keep]

    kind, _tf_default, _ = STAGE_KIND[args.stage]
    text_field = "answer_text" if args.stage == "answer" else "trace_text"
    out_name = _out_name(args.stage, args.prose_prefix)

    # Order cells by descending work (rows to judge) so the big cells go first and
    # the tail is small cells — minimises end-of-stage imbalance.
    plan = []
    total_todo = 0
    for c in cells:
        todo, _, _ = _cell_todo(c, args.stage, text_field, out_name, args.prose_prefix,
                                answered_only=args.answered_only)
        n = _n_todo(todo)
        if n:
            plan.append((c, n))
            total_todo += n
    plan.sort(key=lambda t: -t[1])

    print(f"[dual] stage={args.stage} judge={args.judge_model} backend={args.backend} gpus={args.gpus} "
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
