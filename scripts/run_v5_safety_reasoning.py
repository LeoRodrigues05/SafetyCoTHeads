"""Direction A v5 — safety-reasoning trace judging over FULL completions.

The standard instruments (5-label safety, coherence, CoT-only monitor, pathway)
are already judged for every model under the ``judge/`` ("full") tree, but the
indexed safety-reasoning trace judge has never been run. This driver fills that
gap: for every model it walks the full completion cells and runs *only* the
safety-reasoning judge, writing ``judge_safety_reasoning_trace.jsonl`` and
``safety_reasoning.summary.json`` straight into the existing
``judge/<dataset>/<condition>/seed0/`` directory the metrics report reads.

It is a thin driver over the building blocks in ``run_v5_qwen_subset_judging``
(no subsetting): one 30B judge load is reused across all models, and each cell is
resumable — already-judged ids are skipped, and existing standard-judge files are
never touched.

Dual-GPU: the cell plan can be sharded with ``--num-shards``/``--shard-id`` so two
processes (one pinned to each GPU via ``CUDA_VISIBLE_DEVICES``) split the work
evenly with no file collision — each shard owns a disjoint set of output dirs.

Usage:
    python -m scripts.run_v5_safety_reasoning --dry-run
    python -m scripts.run_v5_safety_reasoning
    python -m scripts.run_v5_safety_reasoning --models qwen3_8b --conditions baseline --datasets jbb
    # GPU 0 shard:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.run_v5_safety_reasoning --num-shards 2 --shard-id 0
    # GPU 1 shard:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.run_v5_safety_reasoning --num-shards 2 --shard-id 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import load_cfg  # noqa: E402
from run_v5_qwen_subset_judging import (  # noqa: E402
    _iter_generation_cells,
    _jsonl_rows,
    _load_judge,
    _run_reasoning_judge,
)

from safety_cot_heads.utils import ensure_dir, jsonl_write, set_seed  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "runs" / "direction_a_v5" / "_smoke_sr"

DEFAULT_MODELS = (
    "qwen3_8b",
    "llama31_8b_control",
    "olmo3_7b_think",
    "olmo3_7b_base",
    "olmo3_7b_base_own",
)


def _judge_cfg_path(model_key: str) -> Path:
    return (
        ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr"
        / model_key / "judge.yaml"
    )


def _collect_cells(model_key: str, datasets: set[str],
                   conditions: set[str] | None) -> list[tuple[str, str, Path]]:
    return list(_iter_generation_cells(model_key, datasets, conditions))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--datasets", nargs="+", default=["jbb", "bt"])
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1536,
                    help="Per-row generation cap. The SR judge emits one JSON object per "
                         "flagged sentence; long traces need ~750-1000 tokens to close the "
                         "array, so the old 768 truncated mid-object on ~60%% of rows. 1536 "
                         "gives headroom for ~32 spans (double the observed max); the parser "
                         "salvages any rarer overflow. The cap is a ceiling, not a target -- "
                         "rows that close early stop early, so raising it barely costs time.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override judge.yaml batch_size. KV cache at this cap is ~0.5GB/seq "
                         "for the 30B judge, so batch 32 (~17GB KV + ~61GB weights) fits a "
                         "183GB GPU with room to spare.")
    ap.add_argument("--keep-4bit", action="store_true",
                    help="Load the judge as configured in judge.yaml (4-bit). Default is to "
                         "override to bf16: the SR pass uses 768 max-new-tokens (vs 96 for the "
                         "standard passes), and 4-bit generation is far slower; bf16 fits a 30B "
                         "judge on a single 183GB GPU and is strictly higher fidelity.")
    ap.add_argument("--device", default="cuda:0",
                    help="device_map for the bf16 judge load (ignored with --keep-4bit). With "
                         "CUDA_VISIBLE_DEVICES pinning each shard to one GPU, leave this at cuda:0.")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="Generation backend. 'hf' is the default Transformers path "
                         "(static batching, length-bucketed). 'vllm' uses continuous "
                         "batching + paged KV cache, eliminating the long-trace straggler "
                         "and typically 5-15x faster for this long-output pass; requires "
                         "vllm installed in the active env.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="vLLM only: fraction of GPU memory for weights + KV cache.")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM only: max context (prompt + generation) per sequence. "
                         "SR prompts run ~2-4k tokens + the generation cap; 8192 is ample.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split the cell plan into this many disjoint shards (data parallel).")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="Which shard (0-based) this process runs. Cells are interleaved so "
                         "row counts stay balanced across shards.")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="SMOKE ONLY: judge at most N rows per cell and write under "
                         "runs/direction_a_v5/_smoke_sr/ instead of the real judge tree.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the cells and source row counts; load nothing, write nothing.")
    args = ap.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        print(f"invalid shard: shard-id={args.shard_id} must be in [0, {args.num_shards})")
        return 2

    datasets = set(args.datasets)
    conditions = set(args.conditions) if args.conditions else None

    # Plan: (model_key, dataset, condition, completion_path) across all models.
    plan: list[tuple[str, str, str, Path]] = []
    for model_key in args.models:
        for dkey, cond, comp in _collect_cells(model_key, datasets, conditions):
            plan.append((model_key, dkey, cond, comp))

    if not plan:
        print("no completion cells found for the requested models/datasets/conditions")
        return 2

    # Interleaved sharding: every num_shards-th cell. Interleaving (vs contiguous
    # slicing) keeps per-shard row counts balanced regardless of model ordering.
    if args.num_shards > 1:
        plan = [c for i, c in enumerate(plan) if i % args.num_shards == args.shard_id]
        if not plan:
            print(f"shard {args.shard_id}/{args.num_shards} is empty")
            return 0

    print(f"models={args.models}")
    print(f"shard={args.shard_id}/{args.num_shards}  cells={len(plan)}  "
          f"max_new_tokens={args.max_new_tokens}  limit_rows={args.limit_rows}")
    print("model                  dataset condition             rows completion_file")
    print("---------------------- ------- -------------------- ---- ---------------")
    total_rows = 0
    for model_key, dkey, cond, comp in plan:
        n = len(_jsonl_rows(comp))
        total_rows += min(n, args.limit_rows) if args.limit_rows else n
        print(f"{model_key:<22} {dkey:<7} {cond:<20} {n:>4} {comp}")
    print(f"shard total rows to judge: {total_rows}")

    if args.dry_run:
        print(f"DRY-RUN complete: {len(plan)} cells, {total_rows} rows; "
              "nothing loaded or written.")
        return 0

    # All five models share the Qwen3-30B-A3B judge, so load it once.
    first_cfg = load_cfg(_judge_cfg_path(args.models[0]))
    set_seed(int(first_cfg.get("seed", 0)))
    if args.backend == "vllm":
        from safety_cot_heads.judging.vllm_backend import load_vllm_judge
        model_name = first_cfg.model.name
        print(f"judge load: vllm bf16 ({model_name}); "
              f"max_model_len={args.max_model_len} gpu_mem_util={args.gpu_mem_util}")
        judge = load_vllm_judge(
            model_name,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            trust_remote_code=bool(first_cfg.model.get("trust_remote_code", False)),
            seed=int(first_cfg.get("seed", 0)),
        )
    elif not args.keep_4bit:
        # Override to bf16 single-GPU: much faster than 4-bit at long max-new-tokens.
        first_cfg.model.load_in_4bit = False
        first_cfg.model.dtype = "bfloat16"
        first_cfg.model.device_map = args.device
        print(f"judge load: bf16 on {args.device} (override of 4-bit judge.yaml; "
              f"use --keep-4bit to disable)")
        judge = _load_judge(first_cfg)
    else:
        print("judge load: as configured in judge.yaml (4-bit)")
        judge = _load_judge(first_cfg)

    cfg_cache: dict[str, object] = {}
    done = 0
    for model_key, dkey, cond, comp in plan:
        cfg = cfg_cache.get(model_key)
        if cfg is None:
            cfg = load_cfg(_judge_cfg_path(model_key))
            if args.batch_size is not None:
                cfg.batch_size = int(args.batch_size)
            cfg_cache[model_key] = cfg
        # Smoke mode: cap rows and divert outputs to a throwaway tree so the real
        # judge/ tree is never polluted by partial smoke runs.
        if args.limit_rows:
            limited = _jsonl_rows(comp)[: args.limit_rows]
            tmp = SMOKE_DIR / "_inputs" / model_key / dkey / cond / f"completions_{cond}.jsonl"
            ensure_dir(tmp.parent)
            jsonl_write(tmp, limited)
            comp = tmp
            out_dir = SMOKE_DIR / model_key / "judge" / dkey / cond / "seed0"
        else:
            out_dir = (
                ROOT / "runs" / "direction_a_v5" / model_key
                / "judge" / dkey / cond / "seed0"
            )
        done += 1
        print(f"[{done}/{len(plan)}] {model_key} {dkey}/{cond} -> {out_dir}")
        summary = _run_reasoning_judge(
            judge, cfg, comp, out_dir, max_new_tokens=args.max_new_tokens,
        )
        print(
            f"    n={summary.get('n')} parsed={summary.get('n_parsed')} "
            f"sr_rate={summary.get('safety_reasoning_rate')} "
            f"cats={summary.get('span_category_hist')}"
        )

    print(f"done: ran safety-reasoning judge over {len(plan)} cells "
          f"across {len(args.models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
