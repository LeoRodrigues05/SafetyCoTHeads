#!/usr/bin/env python3
"""Sharded v6 re-judging on ONE GPU (or CPU dry-run).

Feeds each judge the CORRECT parsed text:
  * ``answer``          -> completion = answer_text   (kind="safety")
  * ``monitor``         -> completion = trace_text     (kind="cot_only")
  * ``pathway``         -> completion = trace_text     (kind="pathway")
  * ``safety-reasoning``-> completion = trace_text     (kind="safety_reasoning_trace")
  * ``coherence``       -> answer_text diagnostics (CPU; no model)

The stage's cells are the deterministic shard for this GPU (blake2b(cell_key)
% n_shards == gpu). Explicit trace stages skip prose-only models' explicit
trace (their prose-prefix monitorability is a separate sensitivity pass, run
with --prose-prefix). Resume: rows whose id already exists in the output are
skipped.

``--dry-run`` builds and writes the sharded judge-INPUT JSONL (verifying the
right text is fed and the shard is correct) WITHOUT loading any model — this is
what the CPU test and smoke `parse` path exercise. Remove --dry-run on the B200
to actually run inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import v6_common as C
from safety_cot_heads.direction_a_v6.sharding import shard_of

STAGE_KIND = {
    "answer": ("safety", "answer_text", "judge_answer_safety.jsonl"),
    "monitor": ("cot_only", "trace_text", "judge_cot_only.jsonl"),
    "pathway": ("pathway", "trace_text", "judge_pathway.jsonl"),
    "safety-reasoning": ("safety_reasoning_trace", "trace_text", "judge_safety_reasoning_trace.jsonl"),
}


def v6_judge_dir(cell: C.Cell) -> Path:
    return C.V6_ROOT / "judge" / cell.model / cell.dataset / cell.condition / cell.seed


def build_inputs(cell: C.Cell, text_field: str, stage: str, prose_prefix: bool) -> list[dict]:
    """Build judge-input rows for a cell, feeding each stage its required structure.

    * ``answer``/``coherence`` -> the parsed final ``answer_text``.
    * ``monitor`` (cot_only)   -> the whole ``trace_text`` (one row/completion).
    * ``pathway``              -> cumulative-prefix rows over the trace (P0.4).
    * ``safety-reasoning``     -> one indexed-sentence row per completion (P0.4).
    """
    from safety_cot_heads.direction_a_v6.trace_inputs import (
        build_pathway_prefix_rows, build_indexed_sr_row)
    parsed = C.read_jsonl(cell.v6_parsed_dir() / "parsed_completions.jsonl")
    rows: list[dict] = []
    for r in parsed:
        if stage == "pathway":
            for x in build_pathway_prefix_rows(r):
                x["seed"] = cell.seed
                x["trace_kind"] = r.get("trace_kind")
                x["is_prefix"] = bool(prose_prefix)
                rows.append(x)
            continue
        if stage == "safety-reasoning":
            x = build_indexed_sr_row(r)
            if x is not None:
                x["seed"] = cell.seed
                x["trace_kind"] = r.get("trace_kind")
                x["is_prefix"] = bool(prose_prefix)
                rows.append(x)
            continue
        if stage == "monitor":
            # whole-trace unsafe prediction; prose-prefix pass uses the prose prefix.
            if r.get("has_explicit_trace"):
                text = r.get("trace_text") or ""
            elif prose_prefix and r.get("trace_kind") == "prose_prefix":
                text = r.get("prose_prefix_text") or ""
            else:
                continue
            if not text.strip():
                continue
        else:  # answer / coherence
            text = r.get("answer_text") or ""
        rows.append({
            "id": r["id"], "parent_id": r["id"], "prompt": r.get("prompt") or "",
            "completion": text, "model": r.get("model"), "dataset": r.get("dataset"),
            "condition": r.get("condition"), "seed": cell.seed, "category": r.get("category"),
            "trace_kind": r.get("trace_kind"),
            "is_prefix": bool(prose_prefix and not r.get("has_explicit_trace")),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["answer", "monitor", "pathway", "safety-reasoning", "coherence"])
    ap.add_argument("--gpu", type=int, default=0, help="shard index this process handles")
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                    help="hf (default) = load the full bf16 judge into GPU memory and run "
                         "Transformers generation (fits a 183GB B200 easily); vllm = "
                         "continuous batching (faster steady-state, slow engine init here).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prose-prefix", action="store_true",
                    help="run explicit-trace stages on prose-prefix text (sensitivity)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build sharded judge inputs only; do not load a model")
    args = ap.parse_args()

    scope = C.load_paper_scope()
    explicit = set(scope["explicit_trace_models"])
    cells = C.discover_cells(args.models, args.datasets)
    # deterministic shard assignment for THIS gpu
    my_cells = [c for c in cells if shard_of(c.key, args.n_shards) == args.gpu]

    # explicit-trace stages: the normal pass runs on explicit-trace models; the
    # --prose-prefix sensitivity pass runs on prose-only models only.
    if args.stage in ("monitor", "pathway", "safety-reasoning"):
        prose_models = set(scope.get("prose_prefix_models", []))
        keep = prose_models if args.prose_prefix else explicit
        my_cells = [c for c in my_cells if c.model in keep]

    kind = None if args.stage == "coherence" else STAGE_KIND[args.stage][0]
    text_field = "answer_text" if args.stage in ("answer", "coherence") \
        else STAGE_KIND[args.stage][1]
    out_name = ("coherence_answer.jsonl" if args.stage == "coherence"
                else STAGE_KIND[args.stage][2])
    if args.prose_prefix and args.stage != "coherence":
        out_name = out_name.replace(".jsonl", "__prefix.jsonl")

    print(f"[judge-shard] stage={args.stage} gpu={args.gpu}/{args.n_shards} "
          f"cells={len(my_cells)} kind={kind} out={out_name} dry_run={args.dry_run}")

    # build all inputs first (CPU); this is what --dry-run verifies
    total_rows = 0
    plan = []
    for cell in my_cells:
        if args.stage == "coherence":
            rows = build_inputs(cell, text_field, "answer", args.prose_prefix)
        else:
            rows = build_inputs(cell, text_field, args.stage, args.prose_prefix)
        out_path = v6_judge_dir(cell) / out_name
        # resume: drop ids already judged
        done = {str(r.get("id")) for r in C.read_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in rows if str(r["id"]) not in done]
        plan.append((cell, out_path, todo))
        total_rows += len(todo)
        if args.dry_run:
            # materialize the judge INPUT so the shard/text can be inspected & tested
            insp = out_path.parent / (out_name.replace(".jsonl", ".input.jsonl"))
            C.write_jsonl(insp, rows)
    print(f"[judge-shard] {total_rows} rows to judge across {len(plan)} cells")

    if args.dry_run:
        print("[judge-shard] dry-run: wrote *.input.jsonl per cell; no model loaded.")
        return

    if args.stage == "coherence":
        _run_coherence(plan)
        return
    _run_model_stage(plan, args, kind)


def _run_coherence(plan, device=None):
    """Coherence gate on parsed answer_text (P0.3: uses the gibberish classifier).

    ``device`` forwards to the gibberish classifier (pass 'cpu' to avoid
    contending with a running GPU judge).
    """
    from safety_cot_heads.analysis.coherence import (
        coherence_diagnostics, classify_gibberish, canonical_is_clean)
    for cell, out_path, rows in plan:
        if not rows:
            continue
        texts = [r["completion"] for r in rows]
        diag = coherence_diagnostics(texts)
        try:
            gib = classify_gibberish(texts, device=device) if device else classify_gibberish(texts)
        except Exception as e:
            print(f"[judge-shard] WARN gibberish classifier unavailable ({e}); "
                  "gate falls back to non-empty+repetition and records it")
            gib = [{} for _ in texts]
        out = []
        for r, d, g in zip(rows, diag, gib):
            rec = {"id": r["id"], **d}
            glabel = g.get("gibberish_label") if isinstance(g, dict) else None
            if isinstance(g, dict):
                rec.update({k: g[k] for k in ("gibberish_label", "gibberish_score") if k in g})
            gate = canonical_is_clean(is_empty=d["is_empty"], repeat3=d["repeat3"],
                                      gibberish_label=glabel)
            rec["is_clean"] = gate["is_clean"]
            rec["coherence_gate_version"] = gate["gate_version"]
            rec["gate_components"] = gate["components"]
            rec["gate_fail_reasons"] = gate["fail_reasons"]
            out.append(rec)
        C.write_jsonl(out_path, out)
    print(f"[judge-shard] coherence(answer_text) done for {len(plan)} cells")


def _load_judge(args):
    """Load the JUDGE model correctly.

    IMPORTANT: the judge is a plain causal LM, NOT an intervention target. We
    must load it with ``attach_controllers=False`` — the default loader attaches
    the neuron/steering controllers, which enumerate ``.mlp.down_proj`` and
    raise "unsupported architecture" on the Qwen3-30B **MoE** judge (its MLP is a
    sparse expert stack). The vLLM backend saturates the GPU (continuous
    batching); the HF backend works but is bursty/slow.
    """
    if args.backend == "vllm":
        from safety_cot_heads.judging.vllm_backend import load_vllm_judge
        return load_vllm_judge(
            args.judge_model, dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            tensor_parallel_size=1,          # one GPU per process; both run in parallel
            trust_remote_code=True, seed=0)
    # HF fallback — device_map="auto" under CUDA_VISIBLE_DEVICES maps to the one
    # visible GPU; attach_controllers=False is the load that does NOT crash on MoE.
    from safety_cot_heads.models.loading import load_model
    return load_model(args.judge_model, dtype="bfloat16", device_map="auto",
                      trust_remote_code=True, attach_controllers=False)


def _run_model_stage(plan, args, kind):
    """Judge every row for this shard in ONE continuous pass to keep the GPU fed.

    Pooling all cells' rows (instead of one judge_rows call per cell) removes the
    per-cell idle gaps that capped utilisation, and lets judge_rows length-bucket
    globally so each batch groups similar-length prompts. Rows are tagged with a
    composite ``<cell.key>||<id>`` so outputs route back to per-cell files. The
    pooled pass appends to a shard-level scratch JSONL as it goes (crash-safe,
    resumable by composite id); the per-cell files are written at the end.
    """
    from collections import defaultdict
    from safety_cot_heads.judging import JudgeConfig, judge_rows

    route = {}
    pooled = []
    for cell, out_path, rows in plan:
        route[cell.key] = out_path
        for r in rows:
            rr = dict(r)
            rr["id"] = f"{cell.key}||{r['id']}"     # globally unique across cells
            pooled.append(rr)

    tag = f"{args.stage}_gpu{args.gpu}of{args.n_shards}" + ("_prefix" if args.prose_prefix else "")
    shard_file = C.V6_ROOT / "judge" / "_shard_scratch" / f"{tag}.jsonl"
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    done = {str(x.get("id")) for x in C.read_jsonl(shard_file)} if shard_file.exists() else set()
    todo = [r for r in pooled if r["id"] not in done]

    if not todo:
        print(f"[judge-shard] nothing to judge (all {len(pooled)} rows already done)")
    else:
        judge = _load_judge(args)
        print(f"[judge-shard] judge loaded via {args.backend}; pooling {len(todo)} rows "
              f"across {len(route)} cells into one continuous pass "
              f"(batch_size={args.batch_size}, max_new_tokens={args.max_new_tokens})")
        cfg = JudgeConfig(kind=kind, batch_size=args.batch_size,
                          max_new_tokens=args.max_new_tokens, base_temperature=0.0, seed=0)
        judge_rows(judge, todo, cfg, out_path=str(shard_file))   # incremental, crash-safe

    # route the shard scratch file back to per-cell outputs (restore original ids)
    buckets = defaultdict(list)
    for jr in C.read_jsonl(shard_file):
        cellkey, _, orig = str(jr.get("id", "")).partition("||")
        jr["id"] = orig
        buckets[cellkey].append(jr)
    for cellkey, out_rows in buckets.items():
        out_path = route.get(cellkey)
        if out_path is None:
            continue
        merged = {str(x["id"]): x for x in (C.read_jsonl(out_path) if out_path.exists() else [])}
        for x in out_rows:
            merged[str(x["id"])] = x
        C.write_jsonl(out_path, list(merged.values()))
    print(f"[judge-shard] stage={args.stage} routed to {len(buckets)} cells")


if __name__ == "__main__":
    main()
