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
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
from pathlib import Path

import v6_common as C
from safety_cot_heads.direction_a_v6.sharding import shard_of

STAGE_KIND = {
    "answer": ("safety", "answer_text", "judge_answer_safety.jsonl"),
    "monitor": ("cot_only", "trace_text", "judge_cot_only.jsonl"),
    # multi-label taxonomy prompt; what every v6 pathway row before 2026-09-24
    # was produced with -- by the zero-shot 30B, NOT the validated 14B judge.
    "pathway": ("pathway", "trace_text", "judge_pathway.jsonl"),
    # validated protocol: fine-tuned 14B judge, one single-label prompt per
    # pathway label (the format it was trained and evaluated on), merged.
    "pathway14b": ("pathway_single", "trace_text", "judge_pathway.jsonl"),
    "safety-reasoning": ("safety_reasoning_trace", "trace_text", "judge_safety_reasoning_trace.jsonl"),
}


def v6_judge_dir(cell: C.Cell) -> Path:
    return C.V6_ROOT / "judge" / cell.model / cell.dataset / cell.condition / cell.seed


def build_inputs(cell: C.Cell, text_field: str, stage: str, prose_prefix: bool,
                 answered_only: bool = False) -> list[dict]:
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
        # answered_only: trace stages skip traces that never reached a final
        # answer (truncated / looping) -- they cannot enter any answer-paired
        # statistic, and long looping traces dominate pathway cost.
        if answered_only and stage != "answer" and r.get("answer_is_empty"):
            continue
        if stage in ("pathway", "pathway14b"):
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
    for x in rows:                     # P0.7: hash of exactly what the judge sees
        x["input_sha256"] = C.input_sha(x.get("prompt"), x.get("completion"))
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
    ap.add_argument("--coherence-device", default=None,
                    help="device for the gibberish classifier (e.g. 'cpu'); default auto")
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
        # resume: skip ids already judged on the SAME input (and, for coherence,
        # under the current gate version); changed rows are re-judged.
        if args.stage == "coherence":
            from safety_cot_heads.analysis.coherence import COHERENCE_GATE_VERSION
            todo, stale = C.resume_todo(rows, out_path, "coherence_gate_version",
                                        COHERENCE_GATE_VERSION)
        else:
            todo, stale = C.resume_todo(rows, out_path)
            if stale and not args.dry_run:
                C.prune_rows(out_path, stale, tag=C.utcnow_iso()[:10])
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
        _run_coherence(plan, device=args.coherence_device)
        return
    _run_model_stage(plan, args, kind)


def _run_coherence(plan, device=None):
    """Coherence gate on parsed answer_text (gate v6.1).

    Each row records both the full declared gate (``is_clean``: non-empty AND
    repeat3 < 0.5 AND not word-salad/noise) and the repetition-only gate that
    every cell was actually scored with before v6.1
    (``is_clean_repetition_only``), so the aggregator can report either without
    re-running this stage. Rows are merged into the existing per-cell file,
    never overwrite it.

    ``device`` forwards to the gibberish classifier (pass 'cpu' to avoid
    contending with a running GPU judge).
    """
    from safety_cot_heads.analysis.coherence import (
        coherence_diagnostics, classify_gibberish, canonical_is_clean)
    n_rows = 0
    for cell, out_path, rows in plan:
        if not rows:
            continue
        texts = [r["completion"] for r in rows]
        diag = coherence_diagnostics(texts)
        try:
            gib = classify_gibberish(texts, device=device)
        except Exception as e:
            # Fail loudly: a silently missing classifier is how v6.0 ended up
            # repetition-only. Record the outage per row instead of guessing.
            print(f"[judge-shard] ERROR gibberish classifier unavailable ({e!r}); "
                  "rows are written with gibberish_available=False")
            gib = [{} for _ in texts]
        new_rows = {}
        for r, d, g in zip(rows, diag, gib):
            rec = {"id": r["id"], **d, "input_sha256": r.get("input_sha256")}
            # classify_gibberish returns {"label", "score", "is_clean"}
            glabel = g.get("label") if isinstance(g, dict) else None
            if glabel is not None:
                rec["gibberish_label"] = glabel
                rec["gibberish_score"] = g.get("score")
            gate = canonical_is_clean(is_empty=d["is_empty"], repeat3=d["repeat3"],
                                      gibberish_label=glabel)
            gate_rep = canonical_is_clean(is_empty=d["is_empty"], repeat3=d["repeat3"],
                                          gibberish_label=glabel, use_gibberish=False)
            rec["is_clean"] = gate["is_clean"]
            rec["is_clean_repetition_only"] = gate_rep["is_clean"]
            rec["coherence_gate_version"] = gate["gate_version"]
            rec["gate_components"] = gate["components"]
            rec["gate_fail_reasons"] = gate["fail_reasons"]
            new_rows[str(r["id"])] = rec
        merged = {str(x["id"]): x for x in C.read_jsonl(out_path)} if out_path.exists() else {}
        merged.update(new_rows)
        C.write_jsonl(out_path, list(merged.values()))
        n_rows += len(new_rows)
    print(f"[judge-shard] coherence(answer_text) gate v6.1: {n_rows} rows across "
          f"{sum(1 for _, _, r in plan if r)} cells")


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
    # A scratch row counts as done only if it was judged on the SAME input text
    # (input_sha256); legacy scratch rows without a hash are re-judged.
    want = {r["id"]: r.get("input_sha256") for r in pooled}
    done = {str(x.get("id")) for x in C.read_jsonl(shard_file)
            if want.get(str(x.get("id"))) is not None
            and x.get("input_sha256") == want[str(x.get("id"))]} if shard_file.exists() else set()
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
        cid = str(jr.get("id", ""))
        # route only rows that answer THIS run's inputs; older scratch rows for
        # since-changed inputs must not be resurrected into the per-cell files.
        if cid not in want or jr.get("input_sha256") != want[cid]:
            continue
        cellkey, _, orig = cid.partition("||")
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
