"""Direction A v5 — build a BLIND human-annotation batch to validate the judges.

We need to confirm the LLM judges are *valid* (a human applying the same rubric
agrees with them). This samples completions across all v5 judge cells and emits a
**blind** task set (no judge labels visible) plus a separately-held judge-label
file, so a human annotates the SAME items the judges did and `score_annotations.py`
can compute human-vs-judge Cohen's kappa.

Two task types, sampled over DISJOINT items (so the full-response safety task can
never contaminate the trace-only cot task for the same completion):

  * safety_5label : prompt + full completion ; judge labels = judge_flat.labels
                    (harmful_response / safe_rejection / reasoning_about_safety /
                     adding_intention / changing_subject)
  * cot_only      : prompt + CoT trace ONLY  ; judge label  = cot_predicts_unsafe

The cot-only trace text is reconstructed with the exact pipeline function
(`safety_cot_heads.direction_a.monitorability.build_cot_only_inputs`) so the human
sees precisely what the cot-only judge saw.

Sampling is stratified across (model x dataset x condition) and prefers unsafe
items (so the harmful class is not vanishingly rare) — adapted from the
prefer-unsafe sampler in scripts/make_side_by_side_report.py.

Outputs (committable) under data/annotations/<batch_id>/:
    tasks.json         BLIND items (NO judge labels)
    judge_labels.json  held-out judge labels keyed by task_id (scoring only)
    manifest.json      provenance: seed, source root, per-stratum counts, judge models

Usage:
    python -m scripts.make_annotation_batch --n-safety 80 --n-cot 40 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from safety_cot_heads.direction_a.monitorability import build_cot_only_inputs

SAFETY_LABELS = [
    "harmful_response",
    "safe_rejection",
    "reasoning_about_safety",
    "adding_intention",
    "changing_subject",
]


# --------------------------------------------------------------------------- #
def read_jsonl(p: str | Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def discover_cells(run_root: str) -> list[tuple[str, str, str, str]]:
    """Return (cell_dir, model, ds, cond) for every v5 judge cell."""
    cells = []
    for d in sorted(glob.glob(os.path.join(run_root, "*/judge/*/*/seed0"))):
        if "_smoke_vm" in d:
            continue
        model, _judge, ds, cond, _seed = Path(d).parts[-5:]
        cells.append((d, model, ds, cond))
    return cells


def build_pools(cells, run_root):
    """Build candidate task pools (with held-out labels) for both task types."""
    safety_pool: list[dict] = []
    cot_pool: list[dict] = []
    judge_models = Counter()
    cot_judge_models = Counter()

    for cell_dir, model, ds, cond in cells:
        # --- completions (source of prompt + full response) ---
        gen_glob = os.path.join(run_root, model, "gen", ds, cond, "seed0", "completions_*.jsonl")
        gen_files = glob.glob(gen_glob)
        comps = read_jsonl(gen_files[0]) if gen_files else []
        comp_by_id = {str(c.get("id")): c for c in comps}

        # --- safety pool (one task per completion with parsed judge labels) ---
        judged_files = glob.glob(os.path.join(cell_dir, "judged_*.jsonl"))
        for r in read_jsonl(judged_files[0]) if judged_files else []:
            labs = (r.get("judge_flat") or {}).get("labels")
            if not labs:
                continue
            cid = str(r.get("id"))
            c = comp_by_id.get(cid)
            if not c or not (c.get("completion") or "").strip():
                continue
            if r.get("judge_model"):
                judge_models[r["judge_model"]] += 1
            safety_pool.append({
                "task_type": "safety_5label",
                "model": model, "dataset": ds, "condition": cond, "id": cid,
                "prompt": c.get("prompt") or "",
                "response": c.get("completion") or "",
                "_labels": {k: bool(labs.get(k)) for k in SAFETY_LABELS},
                "_judge_model": r.get("judge_model"),
                "_unsafe": bool(labs.get("harmful_response")),
            })

        # --- cot pool (trace-only; reconstructed exactly as the judge saw it) ---
        cot_lab: dict[str, bool] = {}
        for r in read_jsonl(os.path.join(cell_dir, "judge_cot_only.jsonl")):
            pid = str(r.get("parent_id") or r.get("id"))
            pred = (r.get("judge_flat") or {}).get("cot_predicts_unsafe")
            if isinstance(pred, bool):
                cot_lab[pid] = pred
                if r.get("judge_model"):
                    cot_judge_models[r["judge_model"]] += 1
        asr_final = {str(r.get("parent_id")): r.get("asr_final")
                     for r in read_jsonl(os.path.join(cell_dir, "monitorability_rows.jsonl"))}
        for ci in build_cot_only_inputs(comps):
            cid = str(ci.get("id"))
            if cid not in cot_lab:
                continue  # no judge label to compare against
            cot_pool.append({
                "task_type": "cot_only",
                "model": model, "dataset": ds, "condition": cond, "id": cid,
                "prompt": ci.get("prompt") or "",
                "cot_text": ci.get("response") or "",
                "_label": {"cot_predicts_unsafe": bool(cot_lab[cid]),
                           "asr_final": asr_final.get(cid)},
                "_judge_model": (cot_judge_models.most_common(1)[0][0]
                                 if cot_judge_models else None),
                "_unsafe": bool(cot_lab[cid]),
            })

    return safety_pool, cot_pool, judge_models, cot_judge_models


def stratified_sample(pool, n, seed, used_keys, unsafe_frac=0.5):
    """Sample ~n items spread across (model,ds,cond) strata with a target unsafe
    fraction (default 50/50) so BOTH classes are well represented — a single-class
    set would make Cohen's kappa degenerate. Within each class we round-robin over
    strata for even model/condition coverage; we then interleave the two class
    queues toward ``unsafe_frac``. Skips keys already taken (cross-type disjoint)."""
    rng = random.Random(seed)
    strata: dict[tuple, dict[str, list]] = defaultdict(lambda: {"unsafe": [], "safe": []})
    for t in pool:
        key = (t["model"], t["dataset"], t["condition"], t["id"])
        if key in used_keys:
            continue
        strata[(t["model"], t["dataset"], t["condition"])][
            "unsafe" if t["_unsafe"] else "safe"].append(t)
    order = sorted(strata)
    rng.shuffle(order)
    for s in order:
        rng.shuffle(strata[s]["unsafe"])
        rng.shuffle(strata[s]["safe"])

    def queue(cls: str) -> list:
        q, idx, more = [], {s: 0 for s in order}, True
        while more:
            more = False
            for s in order:
                lst = strata[s][cls]
                if idx[s] < len(lst):
                    q.append(lst[idx[s]]); idx[s] += 1; more = True
        return q

    unsafe_q, safe_q = queue("unsafe"), queue("safe")
    target_unsafe = round(n * unsafe_frac)
    selected, ui, si, n_unsafe = [], 0, 0, 0
    while len(selected) < n and (ui < len(unsafe_q) or si < len(safe_q)):
        want_unsafe = n_unsafe < target_unsafe
        if want_unsafe and ui < len(unsafe_q):
            t = unsafe_q[ui]; ui += 1; n_unsafe += 1
        elif si < len(safe_q):
            t = safe_q[si]; si += 1
        elif ui < len(unsafe_q):
            t = unsafe_q[ui]; ui += 1; n_unsafe += 1
        else:
            break
        key = (t["model"], t["dataset"], t["condition"], t["id"])
        if key in used_keys:
            continue
        used_keys.add(key)
        selected.append(t)
    return selected


def task_id_of(t: dict) -> str:
    return f"{t['task_type']}::{t['model']}::{t['dataset']}::{t['condition']}::{t['id']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", default="runs/direction_a_v5")
    ap.add_argument("--out-dir", default="data/annotations")
    ap.add_argument("--batch-id", default="batch_v5_001")
    ap.add_argument("--n-safety", type=int, default=80)
    ap.add_argument("--n-cot", type=int, default=40)
    ap.add_argument("--unsafe-frac", type=float, default=0.5,
                    help="target fraction of judged-unsafe items (balanced=0.5 for stable kappa)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = discover_cells(args.run_root)
    print(f"discovered {len(cells)} judge cells under {args.run_root}")
    safety_pool, cot_pool, jm, cjm = build_pools(cells, args.run_root)
    print(f"candidate pools: {len(safety_pool)} safety, {len(cot_pool)} cot")

    used: set[tuple] = set()
    sel_safety = stratified_sample(safety_pool, args.n_safety, args.seed, used, args.unsafe_frac)
    sel_cot = stratified_sample(cot_pool, args.n_cot, args.seed + 1, used, args.unsafe_frac)

    # Deterministic interleave so the annotator doesn't do all-safety then all-cot.
    combined = sel_safety + sel_cot
    random.Random(args.seed + 7).shuffle(combined)

    tasks, judge_labels = [], {}
    for t in combined:
        tid = task_id_of(t)
        blind = {
            "task_id": tid, "task_type": t["task_type"], "model": t["model"],
            "dataset": t["dataset"], "condition": t["condition"], "id": t["id"],
            "prompt": t["prompt"],
        }
        if t["task_type"] == "safety_5label":
            blind["response"] = t["response"]
            judge_labels[tid] = {**t["_labels"], "judge_model": t.get("_judge_model")}
        else:
            blind["cot_text"] = t["cot_text"]
            judge_labels[tid] = {**t["_label"], "judge_model": t.get("_judge_model")}
        tasks.append(blind)

    # Blinding guard: no held-out judge field may leak into the blind task set.
    LEAK = {"harmful_response", "safe_rejection", "reasoning_about_safety",
            "adding_intention", "changing_subject", "cot_predicts_unsafe",
            "asr_final", "judge_model"}
    for t in tasks:
        assert not (set(t) & LEAK), f"blinding leak in {t['task_id']}: {set(t) & LEAK}"

    out = Path(args.out_dir) / args.batch_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (out / "judge_labels.json").write_text(json.dumps(judge_labels, indent=2), encoding="utf-8")

    strata_counts = Counter((t["task_type"], t["model"], t["dataset"]) for t in tasks)
    manifest = {
        "batch_id": args.batch_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "run_root": args.run_root,
        "seed": args.seed,
        "n_requested": {"safety": args.n_safety, "cot": args.n_cot},
        "n_realized": {"safety": len(sel_safety), "cot": len(sel_cot), "total": len(tasks)},
        "judge_models": {"safety": dict(jm), "cot_only": dict(cjm)},
        "rubric_source": "src/safety_cot_heads/judging/judge_prompts.py "
                         "(SAFETY_BEHAVIOR_PROMPT / SAFETY_LABEL_DEFINITIONS / COT_ONLY_PREDICTION_PROMPT)",
        "by_type_model_dataset": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(strata_counts.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_unsafe_s = sum(1 for t in sel_safety if t["_unsafe"])
    n_unsafe_c = sum(1 for t in sel_cot if t["_unsafe"])
    print(f"\nwrote {len(tasks)} blind tasks -> {out}/tasks.json")
    print(f"  safety: {len(sel_safety)} ({n_unsafe_s} judged-harmful)")
    print(f"  cot   : {len(sel_cot)} ({n_unsafe_c} judged cot-unsafe)")
    print(f"  models covered: {sorted({t['model'] for t in tasks})}")
    print(f"  judge_labels.json covers all task_ids: {set(judge_labels) == {t['task_id'] for t in tasks}}")
    print(f"\nNext: python -m scripts.annotate_server --batch {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
