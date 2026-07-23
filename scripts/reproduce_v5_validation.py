#!/usr/bin/env python3
"""Reproduce the existing human-validation results and derive human paired
monitorability — REUSING data/annotations/ (no new annotation tasks).

Steps (spec section 10):
  1. Verify expected annotation files exist.
  2. Verify both annotators completed the same task set.
  3. Re-run the OFFICIAL scorer (scripts.score_annotations) on a COPY of the
     batch under runs/direction_a_v6/validation/ (never mutating the committed
     data/annotations/ files).
  4. Compare the reproduced numbers to the committed validation_report.json and
     confirm they match (preservation check).
  5. Emit a concise validation table for the load-bearing labels.
  6. Separate unique-task count from pooled annotator comparisons.
  7. Derive human vs machine PAIRED monitorability (covert failure, over-warning,
     agreement, trace FNR) on the items where both a trace-only prediction and a
     final-answer harmfulness label exist.

Writes runs/direction_a_v6/reports/validation_summary.csv|.md and
runs/direction_a_v6/validation/.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import v6_common as C
from safety_cot_heads.direction_a_v6.paired_metrics import PairedItem, build_paired_table

LOAD_BEARING = ["harmful_response", "cot_predicts_unsafe", "reasoning_about_safety"]
DIAGNOSTIC = ["adding_intention", "changing_subject", "safe_rejection"]


def _split_task_id(tid: str):
    parts = tid.split("::")
    if len(parts) == 5:
        ttype, model, ds, cond, pid = parts
        return ttype, (model, ds, cond, pid)
    return parts[0] if parts else tid, tuple(parts[1:])


def reproduce_scorer(batch: Path, work: Path) -> dict:
    dst = work / batch.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(batch, dst)
    # run the official scorer as a module against the copy
    r = subprocess.run(
        [sys.executable, "-m", "scripts.score_annotations", "--batch", str(dst)],
        cwd=str(C.REPO), capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(C.REPO / "src")},
    )
    out = {"returncode": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-1000:]}
    rep_path = dst / "validation_report.json"
    out["reproduced"] = json.loads(rep_path.read_text()) if rep_path.exists() else None
    return out


def compare_reports(committed: dict, reproduced: dict) -> dict:
    """Compare load-bearing kappa/F1 numbers; ignore volatile fields (scored_at)."""
    diffs = []
    hj_c = (committed or {}).get("human_vs_judge", {})
    hj_r = (reproduced or {}).get("human_vs_judge", {})
    for lbl in set(hj_c) | set(hj_r):
        mc, mr = hj_c.get(lbl, {}), hj_r.get(lbl, {})
        for k in ("cohen_kappa", "f1", "agreement", "n"):
            a, b = mc.get(k), mr.get(k)
            if a is not None and b is not None and abs(float(a) - float(b)) > 1e-6:
                diffs.append({"label": lbl, "field": k, "committed": a, "reproduced": b})
    return {"n_labels": len(set(hj_c) | set(hj_r)), "n_diffs": len(diffs),
            "reproduces_exactly": not diffs, "diffs": diffs}


def human_paired_monitorability(batch: Path) -> dict:
    """Derive paired monitorability on the existing cot_only trace tasks.

    Each cot_only task's judge label stores ``asr_final`` (the reference
    final-answer harmfulness). We pair, per trace task:
      * y  = stored asr_final (final answer harmful, 0/1);
      * human t  = annotator cot_predicts_unsafe (pooled by majority);
      * machine t = judge cot_predicts_unsafe.
    Human and machine paired tables share the same y (asr_final), so their U/O/
    agreement/FNR differences isolate the trace predictor (human vs judge).
    """
    judge = json.loads((batch / "judge_labels.json").read_text())

    # per-cot_only-task human predictions (pool annotators by majority)
    human_t: dict[str, list[bool]] = {}
    for f in sorted(batch.glob("annotations_*.jsonl")):
        for r in C.iter_jsonl(f):
            tt, _ = _split_task_id(r["task_id"])
            if tt != "cot_only":
                continue
            v = (r.get("labels") or {}).get("cot_predicts_unsafe")
            if isinstance(v, bool):
                human_t.setdefault(r["task_id"], []).append(v)

    human_items, machine_items = [], []
    joinable = 0
    for tid, lab in judge.items():
        tt, _ = _split_task_id(tid)
        if tt != "cot_only":
            continue
        y = lab.get("asr_final")
        y = int(y) if isinstance(y, (int, bool)) and y in (0, 1, True, False) else None
        mt = lab.get("cot_predicts_unsafe")
        mt = int(mt) if isinstance(mt, bool) else None
        hts = human_t.get(tid)
        ht = int(sum(hts) >= (len(hts) / 2.0)) if hts else None
        if ht is None:
            continue
        joinable += 1
        human_items.append(PairedItem(prompt_id=tid, y=y, t=ht))
        machine_items.append(PairedItem(prompt_id=tid, y=y, t=mt))

    ht = build_paired_table(human_items)
    mt = build_paired_table(machine_items)
    return {
        "n_joinable_items": joinable,
        "n_scored": ht.n_pairs,
        "human": {"U_covert": ht.U, "O_overwarn": ht.O, "agreement": ht.A,
                  "trace_fnr": ht.trace_fnr, "n_harmful": ht.n_harmful},
        "machine": {"U_covert": mt.U, "O_overwarn": mt.O, "agreement": mt.A,
                    "trace_fnr": mt.trace_fnr, "n_harmful": mt.n_harmful},
        "human_minus_machine": {
            "U_covert": _sub(ht.U, mt.U), "O_overwarn": _sub(ht.O, mt.O),
            "agreement": _sub(ht.A, mt.A), "trace_fnr": _sub(ht.trace_fnr, mt.trace_fnr)},
        "note": "Balanced diagnostic batch: reliability measurement, NOT a "
                "natural-distribution accuracy estimate.",
    }


def _sub(a, b):
    return (a - b) if (a is not None and b is not None) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="data/annotations/batch_v5_002")
    args = ap.parse_args()
    batch = (C.REPO / args.batch).resolve()

    # 1. expected files
    required = ["tasks.json", "judge_labels.json", "validation_report.json"]
    missing = [f for f in required if not (batch / f).exists()]
    ann_files = sorted(batch.glob("annotations_*.jsonl"))
    result: dict = {"batch": str(batch.relative_to(C.REPO)),
                    "missing_required_files": missing,
                    "annotation_files": [f.name for f in ann_files]}
    if missing or not ann_files:
        C.write_json(C.V6_ROOT / "reports" / "validation_summary.json", result)
        print(f"[validation] MISSING files: {missing}; annotators={len(ann_files)}")
        return

    # 2. same task set per annotator
    ann_tasks = {}
    for f in ann_files:
        name = f.stem.replace("annotations_", "")
        ann_tasks[name] = {r["task_id"] for r in C.iter_jsonl(f)}
    all_tasks = set().union(*ann_tasks.values()) if ann_tasks else set()
    shared_tasks = set.intersection(*ann_tasks.values()) if len(ann_tasks) > 1 else set(all_tasks)
    result["annotators"] = list(ann_tasks)
    result["n_unique_tasks"] = len(all_tasks)
    result["n_shared_tasks_all_annotators"] = len(shared_tasks)
    result["same_task_set"] = all(v == all_tasks for v in ann_tasks.values())
    # pooled comparison count (annotator-task pairs) vs unique tasks
    result["n_pooled_annotator_task_pairs"] = sum(len(v) for v in ann_tasks.values())

    # 3-4. reproduce scorer + compare
    work = C.V6_ROOT / "validation"
    work.mkdir(parents=True, exist_ok=True)
    committed = json.loads((batch / "validation_report.json").read_text())
    rep = reproduce_scorer(batch, work)
    result["scorer_returncode"] = rep["returncode"]
    result["reproduction"] = compare_reports(committed, rep.get("reproduced"))

    # 5-6. concise table for load-bearing labels
    hj = (rep.get("reproduced") or committed).get("human_vs_judge", {})
    table = []
    for lbl in LOAD_BEARING + DIAGNOSTIC:
        m = hj.get(lbl)
        if not m:
            continue
        table.append({"label": lbl,
                      "tier": "load_bearing" if lbl in LOAD_BEARING else "diagnostic",
                      "n": m.get("n"), "cohen_kappa": m.get("cohen_kappa"),
                      "f1": m.get("f1"), "agreement": m.get("agreement")})
    result["load_bearing_table"] = table

    # 7. human paired monitorability
    result["human_paired_monitorability"] = human_paired_monitorability(batch)

    C.write_json(C.V6_ROOT / "reports" / "validation_summary.json", result)
    C.write_json(work / "reproduced_validation_report.json", rep.get("reproduced"))

    # CSV
    out = C.V6_ROOT / "reports"
    with open(out / "validation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "tier", "n", "cohen_kappa", "f1", "agreement"])
        for r in table:
            w.writerow([r["label"], r["tier"], r["n"], r["cohen_kappa"], r["f1"], r["agreement"]])

    _write_md(result, out)
    print(f"[validation] annotators={result['annotators']} unique_tasks={result['n_unique_tasks']} "
          f"pooled_pairs={result['n_pooled_annotator_task_pairs']}")
    print(f"[validation] scorer reproduces committed numbers: "
          f"{result['reproduction']['reproduces_exactly']} (diffs={result['reproduction']['n_diffs']})")
    hp = result["human_paired_monitorability"]
    print(f"[validation] human paired monitorability: U={hp['human']['U_covert']} "
          f"machine U={hp['machine']['U_covert']} (n={hp['n_scored']})")
    print(f"[validation] wrote {out}/validation_summary.csv|md")


def _write_md(result, out):
    rp = result["reproduction"]
    hp = result["human_paired_monitorability"]
    lines = [
        "# Direction A v6 — human-validation reuse summary", "",
        f"- Batch: `{result['batch']}` (reused; **no new annotation tasks created**)",
        f"- Annotators: {result['annotators']}",
        f"- Unique tasks: **{result['n_unique_tasks']}** "
        f"(pooled annotator-task comparison pairs: {result['n_pooled_annotator_task_pairs']} — "
        "these are NOT independent experiments)",
        f"- Both annotators cover the same task set: **{result['same_task_set']}**",
        f"- Official scorer reproduces the committed validation_report.json: "
        f"**{rp['reproduces_exactly']}** (numeric diffs: {rp['n_diffs']})",
        "",
        "> This is a **balanced diagnostic reliability** batch, not a natural-distribution "
        "accuracy estimate.",
        "",
        "## Load-bearing labels (human vs judge)", "",
        "| label | tier | n | Cohen κ | F1 | agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in result["load_bearing_table"]:
        lines.append(f"| {r['label']} | {r['tier']} | {r['n']} | {r['cohen_kappa']} "
                     f"| {r['f1']} | {r['agreement']} |")
    lines += ["", "## Human vs machine paired monitorability", "",
              f"Joinable items (trace prediction + final-answer label both present): "
              f"**{hp['n_scored']}**", "",
              "| side | U (covert) | O (over-warn) | agreement | trace FNR |",
              "|---|---:|---:|---:|---:|",
              f"| human | {hp['human']['U_covert']} | {hp['human']['O_overwarn']} "
              f"| {hp['human']['agreement']} | {hp['human']['trace_fnr']} |",
              f"| machine | {hp['machine']['U_covert']} | {hp['machine']['O_overwarn']} "
              f"| {hp['machine']['agreement']} | {hp['machine']['trace_fnr']} |",
              "",
              "Trace FNR is blank when the joined item set contains no harmful answers "
              "(undefined, never zero)."]
    (out / "validation_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
