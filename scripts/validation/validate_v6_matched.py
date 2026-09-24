#!/usr/bin/env python3
"""Human validation of the v6 judges on input-identical items (no new annotation).

The committed batch (data/annotations/batch_v5_002) was built from v5 judge
inputs. Its reported kappas therefore validate the v5 configuration. The paper's
numbers come from v6 judges, which see different text for some items (answer
only instead of the full completion, trace-only sentence lists, re-derived prose
prefixes, continued traces).

This script keeps an item for a v6 judge only when the text the annotators
saw is IDENTICAL (after stripping outer whitespace) to the text that v6 judge
receives, and compares the human labels with the v6 judge label for that exact
prompt/cell. Items whose input differs are counted and reported, never pooled.

Decisions validated:
  answer   harmful_response        human `response`  vs v6 answer_text
  monitor  cot_predicts_unsafe     human `cot_text`  vs v6 trace_text (explicit)
                                                       or prose_prefix_text (prose)
  sr       has_safety_reasoning    human segments    vs v6 indexed trace sentences
           + sentence-level spans  (only when the sentence lists are identical)
  sr_restricted  (trace, sentence)  the v5 batch showed trace ('cot') + answer
           ('output') sentences; v6 judges trace sentences only. When the cot
           sentences are identical to the v6 sentence list, human labels on
           exactly those sentences are compared (has_SR = any span among them).
           The v6 judge input is then a strict subset of what humans saw.

Excluded by construction: explicit-trace items whose current parse is the
v6.0 truncation artefact (a cut-off trace parsed as the answer) -- these vanish
after the v6.1 re-parse; and items whose shown text is from an older generation
of the cell (it no longer equals the current parse).

Kappa is pooled over (annotator, task) pairs, as in the paper; the 95% CI
bootstraps TASKS (both annotators move together). Annotators are anonymised
(A1, A2). Run it after the v6.1 rerun for final numbers: continued traces change
some explicit-trace items.

Usage:
  .venv/bin/python scripts/validation/validate_v6_matched.py --out runs/direction_a_v6/validation
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import ast
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import v6_common as C  # noqa: E402
from safety_cot_heads.analysis.trajectory import split_sentences  # noqa: E402

BATCH = C.REPO / "data" / "annotations" / "batch_v5_002"


def _lab(x):
    return ast.literal_eval(x) if isinstance(x, str) else (x or {})


def kappa(pairs):
    """Cohen's kappa for a list of (a, b) booleans; None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def boot_ci(task_pairs: dict, n_boot=2000, seed=12345):
    keys = sorted(task_pairs)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [p for k in (rng.choice(keys) for _ in keys) for p in task_pairs[k]]
        k = kappa(sample)
        if k is not None:
            vals.append(k)
    if not vals:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


class Cells:
    """Lazy per-cell caches of parsed rows and v6 judge outputs."""

    def __init__(self):
        self._p, self._j = {}, {}

    def parsed(self, m, ds, cond):
        k = (m, ds, cond)
        if k not in self._p:
            p = C.V6_ROOT / "parsed" / m / ds / cond / "seed0" / "parsed_completions.jsonl"
            self._p[k] = {str(r["id"]): r for r in C.read_jsonl(p)}
        return self._p[k]

    def judged(self, m, ds, cond, name):
        k = (m, ds, cond, name)
        if k not in self._j:
            p = C.V6_ROOT / "judge" / m / ds / cond / "seed0" / name
            self._j[k] = {str(r.get("parent_id") or r["id"]) if name != "judge_safety_reasoning_trace.jsonl"
                          else str(r["id"]): r for r in C.read_jsonl(p)
                          if "::" not in str(r.get("id"))}
        return self._j[k]


def _sr_idx(items) -> set[int]:
    """v6 SR judge sentence indexes: ints, or dicts carrying global_index/index."""
    out = set()
    for i in items or []:
        if isinstance(i, dict):
            i = i.get("global_index", i.get("index"))
        if i is not None:
            out.add(int(i))
    return out


def _sr_restricted(t, pr, hum, cells, res, hh, counts, tt, split_sentences):
    """Compare human SR labels on the trace sentences shared with the v6 input."""
    segs = t.get("segments") or []
    cot = [s for s in segs if s.get("section") == "cot"]
    v6_segs = split_sentences(pr.get("trace_text") or "") if pr.get("has_explicit_trace") else []
    if not cot or [s["text"] for s in cot] != v6_segs:
        return
    jr = cells.judged(t["model"], t["dataset"], t["condition"],
                      "judge_safety_reasoning_trace.jsonl").get(str(t["id"]))
    flat = (jr or {}).get("judge_flat") or {}
    if not isinstance(flat.get("has_safety_reasoning"), bool):
        return
    counts[tt]["restricted_matched"] += 1
    g2pos = {int(s["global_index"]): k for k, s in enumerate(cot)}
    v6_idx = _sr_idx(flat.get("safety_reasoning_sentence_indexes"))
    v6_has = bool(flat["has_safety_reasoning"])
    tid = t["task_id"]
    h_trace = {}
    for a, lab in hum.items():
        pos = {g2pos[int(i)] for i in (lab.get("spans") or {}) if int(i) in g2pos}
        h_trace[a] = bool(pos)
        res["sr_trace_restricted"][tid].append((bool(pos), v6_has))
        res["sr_sentence_restricted"][tid].extend((k in pos, k in v6_idx) for k in range(len(cot)))
    if len(h_trace) == 2:
        a, b = h_trace.values()
        hh["sr_trace_restricted"].append((a, b))
        (l1, l2) = list(hum.values())
        p1 = {g2pos[int(i)] for i in (l1.get("spans") or {}) if int(i) in g2pos}
        p2 = {g2pos[int(i)] for i in (l2.get("spans") or {}) if int(i) in g2pos}
        hh["sr_sentence_restricted"].extend((k in p1, k in p2) for k in range(len(cot)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(BATCH))
    ap.add_argument("--out", default=None, help="directory for validation_v6_matched.{json,md}")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    batch = Path(args.batch)

    tasks = {t["task_id"]: t for t in json.load(open(batch / "tasks.json"))}
    ann_files = sorted(batch.glob("annotations_*.jsonl"))
    human = defaultdict(dict)                    # task_id -> {A1: labels, A2: labels}
    for i, f in enumerate(ann_files, 1):
        for r in C.read_jsonl(f):
            human[r["task_id"]][f"A{i}"] = _lab(r["labels"])
    scope = C.load_paper_scope()
    prose = set(scope["prose_prefix_models"])
    explicit = set(scope["explicit_trace_models"])
    cells = Cells()
    v5_judge = {k: _lab(v) for k, v in json.load(open(batch / "judge_labels.json")).items()}
    concord = []            # (v5 judge on full completion, v6 judge on answer) where input differs

    res = {"answer": defaultdict(list), "monitor": defaultdict(list), "sr": defaultdict(list),
           "sr_sentence": defaultdict(list), "sr_trace_restricted": defaultdict(list),
           "sr_sentence_restricted": defaultdict(list)}
    hh = {"answer": [], "monitor": [], "sr": [], "sr_trace_restricted": [],
          "sr_sentence_restricted": []}
    counts = defaultdict(lambda: defaultdict(int))
    for tid, t in tasks.items():
        m, ds, cond, pid, tt = t["model"], t["dataset"], t["condition"], str(t["id"]), t["task_type"]
        pr = cells.parsed(m, ds, cond).get(pid)
        hum = human.get(tid, {})
        if pr is None or not hum:
            counts[tt]["no_v6_row_or_no_annotation"] += 1
            continue
        if tt == "safety_5label":
            key, shown, v6_in = "answer", t.get("response") or "", pr.get("answer_text") or ""
            jr = cells.judged(m, ds, cond, "judge_answer_safety.jsonl").get(pid)
            v6 = (((jr or {}).get("judge_flat") or {}).get("labels") or {}).get("harmful_response")
            hfield = "harmful_response"
        elif tt == "cot_only":
            key, shown = "monitor", t.get("cot_text") or ""
            is_prose = m in prose
            v6_in = (pr.get("prose_prefix_text") if is_prose else pr.get("trace_text")) or ""
            name = "judge_cot_only__prefix.jsonl" if is_prose else "judge_cot_only.jsonl"
            jr = cells.judged(m, ds, cond, name).get(pid)
            v6 = ((jr or {}).get("judge_flat") or {}).get("cot_predicts_unsafe")
            hfield = "cot_predicts_unsafe"
        elif tt == "safety_reasoning":
            key = "sr"
            segs = [s["text"] for s in t.get("segments") or []]
            v6_segs = split_sentences(pr.get("trace_text") or "") if pr.get("has_explicit_trace") else []
            shown, v6_in = "\x1f".join(segs), "\x1f".join(v6_segs)
            jr = cells.judged(m, ds, cond, "judge_safety_reasoning_trace.jsonl").get(pid)
            flat = (jr or {}).get("judge_flat") or {}
            v6 = flat.get("has_safety_reasoning") if jr else None
            hfield = "has_safety_reasoning"
        else:
            continue
        counts[tt]["annotated"] += 1
        if m in explicit and pr.get("parse_status") in ("prose_prefix", "prose_only"):
            counts[tt]["v60_truncation_artefact"] += 1
            continue
        if key == "sr" and shown.strip() != v6_in.strip():
            _sr_restricted(t, pr, hum, cells, res, hh, counts, tt, split_sentences)
        if shown.strip() != v6_in.strip():
            counts[tt]["input_differs_from_v6"] += 1
            # indirect evidence only: does the human-validated v5 configuration
            # (full completion) agree with v6 (answer only) on these items?
            v5v = v5_judge.get(tid, {}).get("harmful_response")
            if key == "answer" and isinstance(v5v, bool) and isinstance(v6, bool):
                concord.append((tid, v5v, v6))
            continue
        if not isinstance(v6, bool):
            counts[tt]["no_v6_label"] += 1
            continue
        counts[tt]["matched"] += 1
        hv = {a: bool(l.get(hfield)) for a, l in hum.items() if hfield in l}
        res[key][tid] = [(h, v6) for h in hv.values()]
        if len(hv) == 2:
            a, b = list(hv.values())
            hh[key].append((a, b))
        if key == "sr" and jr:                   # sentence level: lists are identical here
            v6_idx = _sr_idx(flat.get("safety_reasoning_sentence_indexes"))
            for a, l in hum.items():
                h_idx = set(int(i) for i in (l.get("spans") or {}))
                res["sr_sentence"][tid].extend((i in h_idx, i in v6_idx) for i in range(len(segs)))

    summary = {}
    for key in ("answer", "monitor", "sr", "sr_sentence", "sr_trace_restricted",
                "sr_sentence_restricted"):
        tp = res[key]
        pairs = [p for v in tp.values() for p in v]
        summary[key] = {
            "n_tasks_matched": len(tp), "n_pairs": len(pairs),
            "kappa_judge_human": kappa(pairs),
            "kappa_ci95": boot_ci(tp, args.n_boot) if tp else None,
            "agreement": (sum(a == b for a, b in pairs) / len(pairs)) if pairs else None,
            "kappa_human_human": kappa(hh.get(key, [])) if key in hh else None,
        }
    by_model = defaultdict(lambda: defaultdict(int))
    for key in ("answer", "monitor", "sr"):
        for tid in res[key]:
            by_model[key][tasks[tid]["model"]] += 1
    conc = {"n": len(concord),
            "agreement": (sum(a == b for _, a, b in concord) / len(concord)) if concord else None,
            "kappa": kappa([(a, b) for _, a, b in concord]),
            "by_model": {m: sum(1 for t_, _, _ in concord if tasks[t_]["model"] == m)
                         for m in sorted({tasks[t_]["model"] for t_, _, _ in concord})},
            "note": "v5 judge (full completion; human-validated) vs v6 judge (answer only) "
                    "on answer items whose input differs -- configuration concordance, "
                    "NOT a human validation of v6"}
    out = {"generated_at_utc": C.utcnow_iso(), "batch": str(batch.relative_to(C.REPO)),
           "answer_config_concordance": conc,
           "v6_root": str(C.V6_ROOT), "summary": summary,
           "counts": {k: dict(v) for k, v in counts.items()},
           "matched_tasks_by_model": {k: dict(v) for k, v in by_model.items()}}

    lines = ["# Human validation of the v6 judges on input-identical items", "",
             "| decision | tasks matched | pairs | kappa judge-human [95% CI] | agreement | kappa human-human |",
             "|---|---:|---:|---|---:|---:|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"
    for key, s in summary.items():
        ci = s["kappa_ci95"]
        lines.append(f"| {key} | {s['n_tasks_matched']} | {s['n_pairs']} | {f(s['kappa_judge_human'])} "
                     f"[{f(ci[0]) if ci else '-'}, {f(ci[1]) if ci else '-'}] | {f(s['agreement'])} | "
                     f"{f(s['kappa_human_human'])} |")
    lines += ["", "Item accounting (items not matched are excluded, never pooled):", ""]
    for tt, c in out["counts"].items():
        lines.append(f"- {tt}: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    lines += ["", "Matched tasks by model: " + json.dumps(out["matched_tasks_by_model"]),
              "", f"Answer-judge configuration concordance on unmatched items (v5 full-completion "
              f"vs v6 answer-only; not human validation): n={conc['n']}, "
              f"agreement={f(conc['agreement'])}, kappa={f(conc['kappa'])}, by model {conc['by_model']}"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        C.write_json(od / "validation_v6_matched.json", out)
        (od / "validation_v6_matched.md").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
