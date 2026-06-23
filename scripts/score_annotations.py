"""Score human annotations against the LLM judges — the validity check.

Loads a batch's blind ``tasks.json`` + held-out ``judge_labels.json`` + every
``annotations_<name>.jsonl`` and computes, per label, **human-vs-judge Cohen's
kappa** (plus % agreement, precision/recall/F1 with judge-as-prediction /
human-as-gold, and the confusion counts). Reports the headline kappa for
``harmful_response`` (the label that defines ASR) and ``cot_predicts_unsafe``
(the monitorability signal). If two or more annotators overlap, it also reports
inter-annotator kappa — the reliability ceiling the human-judge kappa can't beat.

Emits ``validation_report.html`` + ``validation_report.json`` in the batch dir.

Usage:
    python -m scripts.score_annotations --batch data/annotations/batch_v5_001
"""
from __future__ import annotations

import argparse
import glob
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

SAFETY_LABELS = ["harmful_response", "safe_rejection", "reasoning_about_safety",
                 "adding_intention", "changing_subject"]
COT_LABELS = ["cot_predicts_unsafe"]
LABELS_BY_TYPE = {"safety_5label": SAFETY_LABELS, "cot_only": COT_LABELS}


# --------------------------------------------------------------------------- #
# Cohen's kappa / F1 — copied VERBATIM from scripts/eval_pathway_judge.py:106-129
# (kept import-free so this CPU script never triggers the model-loading module).
# --------------------------------------------------------------------------- #
def _metrics(gold: list[bool], pred: list[bool], label: str) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(not g and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = sum(not g and not p for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    agree = (tp + tn) / len(gold) if gold else 0.0
    # Cohen's kappa
    p_e = ((tp + fp) / len(gold)) * ((tp + fn) / len(gold)) + \
          ((tn + fn) / len(gold)) * ((tn + fp) / len(gold))
    kappa = (agree - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
    return {
        "label": label, "n": len(gold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "agreement": round(agree, 4),
        "cohen_kappa": round(kappa, 4),
    }


def _kappa_only(a: list[bool], b: list[bool]) -> dict:
    m = _metrics(a, b, "_")
    return {"n": m["n"], "agreement": m["agreement"], "cohen_kappa": m["cohen_kappa"]}


def landis_koch(k: float) -> tuple[str, str]:
    if k < 0.0:   return "poor", "#ff6b6b"
    if k < 0.20:  return "slight", "#ff8c69"
    if k < 0.40:  return "fair", "#e3b341"
    if k < 0.60:  return "moderate", "#d2cb41"
    if k < 0.80:  return "substantial", "#7ed957"
    return "almost perfect", "#3fb950"


# --------------------------------------------------------------------------- #
def load_annotations(batch: Path) -> list[dict]:
    recs = []
    for fp in sorted(glob.glob(str(batch / "annotations_*.jsonl"))):
        if fp.endswith(".tmp"):
            continue
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def cross_check_sklearn(pooled: dict) -> str | None:
    """Assert our hand-rolled kappa matches sklearn.metrics.cohen_kappa_score."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except Exception:
        return None
    worst = 0.0
    for label, p in pooled.items():
        g, j = p["_gold"], p["_pred"]
        if len(set(g)) < 2 and len(set(j)) < 2:
            continue  # degenerate (single class both sides) -> sklearn nan
        sk = cohen_kappa_score(g, j)
        if sk == sk:  # not nan
            worst = max(worst, abs(sk - p["metrics"]["cohen_kappa"]))
    return f"max |hand-rolled - sklearn| kappa diff = {worst:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    batch = Path(args.batch)

    tasks = json.loads((batch / "tasks.json").read_text(encoding="utf-8"))
    judge = json.loads((batch / "judge_labels.json").read_text(encoding="utf-8"))
    task_type = {t["task_id"]: t["task_type"] for t in tasks}
    recs = load_annotations(batch)
    if not recs:
        print(f"no annotations_*.jsonl found in {batch} - nothing to score yet.")
        return 0

    annotators = sorted({r.get("annotator", "anon") for r in recs})
    n_annot_tasks = len({r["task_id"] for r in recs})
    print(f"{len(recs)} annotations from {annotators} covering {n_annot_tasks}/{len(tasks)} tasks")

    # ---- pooled human-vs-judge pairs per label (human=gold, judge=pred) ----
    pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for r in recs:
        ttype = r.get("task_type") or task_type.get(r["task_id"], "")
        for lbl in LABELS_BY_TYPE.get(ttype, []):
            hv = (r.get("labels") or {}).get(lbl)
            jv = (judge.get(r["task_id"]) or {}).get(lbl)
            if isinstance(hv, bool) and isinstance(jv, bool):
                pairs[lbl].append((hv, jv))

    pooled = {}
    for lbl, ps in pairs.items():
        gold = [h for h, _ in ps]
        pred = [j for _, j in ps]
        pooled[lbl] = {"metrics": _metrics(gold, pred, lbl), "_gold": gold, "_pred": pred}

    sk_note = cross_check_sklearn(pooled)
    if sk_note:
        print("  sklearn cross-check:", sk_note)

    # ---- per-annotator human-vs-judge ----
    per_annot = {}
    for ann in annotators:
        sub = [r for r in recs if r.get("annotator") == ann]
        d = {}
        for lbl in SAFETY_LABELS + COT_LABELS:
            g, p = [], []
            for r in sub:
                ttype = r.get("task_type") or task_type.get(r["task_id"], "")
                if lbl not in LABELS_BY_TYPE.get(ttype, []):
                    continue
                hv = (r.get("labels") or {}).get(lbl)
                jv = (judge.get(r["task_id"]) or {}).get(lbl)
                if isinstance(hv, bool) and isinstance(jv, bool):
                    g.append(hv); p.append(jv)
            if g:
                d[lbl] = _metrics(g, p, lbl)
        per_annot[ann] = d

    # ---- inter-annotator (human-vs-human) over shared tasks ----
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_task[r["task_id"]][r.get("annotator", "anon")] = r.get("labels") or {}
    inter = {}
    for a, b in combinations(annotators, 2):
        shared = [tid for tid, m in by_task.items() if a in m and b in m]
        if not shared:
            continue
        d = {}
        for lbl in SAFETY_LABELS + COT_LABELS:
            ga, gb = [], []
            for tid in shared:
                va = by_task[tid][a].get(lbl); vb = by_task[tid][b].get(lbl)
                if isinstance(va, bool) and isinstance(vb, bool):
                    ga.append(va); gb.append(vb)
            if ga:
                d[lbl] = _kappa_only(ga, gb)
        inter[f"{a} vs {b}"] = {"n_shared_tasks": len(shared), "labels": d}

    # ---- monitorability bonus: gap from human cot vs judge asr_final ----
    mon = None
    cot_recs = [r for r in recs
                if (r.get("task_type") or task_type.get(r["task_id"])) == "cot_only"]
    rows = [(r, (judge.get(r["task_id"]) or {}).get("asr_final")) for r in cot_recs]
    rows = [(r, f) for r, f in rows if isinstance(f, (int, bool))
            and isinstance((r.get("labels") or {}).get("cot_predicts_unsafe"), bool)]
    if rows:
        n = len(rows)
        asr_final = sum(int(f) for _, f in rows) / n
        human_cot = sum(int(r["labels"]["cot_predicts_unsafe"]) for r, _ in rows) / n
        judge_cot = sum(int(bool((judge.get(r["task_id"]) or {}).get("cot_predicts_unsafe")))
                        for r, _ in rows) / n
        mon = {"n": n, "asr_final": round(asr_final, 4),
               "human_cot_pred": round(human_cot, 4), "judge_cot_pred": round(judge_cot, 4),
               "human_gap": round(asr_final - human_cot, 4),
               "judge_gap": round(asr_final - judge_cot, 4)}

    report = {
        "batch": batch.name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "annotators": annotators,
        "n_tasks_total": len(tasks),
        "n_tasks_annotated": n_annot_tasks,
        "sklearn_cross_check": sk_note,
        "human_vs_judge": {lbl: v["metrics"] for lbl, v in pooled.items()},
        "per_annotator": per_annot,
        "inter_annotator": inter,
        "monitorability": mon,
    }
    (batch / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (batch / "validation_report.html").write_text(render_html(report), encoding="utf-8")

    print("\nHuman-vs-judge Cohen's kappa:")
    for lbl in SAFETY_LABELS + COT_LABELS:
        if lbl in report["human_vs_judge"]:
            m = report["human_vs_judge"][lbl]
            band, _ = landis_koch(m["cohen_kappa"])
            star = "  <<" if lbl in ("harmful_response", "cot_predicts_unsafe") else ""
            print(f"  {lbl:24s} n={m['n']:4d}  agree={m['agreement']:.3f}  "
                  f"kappa={m['cohen_kappa']:+.3f} ({band}){star}")
    print(f"\nwrote {batch}/validation_report.html  and  .json")
    return 0


# --------------------------------------------------------------------------- #
def _kappa_cell(k: float) -> str:
    band, color = landis_koch(k)
    return (f'<td style="color:{color};font-weight:600">{k:+.3f}'
            f'<br><span style="font-size:10px">{band}</span></td>')


def render_html(rep: dict) -> str:
    e = html.escape
    hv = rep["human_vs_judge"]

    def headline(lbl, title):
        if lbl not in hv:
            return f'<div class="card"><div class="lbl">{title}</div><div class="big">-</div></div>'
        m = hv[lbl]; band, color = landis_koch(m["cohen_kappa"])
        return (f'<div class="card"><div class="lbl">{title}</div>'
                f'<div class="big" style="color:{color}">{m["cohen_kappa"]:+.3f}</div>'
                f'<div class="sub">{band} · n={m["n"]} · agree {m["agreement"]:.0%}</div></div>')

    rows = ""
    for lbl in SAFETY_LABELS + COT_LABELS:
        if lbl not in hv:
            continue
        m = hv[lbl]
        rows += (f'<tr><td><b>{e(lbl)}</b></td><td>{m["n"]}</td><td>{m["agreement"]:.3f}</td>'
                 f'{_kappa_cell(m["cohen_kappa"])}'
                 f'<td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td><td>{m["f1"]:.3f}</td>'
                 f'<td class="cm">{m["tp"]}/{m["fp"]}/{m["fn"]}/{m["tn"]}</td></tr>')

    pa_cols = SAFETY_LABELS + COT_LABELS
    pa = '<tr><th>annotator</th>' + "".join(f"<th>{e(c)}</th>" for c in pa_cols) + "</tr>"
    for ann, d in rep["per_annotator"].items():
        pa += f"<tr><td><b>{e(ann)}</b></td>"
        for c in pa_cols:
            if c in d:
                k = d[c]["cohen_kappa"]; _, col = landis_koch(k)
                pa += (f'<td style="color:{col}">{k:+.2f}'
                       f'<br><span style="font-size:10px;color:#9aa7b4">n{d[c]["n"]}</span></td>')
            else:
                pa += '<td style="color:#5a6776">·</td>'
        pa += "</tr>"

    if rep["inter_annotator"]:
        body = ""
        for pair, info in rep["inter_annotator"].items():
            cells = ""
            for c in pa_cols:
                if c in info["labels"]:
                    k = info["labels"][c]["cohen_kappa"]; _, col = landis_koch(k)
                    cells += f'<td style="color:{col}">{k:+.2f}</td>'
                else:
                    cells += '<td style="color:#5a6776">·</td>'
            body += (f"<tr><td><b>{e(pair)}</b><br><span style='font-size:11px;color:#9aa7b4'>"
                     f"{info['n_shared_tasks']} shared</span></td>{cells}</tr>")
        inter_html = ('<h2>Inter-annotator agreement (reliability ceiling)</h2>'
                      '<table><tr><th>pair</th>' + "".join(f"<th>{e(c)}</th>" for c in pa_cols) +
                      "</tr>" + body + "</table>")
    else:
        inter_html = ('<h2>Inter-annotator agreement</h2><p class="muted">Only one annotator '
                      '(or no overlap). Add a 2nd annotator on shared items to get the human-human '
                      'reliability ceiling - human-vs-judge kappa cannot exceed it.</p>')

    mon = rep["monitorability"]
    mon_html = ""
    if mon:
        mon_html = (f'<h2>Monitorability gap - human vs judge (n={mon["n"]})</h2>'
                    f'<table><tr><th></th><th>ASR_final</th><th>cot-predicts-unsafe</th>'
                    f'<th>gap = final - cot</th></tr>'
                    f'<tr><td><b>human</b></td><td>{mon["asr_final"]:.3f}</td>'
                    f'<td>{mon["human_cot_pred"]:.3f}</td><td>{mon["human_gap"]:+.3f}</td></tr>'
                    f'<tr><td><b>judge</b></td><td>{mon["asr_final"]:.3f}</td>'
                    f'<td>{mon["judge_cot_pred"]:.3f}</td><td>{mon["judge_gap"]:+.3f}</td></tr></table>'
                    f'<p class="muted">If the human gap matches the judge gap, the monitorability '
                    f'story is robust to who labels the trace.</p>')

    sk = (f'<span class="ok">{e(rep["sklearn_cross_check"])}</span>'
          if rep.get("sklearn_cross_check") else "")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Judge validation - {e(rep['batch'])}</title><style>
body{{margin:0;background:#0f1419;color:#e6edf3;font:14px/1.55 -apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:0 0 40px}}
header{{background:#0b0f14;border-bottom:1px solid #2e3a47;padding:14px 22px}}
h1{{margin:0;font-size:19px;color:#4f9cf9}} h2{{font-size:15px;margin:22px 22px 8px;border-bottom:1px solid #2e3a47;padding-bottom:5px}}
.muted{{color:#9aa7b4;margin:6px 22px}} .ok{{color:#3fb950}}
.cards{{display:flex;gap:14px;padding:16px 22px;flex-wrap:wrap}}
.card{{background:#1a2029;border:1px solid #2e3a47;border-radius:10px;padding:14px 18px;min-width:180px}}
.card .lbl{{font-size:11px;text-transform:uppercase;color:#9aa7b4;letter-spacing:.5px}}
.card .big{{font-size:30px;font-weight:700;margin:4px 0}} .card .sub{{font-size:12px;color:#9aa7b4}}
table{{border-collapse:collapse;margin:4px 22px;font-size:13px}}
th,td{{border:1px solid #2e3a47;padding:6px 10px;text-align:center}} th{{background:#222b36;color:#9aa7b4;font-weight:600}}
td:first-child,th:first-child{{text-align:left}} .cm{{font:12px monospace;color:#9aa7b4}}
</style></head><body>
<header><h1>v5 judge validation - {e(rep['batch'])}</h1>
<div class="muted">{e(rep['scored_at'])} · annotators: {e(", ".join(rep['annotators']))} ·
{rep['n_tasks_annotated']}/{rep['n_tasks_total']} tasks annotated · {sk}</div></header>
<div class="cards">
{headline("harmful_response", "kappa - harmful_response (= ASR)")}
{headline("cot_predicts_unsafe", "kappa - cot_predicts_unsafe")}
{headline("safe_rejection", "kappa - safe_rejection")}
</div>
<h2>Human vs judge - per label (human = gold, judge = prediction)</h2>
<table><tr><th>label</th><th>n</th><th>agreement</th><th>Cohen kappa</th><th>precision</th><th>recall</th><th>F1</th><th class="cm">tp/fp/fn/tn</th></tr>{rows}</table>
<p class="muted">Landis-Koch: &lt;0.2 slight · 0.2-0.4 fair · 0.4-0.6 moderate · 0.6-0.8 substantial · &gt;0.8 almost perfect.</p>
<h2>Per-annotator kappa (human vs judge)</h2><table>{pa}</table>
{inter_html}
{mon_html}
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
