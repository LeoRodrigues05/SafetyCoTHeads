#!/usr/bin/env python3
"""Render an HTML report of every judge summary.json already computed."""
import json, html, datetime
from pathlib import Path

ROOT = Path("runs/direction_a_v5")
COND_ORDER = ["baseline","neurons_top256","neurons_top512","neurons_top1024",
              "ships_top3","ships_top5","ships_top8",
              "steering_a0.5","steering_a1.0","steering_a1.5"]
COLS = [("n","n"),("harmful_rate","ASR (harmful)"),("refusal_rate","refusal"),
        ("reasoning_rate","reasoning"),("adding_intent_rate","add_intent"),
        ("changing_subject_rate","chg_subject"),("coh_clean","coh clean"),
        ("asr_cot_pred","cot ASR"),("gap","mon gap")]

def cell(summary, cond):
    b = summary.get("per_condition_basic", {}).get(cond, {})
    m = summary.get("monitorability", {}).get("per_condition", {}).get(cond, {})
    c = summary.get("coherence", {})
    return {"n":b.get("n"),"harmful_rate":b.get("harmful_rate"),
            "refusal_rate":b.get("refusal_rate"),"reasoning_rate":b.get("reasoning_rate"),
            "adding_intent_rate":b.get("adding_intent_rate"),
            "changing_subject_rate":b.get("changing_subject_rate"),
            "coh_clean":c.get("clean_rate"),"asr_cot_pred":m.get("asr_cot_pred"),
            "gap":m.get("gap")}

# data[model][tree][ds][cond] = cell dict
data = {}
for sp in sorted(ROOT.glob("*/*/*/*/seed0/summary.json")):
    rel = sp.relative_to(ROOT).parts          # model, tree, ds, cond, seed0, summary.json
    if len(rel) < 6: continue
    model, tree, ds, cond = rel[0], rel[1], rel[2], rel[3]
    try: s = json.loads(sp.read_text())
    except Exception: continue
    data.setdefault(model,{}).setdefault(tree,{}).setdefault(ds,{})[cond] = cell(s, cond)

def fmt(v):
    if v is None: return '<td class="na">–</td>'
    if isinstance(v,float): return f'<td>{v:.3f}</td>'
    return f'<td>{html.escape(str(v))}</td>'

parts = ["""<!doctype html><meta charset=utf-8><title>v5 judge metrics</title>
<style>body{font:14px/1.5 system-ui,Arial;margin:24px;color:#222}
h1{font-size:20px}h2{margin-top:28px}h3{margin:14px 0 4px;color:#555}
table{border-collapse:collapse;margin:4px 0 18px;font-size:13px}
th,td{border:1px solid #ddd;padding:3px 8px;text-align:right}
th{background:#f4f4f4}td:first-child,th:first-child{text-align:left}
.na{color:#bbb}caption{text-align:left;font-weight:600;padding:4px 0}
tr:nth-child(even) td{background:#fafafa}</style>"""]
parts.append(f"<h1>Direction A v5 — computed judge metrics</h1>"
             f"<p>Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. "
             f"Source: <code>runs/direction_a_v5/*/&lt;tree&gt;/&lt;ds&gt;/&lt;cond&gt;/seed0/summary.json</code> "
             f"(judge: Qwen3-30B-A3B). <b>4-bit unless re-judged.</b></p>")

n_models = n_tables = 0
for model in sorted(data):
    n_models += 1
    parts.append(f"<h2>{html.escape(model)}</h2>")
    for tree in sorted(data[model]):
        for ds in sorted(data[model][tree]):
            n_tables += 1
            parts.append(f'<table><caption>{html.escape(tree)} &middot; {html.escape(ds)}</caption>')
            parts.append("<tr><th>condition</th>" + "".join(f"<th>{html.escape(lbl)}</th>" for _,lbl in COLS) + "</tr>")
            conds = [c for c in COND_ORDER if c in data[model][tree][ds]] + \
                    [c for c in sorted(data[model][tree][ds]) if c not in COND_ORDER]
            for cond in conds:
                row = data[model][tree][ds][cond]
                parts.append(f"<tr><td>{html.escape(cond)}</td>" +
                             "".join(fmt(row.get(k)) for k,_ in COLS) + "</tr>")
            parts.append("</table>")

out = ROOT / "metrics_report.html"
out.write_text("\n".join(parts))
print(f"wrote {out}  ({n_models} models, {n_tables} tables)")
