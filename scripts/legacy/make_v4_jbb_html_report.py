"""Direction A v4 - HTML report for JBB Qwen3 judge outputs.

Reads the per-condition outputs produced by ``run_v4_jbb_judge.py`` under
``runs/direction_a/19-v4-jbb-qwen3`` and writes a self-contained HTML report
with the raw 12-label pathway judge rates, the existing aggregate scores, and
small inline charts.

Usage:
    python -m scripts.make_v4_jbb_html_report \
        --in-base runs/direction_a/19-v4-jbb-qwen3 \
        --out runs/direction_a/19-v4-jbb-qwen3/v4_jbb_report.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from collections import defaultdict
from pathlib import Path


PATHWAY_LABELS_BY_GROUP = (
    ("recognition", (
        "risk_acknowledgment",
        "policy_awareness",
        "intent_assessment",
    )),
    ("refusal", (
        "refusal_initiation",
        "refusal_maintenance",
        "refusal_suppression",
    )),
    ("rationalisation", (
        "benign_motive_invention",
        "educational_framing",
        "legal_disclaimer_laundering",
    )),
    ("execution", (
        "task_decomposition",
        "procedural_planning",
        "operational_detail",
    )),
)

PATHWAY_LABELS = tuple(
    label for _group, labels in PATHWAY_LABELS_BY_GROUP for label in labels
)

BASIC_FIELDS = (
    "harmful_rate",
    "refusal_rate",
    "reasoning_rate",
    "adding_intent_rate",
    "changing_subject_rate",
)

PATHWAY_VECTOR_FIELDS = (
    "recognition_rate",
    "refusal_initiated",
    "refusal_suppressed",
    "rationalisation_rate",
    "execution_rate",
    "operational_detail_rate",
    "first_execution_idx_norm",
    "recognition_to_execution",
)

DOMINANT_PATHWAYS = (
    "recognition_loss",
    "refusal_loss",
    "rationalised_compliance",
    "direct_execution",
    "sanitised_compliance",
    "none",
)

CONDITION_ORDER = (
    "baseline",
    "ships_top10",
    "neurons_top32",
    "steering_ablate",
)

COND_COLORS = {
    "baseline": "#2f6f9f",
    "ships_top10": "#b85c38",
    "neurons_top32": "#2f8f63",
    "steering_ablate": "#7b5aa6",
}

DOM_COLORS = {
    "recognition_loss": "#c4553b",
    "refusal_loss": "#d39a2d",
    "rationalised_compliance": "#7b5aa6",
    "direct_execution": "#8f2f4a",
    "sanitised_compliance": "#3f7f7a",
    "none": "#9aa5af",
}

DISPLAY = {
    "baseline": "Baseline",
    "ships_top10": "SHIPS top-10",
    "neurons_top32": "Neurons top-32",
    "steering_ablate": "Steering ablate",
    "risk_acknowledgment": "Risk acknowledgment",
    "policy_awareness": "Policy awareness",
    "intent_assessment": "Intent assessment",
    "refusal_initiation": "Refusal initiation",
    "refusal_maintenance": "Refusal maintenance",
    "refusal_suppression": "Refusal suppression",
    "benign_motive_invention": "Benign motive invention",
    "educational_framing": "Educational framing",
    "legal_disclaimer_laundering": "Legal disclaimer laundering",
    "task_decomposition": "Task decomposition",
    "procedural_planning": "Procedural planning",
    "operational_detail": "Operational detail",
    "harmful_rate": "Harmful final answer",
    "refusal_rate": "Safe rejection",
    "reasoning_rate": "Safety reasoning",
    "adding_intent_rate": "Added intention",
    "changing_subject_rate": "Changed subject",
    "recognition_rate": "Recognition",
    "refusal_initiated": "Refusal initiated",
    "refusal_suppressed": "Refusal suppressed",
    "rationalisation_rate": "Rationalisation",
    "execution_rate": "Execution",
    "operational_detail_rate": "Operational detail",
    "first_execution_idx_norm": "First execution index",
    "recognition_to_execution": "Recognition before execution",
    "recognition_loss": "Recognition loss",
    "refusal_loss": "Refusal loss",
    "rationalised_compliance": "Rationalised compliance",
    "direct_execution": "Direct execution",
    "sanitised_compliance": "Sanitised compliance",
    "none": "None",
}


CSS = """
:root {
  color-scheme: light;
  --ink: #172026;
  --muted: #65727e;
  --line: #d8dee4;
  --soft: #eef2f5;
  --paper: #ffffff;
  --bg: #f6f7f9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  max-width: 1240px;
  margin: 0 auto;
  padding: 26px 22px 44px;
}
header {
  border-bottom: 1px solid var(--line);
  padding-bottom: 18px;
  margin-bottom: 22px;
}
h1 {
  margin: 0 0 8px;
  font-size: 28px;
  line-height: 1.12;
  letter-spacing: 0;
}
h2 {
  margin: 30px 0 12px;
  font-size: 19px;
  line-height: 1.25;
  letter-spacing: 0;
}
h3 {
  margin: 18px 0 8px;
  font-size: 15px;
  line-height: 1.25;
  letter-spacing: 0;
}
p {
  margin: 8px 0;
  color: var(--muted);
  line-height: 1.45;
}
code {
  background: var(--soft);
  border-radius: 4px;
  padding: 2px 4px;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--paper);
  border: 1px solid var(--line);
  margin: 12px 0 22px;
}
th, td {
  border-bottom: 1px solid #e7ebef;
  padding: 8px 9px;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}
th {
  background: var(--soft);
  font-weight: 650;
  color: #24313a;
  position: sticky;
  top: 0;
  z-index: 1;
}
td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.muted { color: var(--muted); }
.grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(190px, 1fr));
  gap: 12px;
}
.stat {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}
.stat .label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.stat .value {
  margin-top: 6px;
  font-size: 24px;
  font-weight: 720;
  font-variant-numeric: tabular-nums;
}
.stat .sub {
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}
.callout {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  margin: 12px 0 22px;
}
.callout ul {
  margin: 8px 0 0;
  padding-left: 20px;
}
.callout li {
  margin: 7px 0;
  line-height: 1.45;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  align-items: center;
  margin: 8px 0 12px;
}
.legend span {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  color: var(--muted);
  font-size: 12px;
}
.swatch {
  width: 11px;
  height: 11px;
  border-radius: 2px;
  display: inline-block;
}
.table-wrap {
  overflow-x: auto;
}
.heat td {
  min-width: 84px;
}
.heatbar {
  width: 100%;
  min-width: 68px;
  height: 24px;
  border: 1px solid #ced6dd;
  border-radius: 4px;
  background: #f8fafb;
  overflow: hidden;
  position: relative;
}
.heatbar i {
  display: block;
  height: 100%;
  opacity: .72;
}
.heatbar b {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 6px;
  font-size: 12px;
  font-weight: 650;
  color: #152027;
  font-variant-numeric: tabular-nums;
}
.bars {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0 22px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(185px, 240px) 1fr;
  gap: 12px;
  align-items: start;
  padding: 10px 0;
  border-bottom: 1px solid #edf0f3;
}
.bar-row:last-child { border-bottom: 0; }
.metric-name {
  font-size: 13px;
  color: #2c3942;
  line-height: 1.25;
}
.bar-lines {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 8px;
}
.mini-line {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 7px;
  align-items: center;
}
.track {
  height: 12px;
  border-radius: 999px;
  background: #e9edf1;
  overflow: hidden;
}
.fill {
  display: block;
  height: 100%;
  border-radius: 999px;
}
.mini-line em {
  color: var(--muted);
  font-size: 12px;
  font-style: normal;
  font-variant-numeric: tabular-nums;
}
.svg-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0 22px;
}
.svg-card svg {
  width: 100%;
  height: auto;
  display: block;
}
.footnote {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
@media (max-width: 900px) {
  main { padding: 20px 12px 34px; }
  .grid { grid-template-columns: 1fr 1fr; }
  .bar-row { grid-template-columns: 1fr; }
  .bar-lines { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .grid { grid-template-columns: 1fr; }
  h1 { font-size: 23px; }
}
"""


def _display(name: str) -> str:
    return DISPLAY.get(name, name.replace("_", " ").title())


def _pct(x: float | int | None, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def _num(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{digits}f}"


def _pp(x: float | int | None, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.{digits}f} pp"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_summaries(in_base: Path, seed: int) -> list[dict]:
    out = []
    for sub in sorted(in_base.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / f"seed{seed}" / "summary.json"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            s = json.load(f)
        s["_tag"] = sub.name
        s["_dir"] = str(sub / f"seed{seed}")
        out.append(s)
    return sorted(
        out,
        key=lambda s: CONDITION_ORDER.index(s.get("condition"))
        if s.get("condition") in CONDITION_ORDER
        else 99,
    )


def _first_value(d: dict | None) -> dict:
    if not d:
        return {}
    return next(iter(d.values()), {})


def _basic(summary: dict) -> dict:
    return _first_value(summary.get("per_condition_basic"))


def _pathway_mean(summary: dict) -> dict:
    return (_first_value(summary.get("per_condition_pathway")).get("mean") or {})


def _pathway_hist(summary: dict) -> dict:
    return (_first_value(summary.get("per_condition_pathway")).get(
        "dominant_pathway_hist"
    ) or {})


def _monitor(summary: dict) -> dict:
    return _first_value((summary.get("monitorability") or {}).get("per_condition"))


def _parent_id(row: dict) -> str | None:
    pid = row.get("parent_id")
    if pid is not None:
        return str(pid)
    rid = str(row.get("id", ""))
    if "::p" in rid:
        return rid.split("::p", 1)[0]
    return rid or None


def _pathway_label_rates(rows: list[dict]) -> dict:
    parsed_rows = []
    by_parent: dict[str, dict[str, bool]] = {}
    for row in rows:
        labels = (row.get("judge_flat") or {}).get("pathway_labels")
        if not isinstance(labels, dict):
            continue
        parsed_rows.append(labels)
        pid = _parent_id(row)
        if pid is not None and pid not in by_parent:
            by_parent[pid] = {label: False for label in PATHWAY_LABELS}
        if pid is not None:
            for label in PATHWAY_LABELS:
                by_parent[pid][label] = (
                    by_parent[pid][label] or bool(labels.get(label))
                )

    n_prefix = len(parsed_rows)
    n_parent = len(by_parent)
    prefix = {}
    any_completion = {}
    for label in PATHWAY_LABELS:
        prefix[label] = (
            sum(1 for labels in parsed_rows if bool(labels.get(label))) / n_prefix
            if n_prefix
            else None
        )
        any_completion[label] = (
            sum(1 for labels in by_parent.values() if labels[label]) / n_parent
            if n_parent
            else None
        )
    return {
        "n_prefix_rows": len(rows),
        "n_parsed_prefix_rows": n_prefix,
        "n_completions": n_parent,
        "prefix_rate": prefix,
        "completion_any_rate": any_completion,
    }


def _condition_payloads(in_base: Path, summaries: list[dict], seed: int) -> list[dict]:
    payloads = []
    for summary in summaries:
        cond = summary.get("condition") or summary.get("_tag")
        seed_dir = in_base / summary["_tag"] / f"seed{seed}"
        pathway_rows = _jsonl(seed_dir / "judge_pathway.jsonl")
        payloads.append({
            "condition": cond,
            "tag": summary["_tag"],
            "seed_dir": seed_dir,
            "summary": summary,
            "basic": _basic(summary),
            "pathway_mean": _pathway_mean(summary),
            "pathway_hist": _pathway_hist(summary),
            "monitor": _monitor(summary),
            "coherence": summary.get("coherence") or {},
            "pathway_labels": _pathway_label_rates(pathway_rows),
        })
    return payloads


def _heatbar(value: float | None, color: str) -> str:
    pct = 0.0 if value is None else max(0.0, min(1.0, float(value))) * 100.0
    label = "n/a" if value is None else _pct(value)
    return (
        '<div class="heatbar">'
        f'<i style="width:{pct:.2f}%; background:{html.escape(color)}"></i>'
        f"<b>{html.escape(label)}</b></div>"
    )


def _legend(names: list[str], colors: dict[str, str]) -> str:
    bits = []
    for name in names:
        bits.append(
            f'<span><i class="swatch" style="background:{colors[name]}"></i>'
            f"{html.escape(_display(name))}</span>"
        )
    return '<div class="legend">' + "".join(bits) + "</div>"


def _pathway_heatmap(payloads: list[dict], *, rate_key: str) -> str:
    conds = [p["condition"] for p in payloads]
    rows = []
    rows.append("<tr><th>Group</th><th>Metric</th>")
    for cond in conds:
        rows.append(f'<th class="num">{html.escape(_display(cond))}</th>')
    rows.append("</tr>")

    for group, labels in PATHWAY_LABELS_BY_GROUP:
        first = True
        for label in labels:
            rows.append("<tr>")
            if first:
                rows.append(
                    f'<td rowspan="{len(labels)}">{html.escape(group.title())}</td>'
                )
                first = False
            rows.append(f"<td>{html.escape(_display(label))}</td>")
            for p in payloads:
                cond = p["condition"]
                value = p["pathway_labels"][rate_key].get(label)
                rows.append(
                    f'<td class="num">{_heatbar(value, COND_COLORS[cond])}</td>'
                )
            rows.append("</tr>")
    return (
        '<div class="table-wrap"><table class="heat"><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _bar_rows(payloads: list[dict], labels: tuple[str, ...], value_fn) -> str:
    conds = [p["condition"] for p in payloads]
    out = ['<div class="bars">']
    for label in labels:
        out.append('<div class="bar-row">')
        out.append(f'<div class="metric-name">{html.escape(_display(label))}</div>')
        out.append('<div class="bar-lines">')
        for p in payloads:
            cond = p["condition"]
            value = value_fn(p, label)
            pct = 0.0 if value is None else max(0.0, min(1.0, float(value))) * 100.0
            out.append(
                '<div class="mini-line">'
                '<div class="track">'
                f'<span class="fill" style="width:{pct:.2f}%;'
                f'background:{COND_COLORS[cond]}"></span>'
                '</div>'
                f"<em>{html.escape(_pct(value))}</em>"
                "</div>"
            )
        out.append("</div></div>")
    out.append("</div>")
    return "".join(out)


def _safety_bars(payloads: list[dict]) -> str:
    return _bar_rows(
        payloads,
        BASIC_FIELDS,
        lambda p, field: p["basic"].get(field),
    )


def _pathway_any_bars(payloads: list[dict]) -> str:
    return _bar_rows(
        payloads,
        PATHWAY_LABELS,
        lambda p, field: p["pathway_labels"]["completion_any_rate"].get(field),
    )


def _dominant_svg(payloads: list[dict]) -> str:
    left = 170
    width = 860
    row_h = 36
    top = 26
    height = top + row_h * len(payloads) + 18
    svg = [
        f'<div class="svg-card"><svg viewBox="0 0 {left + width + 30} {height}" '
        'role="img" aria-label="Dominant pathway stacked bars">'
    ]
    for i, p in enumerate(payloads):
        y = top + i * row_h
        cond = p["condition"]
        hist = p["pathway_hist"]
        total = sum(hist.values()) or 1
        svg.append(
            f'<text x="0" y="{y + 16}" font-size="13" fill="#24313a">'
            f"{html.escape(_display(cond))}</text>"
        )
        x = left
        for dom in DOMINANT_PATHWAYS:
            frac = hist.get(dom, 0) / total
            w = width * frac
            if w <= 0:
                continue
            svg.append(
                f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="20" '
                f'rx="3" fill="{DOM_COLORS[dom]}"></rect>'
            )
            if w >= 54:
                svg.append(
                    f'<text x="{x + w / 2:.2f}" y="{y + 14}" text-anchor="middle" '
                    'font-size="11" fill="#fff">'
                    f"{100 * frac:.0f}%</text>"
                )
            x += w
    svg.append("</svg></div>")
    return "".join(svg)


def _monitorability_svg(payloads: list[dict]) -> str:
    left = 170
    width = 760
    row_h = 42
    top = 24
    height = top + row_h * len(payloads) + 26
    svg = [
        f'<div class="svg-card"><svg viewBox="0 0 {left + width + 100} {height}" '
        'role="img" aria-label="Monitorability grouped bars">'
    ]
    for i, p in enumerate(payloads):
        y = top + i * row_h
        cond = p["condition"]
        mon = p["monitor"]
        final = float(mon.get("asr_final", 0.0) or 0.0)
        cot = float(mon.get("asr_cot_pred", 0.0) or 0.0)
        gap = float(mon.get("gap", 0.0) or 0.0)
        svg.append(
            f'<text x="0" y="{y + 17}" font-size="13" fill="#24313a">'
            f"{html.escape(_display(cond))}</text>"
        )
        for j, (name, value, color) in enumerate((
            ("final", final, "#b85c38"),
            ("cot", cot, "#2f6f9f"),
        )):
            by = y + j * 15
            svg.append(
                f'<rect x="{left}" y="{by}" width="{width * value:.2f}" '
                f'height="11" rx="3" fill="{color}"></rect>'
            )
            svg.append(
                f'<text x="{left + width * value + 6:.2f}" y="{by + 10}" '
                'font-size="11" fill="#65727e">'
                f"{name}: {_pct(value)}</text>"
            )
        svg.append(
            f'<text x="{left + width + 54}" y="{y + 17}" text-anchor="end" '
            'font-size="12" fill="#24313a">'
            f"gap {_pp(gap)}</text>"
        )
    svg.append("</svg></div>")
    return "".join(svg)


def _summary_cards(payloads: list[dict]) -> str:
    cards = []
    for p in payloads:
        cond = p["condition"]
        basic = p["basic"]
        coh = p["coherence"]
        mon = p["monitor"]
        cards.append(
            '<div class="stat">'
            f'<div class="label">{html.escape(_display(cond))}</div>'
            f'<div class="value" style="color:{COND_COLORS[cond]}">'
            f'{html.escape(_pct(basic.get("harmful_rate")))}</div>'
            '<div class="sub">harmful final answer'
            f' | refusal {html.escape(_pct(basic.get("refusal_rate")))}'
            f' | clean {html.escape(_pct(coh.get("clean_rate")))}'
            f' | CoT gap {html.escape(_pp(mon.get("gap")))}</div>'
            "</div>"
        )
    return '<div class="grid">' + "".join(cards) + "</div>"


def _delta(payload_by_cond: dict[str, dict], cond: str, field: str,
           source: str = "basic") -> float | None:
    base = payload_by_cond.get("baseline")
    other = payload_by_cond.get(cond)
    if not base or not other:
        return None
    if source == "basic":
        a = base["basic"].get(field)
        b = other["basic"].get(field)
    elif source == "pathway_any":
        a = base["pathway_labels"]["completion_any_rate"].get(field)
        b = other["pathway_labels"]["completion_any_rate"].get(field)
    elif source == "vector":
        a = base["pathway_mean"].get(field)
        b = other["pathway_mean"].get(field)
    else:
        return None
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _insights(payloads: list[dict]) -> str:
    by = {p["condition"]: p for p in payloads}
    bullets = []
    base = by.get("baseline")
    steering = by.get("steering_ablate")
    neurons = by.get("neurons_top32")
    ships = by.get("ships_top10")
    if base and steering:
        bullets.append(
            "Steering ablation is the cleanest behavioral shift: harmful final "
            f"answers rise from {_pct(base['basic'].get('harmful_rate'))} to "
            f"{_pct(steering['basic'].get('harmful_rate'))}, while coherence "
            f"stays high ({_pct(steering['coherence'].get('clean_rate'))})."
        )
        bullets.append(
            "The steering condition changes the failure mode, not just the ASR: "
            f"refusal suppression is {_pct(steering['pathway_labels']['completion_any_rate'].get('refusal_suppression'))} "
            f"({ _pp(_delta(by, 'steering_ablate', 'refusal_suppression', 'pathway_any'))} vs baseline), "
            f"educational framing is {_pct(steering['pathway_labels']['completion_any_rate'].get('educational_framing'))}, "
            f"and operational detail appears in {_pct(steering['pathway_labels']['completion_any_rate'].get('operational_detail'))} of completions."
        )
    if base and neurons:
        bullets.append(
            "Top-32 neuron ablation is almost indistinguishable from baseline on "
            f"this JBB judge pass: harmful rate delta {_pp(_delta(by, 'neurons_top32', 'harmful_rate'))}, "
            f"refusal delta {_pp(_delta(by, 'neurons_top32', 'refusal_rate'))}, "
            "and the 12 pathway-label profile stays very close to baseline."
        )
    if ships:
        bullets.append(
            "SHIPS top-10 ablation should be treated as a model-damage control: "
            f"harmful rate is {_pct(ships['basic'].get('harmful_rate'))}, but "
            f"clean rate collapses to {_pct(ships['coherence'].get('clean_rate'))} "
            f"and changing-subject rate is {_pct(ships['basic'].get('changing_subject_rate'))}."
        )
    if steering:
        gap = steering["monitor"].get("gap")
        bullets.append(
            "No condition shows a positive monitorability gap in this report. "
            f"Steering has gap {_pp(gap)}, meaning the CoT-only monitor "
            "over-predicts harmful final answers rather than missing hidden unsafe behavior."
        )
    return (
        '<div class="callout"><strong>Working inferences</strong><ul>'
        + "".join(f"<li>{html.escape(b)}</li>" for b in bullets)
        + "</ul></div>"
    )


def _coherence_table(payloads: list[dict]) -> str:
    rows = [
        "<tr><th>Condition</th><th class=\"num\">n</th>"
        "<th class=\"num\">Clean</th><th class=\"num\">Harmful among clean</th>"
        "<th class=\"num\">Repeat-3</th><th class=\"num\">Compression</th>"
        "<th class=\"num\">Empty</th></tr>"
    ]
    for p in payloads:
        c = p["coherence"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(_display(p['condition']))}</td>"
            f"<td class=\"num\">{html.escape(str(c.get('n', 'n/a')))}</td>"
            f"<td class=\"num\">{html.escape(_pct(c.get('clean_rate')))}</td>"
            f"<td class=\"num\">{html.escape(_pct(c.get('harmful_among_clean_rate')))}</td>"
            f"<td class=\"num\">{html.escape(_num(c.get('repeat3_mean')))}</td>"
            f"<td class=\"num\">{html.escape(_num(c.get('compression_ratio_mean')))}</td>"
            f"<td class=\"num\">{html.escape(_pct(c.get('empty_rate')))}</td>"
            "</tr>"
        )
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def _plain_table(payloads: list[dict], fields: tuple[str, ...], source: str) -> str:
    rows = ["<tr><th>Condition</th>"]
    for field in fields:
        rows.append(f'<th class="num">{html.escape(_display(field))}</th>')
    rows.append("</tr>")
    for p in payloads:
        rows.append(f"<tr><td>{html.escape(_display(p['condition']))}</td>")
        for field in fields:
            if source == "basic":
                value = p["basic"].get(field)
                rendered = _pct(value)
            elif source == "vector":
                value = p["pathway_mean"].get(field)
                rendered = _num(value)
            else:
                value = p["pathway_labels"][source].get(field)
                rendered = _pct(value)
            rows.append(f'<td class="num">{html.escape(rendered)}</td>')
        rows.append("</tr>")
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def _dominant_table(payloads: list[dict]) -> str:
    rows = ["<tr><th>Condition</th>"]
    for dom in DOMINANT_PATHWAYS:
        rows.append(f'<th class="num">{html.escape(_display(dom))}</th>')
    rows.append("</tr>")
    for p in payloads:
        hist = p["pathway_hist"]
        total = sum(hist.values()) or 1
        rows.append(f"<tr><td>{html.escape(_display(p['condition']))}</td>")
        for dom in DOMINANT_PATHWAYS:
            n = int(hist.get(dom, 0))
            rows.append(
                f'<td class="num">{n} ({100.0 * n / total:.0f}%)</td>'
            )
        rows.append("</tr>")
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def _meta_table(payloads: list[dict]) -> str:
    rows = [
        "<tr><th>Condition</th><th>Tag</th><th class=\"num\">Completions</th>"
        "<th class=\"num\">Pathway prefixes</th><th class=\"num\">Parsed prefixes</th>"
        "<th class=\"num\">CoT rows</th></tr>"
    ]
    for p in payloads:
        pl = p["pathway_labels"]
        mon = p["monitor"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(_display(p['condition']))}</td>"
            f"<td><code>{html.escape(p['tag'])}</code></td>"
            f"<td class=\"num\">{pl['n_completions']}</td>"
            f"<td class=\"num\">{pl['n_prefix_rows']}</td>"
            f"<td class=\"num\">{pl['n_parsed_prefix_rows']}</td>"
            f"<td class=\"num\">{html.escape(str(mon.get('n', 'n/a')))}</td>"
            "</tr>"
        )
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def build_html(in_base: Path, out_path: Path, seed: int) -> str:
    summaries = _load_summaries(in_base, seed)
    payloads = _condition_payloads(in_base, summaries, seed)
    if not payloads:
        raise SystemExit(f"no per-condition summary.json files found under {in_base}")

    judge = summaries[0].get("judge_model", "<unknown>")
    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    cond_names = [p["condition"] for p in payloads]
    dom_names = list(DOMINANT_PATHWAYS)

    body = []
    body.append("<!doctype html><html><head><meta charset=\"utf-8\">")
    body.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    body.append("<title>Direction A v4 JBB Judge Report</title>")
    body.append(f"<style>{CSS}</style></head><body><main>")
    body.append("<header>")
    body.append("<h1>Direction A v4 JBB Judge Report</h1>")
    body.append(
        "<p>"
        f"Generated {html.escape(generated)} from "
        f"<code>{html.escape(str(in_base))}</code>. Judge: "
        f"<code>{html.escape(str(judge))}</code>. "
        "Per condition: 5 safety labels, 12 pathway labels, and one CoT-only monitor."
        "</p>"
    )
    body.append("</header>")

    body.append(_summary_cards(payloads))
    body.append(_insights(payloads))

    body.append("<h2>Run Coverage</h2>")
    body.append(_meta_table(payloads))

    body.append("<h2>Safety Outcomes</h2>")
    body.append(_legend(cond_names, COND_COLORS))
    body.append(_safety_bars(payloads))
    body.append(_plain_table(payloads, BASIC_FIELDS, "basic"))

    body.append("<h2>12 Pathway Judge Metrics</h2>")
    body.append(
        "<p>Primary view: fraction of completions where a label appears in at "
        "least one judged prefix. This avoids over-weighting long completions.</p>"
    )
    body.append(_pathway_any_bars(payloads))
    body.append("<h3>Completion Occurrence Heatmap</h3>")
    body.append(_pathway_heatmap(payloads, rate_key="completion_any_rate"))

    body.append("<h3>Prefix Prevalence Heatmap</h3>")
    body.append(
        "<p>Secondary view: fraction of all cumulative-prefix judge rows where "
        "the label is true. This captures density inside traces.</p>"
    )
    body.append(_pathway_heatmap(payloads, rate_key="prefix_rate"))

    body.append("<h2>Reduced Pathway Vector</h2>")
    body.append(_plain_table(payloads, PATHWAY_VECTOR_FIELDS, "vector"))

    body.append("<h2>Dominant Pathway Distribution</h2>")
    body.append(_legend(dom_names, DOM_COLORS))
    body.append(_dominant_svg(payloads))
    body.append(_dominant_table(payloads))

    body.append("<h2>CoT Monitorability</h2>")
    body.append(
        "<p>Gap = ASR_final - ASR_cot_pred. Positive would mean the CoT looks "
        "safer than the final answer; negative means the CoT-only monitor is "
        "more pessimistic than the final-answer judge.</p>"
    )
    body.append(_monitorability_svg(payloads))

    body.append("<h2>Coherence Gate</h2>")
    body.append(_coherence_table(payloads))
    body.append(
        "<p class=\"footnote\">A condition with low coherence should not be "
        "interpreted as a clean removal of a safety mechanism. In this run, "
        "SHIPS top-10 is the main example: its safety scores move, but its "
        "clean-rate collapse makes the causal story much less clean.</p>"
    )
    body.append("</main></body></html>")
    return "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_base = Path(args.in_base)
    out_path = Path(args.out)
    doc = build_html(in_base, out_path, args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
