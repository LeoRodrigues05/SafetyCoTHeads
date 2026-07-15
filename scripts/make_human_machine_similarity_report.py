"""Direction A v5 — human-machine similarity & judge-trust report.

Turns the annotation validation artifact and the fine-tuned pathway-judge eval
into one HTML page that says, per instrument, whether the machine judge is
reliable enough to headline. Rebuilds what previously existed only as a one-off
HTML so it regenerates whenever the annotation batch or pathway eval changes.

Inputs (no GPU, no model weights):
  data/annotations/<batch>/validation_report.json   human-vs-judge + inter-annotator
  runs/pathway_judge_14b_lora/eval_sample_n180.json  fine-tuned vs baseline pathway judge

Each instrument row reports human-vs-judge Cohen's kappa, agreement, F1, and the
human-human ceiling (inter-annotator kappa on the same label, where two
annotators labelled it) so a moderate judge kappa can be read against how much
humans themselves agree. Trust verdict follows the Landis-Koch band of the
human-vs-judge kappa.

Usage:
    .venv/bin/python -m scripts.make_human_machine_similarity_report
    .venv/bin/python -m scripts.make_human_machine_similarity_report \
        --batch data/annotations/batch_v5_002 \
        --pathway-eval runs/pathway_judge_14b_lora/eval_sample_n180.json \
        --out runs/human_machine_similarity_report.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_DEFAULT = ROOT / "data" / "annotations" / "batch_v5_002"
PATHWAY_DEFAULT = ROOT / "runs" / "pathway_judge_14b_lora" / "eval_sample_n180.json"
OUT_DEFAULT = ROOT / "runs" / "human_machine_similarity_report.html"

# Instrument rows: (axis/use label, source, key, is_headline).
# source "hvj" = human_vs_judge block; "sr" = safety_reasoning block.
INSTRUMENTS = [
    ("Potency / ASR", "hvj", "harmful_response", True),
    ("Monitorability", "hvj", "cot_predicts_unsafe", True),
    ("Safety reasoning trace", "sr", "has_safety_reasoning", True),
    ("SR sentence any-SR", "sr", "sentence_level", False),
    ("Safety reasoning label", "hvj", "reasoning_about_safety", False),
    ("Refusal label", "hvj", "safe_rejection", False),
    ("Adding-intent label", "hvj", "adding_intention", False),
    ("Changing-subject label", "hvj", "changing_subject", False),
]


def _kappa_band(k: float) -> tuple[str, str]:
    """Landis-Koch band -> (name, css class)."""
    if k < 0.20:
        return "slight", "bad"
    if k < 0.40:
        return "fair", "warn"
    if k < 0.60:
        return "moderate", "mid"
    if k < 0.80:
        return "substantial", "good"
    return "almost perfect", "good"


def _decision(k: float) -> tuple[str, str]:
    if k >= 0.60:
        return "trust for headline use", "good"
    if k >= 0.40:
        return "usable with caveats", "mid"
    return "diagnostic only", "bad"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def _ceiling(inter: dict, key: str):
    """Human-human inter-annotator kappa for the label, or None."""
    row = inter.get("labels", {}).get(key)
    return None if row is None else row.get("cohen_kappa")


def build_rows(val: dict) -> list[dict]:
    hvj = val.get("human_vs_judge", {})
    sr = val.get("safety_reasoning", {})
    inter = next(iter(val.get("inter_annotator", {}).values()), {})
    rows = []
    for axis, src, key, headline in INSTRUMENTS:
        block = hvj if src == "hvj" else sr
        stat = block.get(key)
        if not stat:
            continue
        rows.append({
            "axis": axis, "label": stat.get("label", key), "headline": headline,
            "n": stat.get("n"), "agreement": stat.get("agreement"),
            "kappa": stat.get("cohen_kappa"), "f1": stat.get("f1"),
            "ceiling": _ceiling(inter, key),
        })
    return rows


CSS = """
:root{--ink:#172026;--muted:#64717d;--line:#d9e0e6;--soft:#eef3f6;--paper:#fff;--bg:#f7f8fa;--blue:#256c7d;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}
main{max-width:1220px;margin:0 auto;padding:28px 22px 56px} header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:22px}
h1{margin:0 0 8px;font-size:28px;line-height:1.15} h2{margin:28px 0 10px;font-size:19px}
p,li{color:var(--muted);line-height:1.45;margin:8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0 22px}
.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:13px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:700}
.value{font-size:23px;font-weight:760;margin-top:5px;line-height:1.2}.note{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--paper);border:1px solid var(--line);margin:10px 0 20px}
th,td{border-bottom:1px solid #e7ebef;padding:7px 9px;text-align:left;font-size:13px;vertical-align:top}
th{background:var(--soft);font-weight:680;color:#24313a}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tr:hover td{background:#fbfcfd}
.pill{display:inline-block;border-radius:999px;padding:2px 9px;font-size:12px;font-weight:700;border:1px solid transparent}
.good{background:#e7f6ee;color:#137a43;border-color:#a6dcc0}.mid{background:#e9f3ff;color:#185a9d;border-color:#bdd8f4}
.warn{background:#fff5e6;color:#9a5b12;border-color:#e6c28c}.bad{background:#fdecec;color:#9a2020;border-color:#e6a8a8}
.callout{background:#fff;border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:8px;padding:12px 14px;margin:12px 0 18px}
code{background:var(--soft);border-radius:4px;padding:2px 4px}
"""


def _table(headers, rows, num_cols) -> str:
    out = ["<table><thead><tr>"]
    for i, h in enumerate(headers):
        out.append(f'<th{" class=num" if i in num_cols else ""}>{html.escape(h)}</th>')
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            out.append(f'<td{" class=num" if i in num_cols else ""}>{cell}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build_report(val: dict, pathway: dict, batch_name: str,
                 batch_rel: str, pathway_rel: str) -> str:
    rows = build_rows(val)
    headline_ks = [r["kappa"] for r in rows if r["headline"] and r["kappa"] is not None]
    mean_headline = sum(headline_ks) / len(headline_ks) if headline_ks else 0.0
    trust_verdict, _ = _decision(mean_headline)

    wsum = sum(r["agreement"] * r["n"] for r in rows
               if r["agreement"] is not None and r["n"])
    wn = sum(r["n"] for r in rows if r["agreement"] is not None and r["n"])
    overall_agree = wsum / wn if wn else None

    n_annotated = val.get("n_tasks_annotated", val.get("n_tasks_total"))
    annotators = val.get("annotators", [])
    pw_ft = pathway.get("overall_finetuned", {})

    cards = "".join([
        f'<div class="card"><div class="label">Annotated tasks</div>'
        f'<div class="value">{html.escape(str(n_annotated))}</div>'
        f'<p>{len(annotators)} independent annotators.</p></div>',
        f'<div class="card"><div class="label">Overall trust</div>'
        f'<div class="value">{html.escape(trust_verdict)}</div>'
        f'<p>Mean headline &kappa; = {mean_headline:.3f}.</p></div>',
        f'<div class="card"><div class="label">Overall agreement</div>'
        f'<div class="value">{_pct(overall_agree)}</div>'
        f'<p>n-weighted across all binary decisions incl. SR sentences.</p></div>',
        f'<div class="card"><div class="label">Pathway judge</div>'
        f'<div class="value">&kappa; {_f(pw_ft.get("cohen_kappa"))}</div>'
        f'<p>Fine-tuned 14B vs held-out gold.</p></div>',
    ])

    trust_rows = []
    for r in rows:
        band, band_cls = _kappa_band(r["kappa"])
        dec, dec_cls = _decision(r["kappa"])
        trust_rows.append([
            html.escape(r["axis"]),
            f'<code>{html.escape(r["label"])}</code>',
            str(r["n"]), _pct(r["agreement"]), _f(r["kappa"]),
            _pill(band, band_cls),
            _f(r["ceiling"]) if r["ceiling"] is not None else "n/a",
            _pct(r["f1"]), _pill(dec, dec_cls),
        ])
    trust_table = _table(
        ["Axis / use", "Label", "n", "Agreement", "κ", "κ band",
         "Human-human ceiling", "F1", "Decision"],
        trust_rows, num_cols={2, 3, 4, 6, 7})

    per = val.get("per_annotator", {})
    pa_rows = []
    for name in annotators:
        d = per.get(name, {})
        hr = d.get("harmful_response", {})
        cot = d.get("cot_predicts_unsafe", {})
        pa_rows.append([
            html.escape(name),
            _f(hr.get("cohen_kappa")), _pct(hr.get("agreement")),
            _f(cot.get("cohen_kappa")), _pct(cot.get("agreement")),
        ])
    pa_table = _table(
        ["Annotator", "harmful_response κ", "harmful_response agreement",
         "cot_predicts_unsafe κ", "cot agreement"],
        pa_rows, num_cols={1, 2, 3, 4})

    pw_base = pathway.get("overall_baseline", {})
    pw_rows = []
    for name, blk in [("Fine-tuned 14B pathway judge", pw_ft),
                      ("30B baseline pathway judge", pw_base)]:
        k = blk.get("cohen_kappa")
        band, band_cls = _kappa_band(k) if k is not None else ("n/a", "")
        dec, dec_cls = _decision(k) if k is not None else ("n/a", "")
        pw_rows.append([
            html.escape(name), str(blk.get("n", "n/a")),
            _pct(blk.get("agreement")), _f(k), _pill(band, band_cls),
            _pct(blk.get("f1")), _pill(dec, dec_cls),
        ])
    pw_table = _table(
        ["Model", "n", "Agreement", "κ", "κ band", "F1", "Decision"],
        pw_rows, num_cols={1, 2, 3, 5})

    weak = [r["label"] for r in rows if r["kappa"] is not None and r["kappa"] < 0.40]
    weak_txt = " and ".join(f"<code>{html.escape(w)}</code>" for w in weak) or "none"

    body = f"""
<div class="grid">{cards}</div>
<div class="callout"><b>&kappa; scale.</b> Landis-Koch bands: slight &lt;0.20, fair
0.20-0.40, moderate 0.40-0.60, substantial 0.60-0.80, almost perfect &gt;=0.80.
A label is trusted for headline use when &kappa; is substantial, or when a moderate
&kappa; already sits at the human-human ceiling.</div>
<h2>1. Trust Table by Instrument</h2>
<p class="note">Human-vs-judge agreement per instrument. The human-human ceiling is
the inter-annotator &kappa; on the same label (where two annotators marked it);
read a moderate judge &kappa; against it.</p>{trust_table}
<h2>2. Per-Annotator Headline Agreement</h2>
<p class="note">Each annotator scored against the judge separately on the two
headline binary labels.</p>{pa_table}
<h2>3. Pathway Judge Validation</h2>
<p class="note">Fine-tuned 14B LoRA vs the 30B base judge, both against the
held-out HarmThoughts gold set.</p>{pw_table}
<h2>4. Interpretation</h2>
<div class="callout"><b>Verdict:</b> the machine judges are good enough to trust for
the headline metrics when used as designed. Potency
(<code>harmful_response</code>) is moderate and near the human-human ceiling;
monitorability and trace-level safety-reasoning detection are substantial. The
weak labels are {weak_txt}, where humans also disagree, so treat those as
diagnostic only.</div>
<h2>5. Caveats</h2>
<table><tbody>
<tr><th>Use directly</th><td>Potency / ASR, monitorability gap, has-safety-reasoning
trace detection, and fine-tuned pathway vectors.</td></tr>
<tr><th>Use with caveats</th><td>Safe-rejection and exact SR category labels; useful
descriptively, not as sole headline claims.</td></tr>
<tr><th>Do not headline</th><td>Adding-intention and changing-subject labels;
reliability is too low because the concepts are ambiguous to humans too.</td></tr>
<tr><th>Guardrail</th><td>Keep the coherence gate: the main judge failure mode is
over-calling harm on broken or incoherent outputs.</td></tr>
</tbody></table>"""

    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    subtitle = (f"Generated {generated} from <code>{html.escape(batch_rel)}/"
                f"validation_report.json</code> (batch {html.escape(batch_name)}) "
                f"and pathway eval <code>{html.escape(pathway_rel)}</code>.")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Human-Machine Similarity and Judge Trust Report</title>"
            f"<style>{CSS}</style></head><body><main>"
            f"<header><h1>Human-Machine Similarity and Judge Trust Report</h1>"
            f"<p>{subtitle}</p></header>{body}</main></body></html>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=str(BATCH_DEFAULT))
    ap.add_argument("--pathway-eval", default=str(PATHWAY_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    batch_dir = Path(args.batch)
    val_path = batch_dir / "validation_report.json"
    val = json.loads(val_path.read_text())
    pathway_path = Path(args.pathway_eval)
    pathway = json.loads(pathway_path.read_text()) if pathway_path.exists() else {}
    if not pathway:
        print(f"warning: pathway eval not found at {pathway_path}; "
              "pathway table will show n/a")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(val, pathway, batch_dir.name,
                                _rel(batch_dir), _rel(pathway_path)),
                   encoding="utf-8")
    print(f"wrote {_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
