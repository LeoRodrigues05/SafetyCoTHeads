"""Direction A v5 — metrics & judging-status overview report.

Scans every judge source on disk (full n=100 tree + the n=25/n=40 subset
trees) for every model, walks the canonical condition grid so that not-yet-
judged cells show up explicitly as ``missing``, and emits a single HTML page
with several tables:

  1. Judges / instruments in use.
  2. Coverage & status  (complete / partial / missing per pass).
  3. Core metrics       (potency + quality: harmful, refusal, clean, ASR-clean).
  4. Monitorability     (ASR final vs CoT-pred + gap).
  5. CoT-trace safety   (indexed safety-reasoning: parsed, rate, position, extent).
  6. Pathway vector     (recognition / refusal-suppression / execution + dominant).

Every numeric cell is a per-condition average already stored in summary.json
(or safety_reasoning.summary.json); cells with no data are rendered ``n/a``.

Usage:
    python -m scripts.make_v5_metrics_status_report \
        --out runs/direction_a_v5/metrics_status_report.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "direction_a_v5"

MODELS = ("qwen3_8b", "llama31_8b_control")
DATASETS = ("jbb", "bt")
CONDITIONS = (
    "baseline",
    "ships_top3", "ships_top5", "ships_top8",
    "neurons_top256", "neurons_top512", "neurons_top1024",
    "steering_a0.5", "steering_a1.0", "steering_a1.5",
    "steering_ablate",
)
# judge-root dir name -> short source label
SOURCES = (
    ("judge", "full"),
    ("judge_subset_n25_fast", "n25"),
    ("judge_subset_n40", "n40"),
)

DISPLAY = {
    "qwen3_8b": "Qwen3 8B",
    "llama31_8b_control": "Llama 3.1 8B",
    "jbb": "JBB",
    "bt": "BT",
}


def _disp(x: str) -> str:
    return DISPLAY.get(x, x.replace("_", " "))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        return rows
    return rows


def _harmful_label(row: dict) -> bool:
    """Read the harmful flag from a merged safety judgment.

    The flag is nested at ``judge_flat.labels.harmful_response`` — reading it
    off the top level (as the old summary aggregator did) silently yields 0.
    """
    return bool((row.get("judge_flat") or {}).get("labels", {})
                .get("harmful_response"))


def _derive_harmful_among_clean(seed_dir: Path, cond: str) -> float | None:
    """Recompute harmful_among_clean from raw files, immune to stale summaries.

    clean ids come from ``coherence.jsonl``; harmful is read from the merged
    ``judged_<cond>.jsonl`` via the correct nested label path. Returns ``None``
    when the raw inputs are absent (caller falls back to the stored value).
    """
    coh_rows = _read_jsonl(seed_dir / "coherence.jsonl")
    judged = _read_jsonl(seed_dir / f"judged_{cond}.jsonl")
    if not coh_rows or not judged:
        return None
    clean_ids = {str(r["id"]) for r in coh_rows if r.get("is_clean")}
    clean_safety = [s for s in judged if str(s.get("id")) in clean_ids]
    if not clean_safety:
        return None
    n_harmful = sum(1 for s in clean_safety if _harmful_label(s))
    return n_harmful / len(clean_safety)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def _num(x, fmt: str = ".3f") -> str:
    return "n/a" if x is None else format(float(x), fmt)


def _first(d: dict | None) -> dict:
    if not d:
        return {}
    return next(iter(d.values()), {}) or {}


def _badge(status: str) -> str:
    cls = {"complete": "good", "partial": "warn", "missing": "bad"}.get(status, "")
    return f'<span class="badge {cls}">{status}</span>'


# ---------------------------------------------------------------------------
# Cell discovery
# ---------------------------------------------------------------------------
def _collect_cells() -> list[dict]:
    cells: list[dict] = []
    for model in MODELS:
        for root_name, src_label in SOURCES:
            root = RUN_ROOT / model / root_name
            if not root.exists():
                continue
            # only include a source if it has at least one summary anywhere
            if not any(root.rglob("summary.json")):
                continue
            for dset in DATASETS:
                for cond in CONDITIONS:
                    seed_dir = root / dset / cond / "seed0"
                    summary = _read_json(seed_dir / "summary.json")
                    sr = _read_json(seed_dir / "safety_reasoning.summary.json")
                    cells.append(_describe_cell(
                        model, src_label, dset, cond, seed_dir, summary, sr))
    return cells


def _describe_cell(model, src, dset, cond, seed_dir, summary, sr) -> dict:
    basic = _first(summary.get("per_condition_basic")) if summary else {}
    coh = (summary or {}).get("coherence") or {}
    mon = _first((summary or {}).get("monitorability", {}).get("per_condition")) if summary else {}
    pathway = _first(summary.get("per_condition_pathway")) if summary else {}
    pw_mean = (pathway or {}).get("mean") or {}

    n_rows = (summary or {}).get("n_completions")

    # ASR-clean (harmful_among_clean): re-derive from raw judgments so the
    # report is immune to the stale/buggy stored value. Fall back to stored.
    hac = _derive_harmful_among_clean(seed_dir, cond)
    hac_derived = hac is not None
    if hac is None:
        hac = coh.get("harmful_among_clean_rate")

    # status per pass
    if summary is None:
        std_status = "missing"
    elif basic and coh:
        std_status = "complete"
    else:
        std_status = "partial"
    mon_status = "complete" if (mon and mon.get("asr_cot_pred") is not None) else "missing"
    pw_status = "complete" if pw_mean else "missing"
    sr_status = "complete" if sr else "missing"

    return {
        "model": model, "src": src, "dataset": dset, "condition": cond,
        "n_rows": n_rows, "judge_model": (summary or {}).get("judge_model"),
        "std_status": std_status, "mon_status": mon_status,
        "pw_status": pw_status, "sr_status": sr_status,
        "harmful_among_clean": hac, "hac_derived": hac_derived,
        "basic": basic, "coh": coh, "mon": mon, "pw": pw_mean,
        "dominant_hist": (pathway or {}).get("dominant_pathway_hist") or {},
        "sr": sr or {},
    }


def _cond_rank(cond: str) -> int:
    try:
        return CONDITIONS.index(cond)
    except ValueError:
        return 99


def _sort_key(c: dict):
    return (c["model"], c["src"], c["dataset"], _cond_rank(c["condition"]))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#172026;--muted:#64717d;--line:#d9e0e6;--soft:#eef3f6;--paper:#fff;--bg:#f7f8fa;}
*{box-sizing:border-box;}
body{margin:0;color:var(--ink);background:var(--bg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}
main{max-width:1500px;margin:0 auto;padding:26px 22px 60px;}
header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px;}
h1{margin:0 0 6px;font-size:26px;}
h2{margin:30px 0 10px;font-size:18px;}
p{color:var(--muted);line-height:1.45;margin:6px 0;}
table{width:100%;border-collapse:collapse;background:var(--paper);border:1px solid var(--line);margin:10px 0 16px;}
th,td{border-bottom:1px solid #e7ebef;padding:7px 9px;text-align:left;font-size:13px;vertical-align:top;}
th{background:var(--soft);font-weight:650;color:#24313a;position:sticky;top:0;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
.badge{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600;}
.badge.good{background:#e7f6ee;color:#137a43;border:1px solid #a6dcc0;}
.badge.warn{background:#fff5e6;color:#9a5b12;border:1px solid #e6c28c;}
.badge.bad{background:#fdecec;color:#9a2020;border:1px solid #e6a8a8;}
.grid{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px;margin:6px 0 18px;}
.stat{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px;}
.stat .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;}
.stat .value{margin-top:6px;font-size:24px;font-weight:720;}
tr:hover td{background:#fbfcfd;}
.src{font-size:11px;color:var(--muted);}
"""


def _table(headers, rows, numeric=None) -> str:
    numeric = numeric or set()
    out = ["<table><thead><tr>"]
    for i, h in enumerate(headers):
        out.append(f'<th{" class=num" if i in numeric else ""}>{html.escape(h)}</th>')
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            out.append(f'<td{" class=num" if i in numeric else ""}>{cell}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _id_cols(c: dict) -> list[str]:
    return [
        html.escape(_disp(c["model"])),
        f'<span class="src">{html.escape(c["src"])}</span>',
        html.escape(_disp(c["dataset"])),
        html.escape(c["condition"]),
    ]


def _stat_cards(cells: list[dict]) -> str:
    judged = [c for c in cells if c["std_status"] != "missing"]
    complete = [c for c in cells if c["std_status"] == "complete"]
    sr_done = [c for c in cells if c["sr_status"] == "complete"]
    srcs = sorted({(c["model"], c["src"]) for c in judged})
    cards = [
        ("Grid cells", str(len(cells)), "model × source × dataset × condition"),
        ("Std-metrics judged", f"{len(complete)}/{len(cells)}", "5-label + coherence complete"),
        ("CoT-trace judged", f"{len(sr_done)}/{len(cells)}", "safety-reasoning trace present"),
        ("Active sources", str(len(srcs)),
         ", ".join(f"{_disp(m)}·{s}" for m, s in srcs) or "none"),
    ]
    out = ['<div class="grid">']
    for label, value, sub in cards:
        out.append(f'<div class="stat"><div class="label">{html.escape(label)}</div>'
                   f'<div class="value">{html.escape(value)}</div>'
                   f'<p style="margin:4px 0 0">{html.escape(sub)}</p></div>')
    out.append("</div>")
    return "\n".join(out)


def _judges_table(cells: list[dict]) -> str:
    judge_models = sorted({c["judge_model"] for c in cells if c["judge_model"]})
    jm = judge_models[0] if judge_models else "n/a"
    rows = [
        ["5-label safety judge", "harmful_response, safe_rejection, reasoning_about_safety, adding_intention, changing_subject", jm],
        ["Coherence / gibberish gate", "clean_rate, empty_rate, repeat3, compression (heuristic, no LLM)", "heuristic classifier (CPU)"],
        ["CoT-only monitor", "cot_predicts_unsafe → asr_cot_pred, monitorability gap", jm],
        ["Indexed safety-reasoning trace", "per-line safety-reasoning location + category (risk/policy/intent/refusal)", jm],
        ["Pathway 12-label (prefix)", "8-dim pathway vector + dominant_pathway", jm],
    ]
    rows = [[html.escape(a), html.escape(b), html.escape(c)] for a, b, c in rows]
    return _table(["Judge / instrument", "Produces", "Model"], rows)


def _status_table(cells: list[dict]) -> str:
    headers = ["Model", "Src", "Dataset", "Condition", "Rows",
               "Std metrics", "Monitor (CoT)", "Pathway", "Safety-reasoning"]
    rows = []
    for c in cells:
        rows.append(_id_cols(c) + [
            "n/a" if c["n_rows"] is None else str(c["n_rows"]),
            _badge(c["std_status"]), _badge(c["mon_status"]),
            _badge(c["pw_status"]), _badge(c["sr_status"]),
        ])
    return _table(headers, rows, numeric={4})


def _core_metrics_table(cells: list[dict]) -> str:
    headers = ["Model", "Src", "Dataset", "Condition",
               "Harmful", "Refusal", "Clean", "ASR (clean)", "Reasoning (final)"]
    rows = []
    for c in cells:
        b, coh = c["basic"], c["coh"]
        asr_clean = _pct(c["harmful_among_clean"])
        if c["harmful_among_clean"] is not None and not c["hac_derived"]:
            asr_clean += '<span class="src"> (stored)</span>'
        rows.append(_id_cols(c) + [
            _pct(b.get("harmful_rate")), _pct(b.get("refusal_rate")),
            _pct(coh.get("clean_rate")), asr_clean,
            _pct(b.get("reasoning_rate")),
        ])
    return _table(headers, rows, numeric=set(range(4, 9)))


def _monitor_table(cells: list[dict]) -> str:
    headers = ["Model", "Src", "Dataset", "Condition",
               "ASR final", "ASR CoT-pred", "Gap"]
    rows = []
    for c in cells:
        m = c["mon"]
        rows.append(_id_cols(c) + [
            _pct(m.get("asr_final")), _pct(m.get("asr_cot_pred")),
            _num(m.get("gap"), "+.3f"),
        ])
    return _table(headers, rows, numeric={4, 5, 6})


def _cot_trace_table(cells: list[dict]) -> str:
    headers = ["Model", "Src", "Dataset", "Condition",
               "Trace parsed", "SR rate", "First pos", "Extent frac",
               "Risk", "Policy", "Intent", "Refusal"]
    rows = []
    for c in cells:
        sr = c["sr"]
        parsed = "n/a"
        if sr:
            parsed = f'{sr.get("n_parsed", 0)}/{sr.get("n", 0)}'
        hist = sr.get("span_category_hist") or {}
        rows.append(_id_cols(c) + [
            parsed,
            _pct(sr.get("safety_reasoning_rate")) if sr else "n/a",
            _num(sr.get("first_position_norm_mean")) if sr else "n/a",
            _pct(sr.get("extent_fraction_mean")) if sr else "n/a",
            str(hist.get("risk_acknowledgment", "n/a") if sr else "n/a"),
            str(hist.get("policy_boundary", "n/a") if sr else "n/a"),
            str(hist.get("intent_assessment", "n/a") if sr else "n/a"),
            str(hist.get("refusal_reasoning", "n/a") if sr else "n/a"),
        ])
    return _table(headers, rows, numeric=set(range(4, 12)))


def _pathway_table(cells: list[dict]) -> str:
    headers = ["Model", "Src", "Dataset", "Condition",
               "Recognition", "Refusal supp.", "Execution", "Op. detail",
               "Dominant pathway"]
    rows = []
    for c in cells:
        pw = c["pw"]
        dom = c["dominant_hist"]
        dom_str = max(dom, key=dom.get) if dom else "n/a"
        rows.append(_id_cols(c) + [
            _pct(pw.get("recognition_rate")) if pw else "n/a",
            _pct(pw.get("refusal_suppressed")) if pw else "n/a",
            _pct(pw.get("execution_rate")) if pw else "n/a",
            _pct(pw.get("operational_detail_rate")) if pw else "n/a",
            html.escape(dom_str),
        ])
    return _table(headers, rows, numeric={4, 5, 6, 7})


def build_report(cells: list[dict]) -> str:
    cells = sorted(cells, key=_sort_key)
    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    body = [
        _stat_cards(cells),
        "<h2>1. Judges / instruments</h2>", _judges_table(cells),
        "<h2>2. Coverage &amp; status</h2>",
        "<p>complete = pass ran and aggregated; partial = summary present but a "
        "block missing; missing = not judged yet.</p>", _status_table(cells),
        "<h2>3. Core metrics — potency &amp; quality</h2>",
        "<p>ASR (clean) = harmful_among_clean_rate, the headline potency input, "
        "<b>re-derived from raw judgments</b> (coherence.jsonl &amp; "
        "judged_&lt;cond&gt;.jsonl) so it is immune to the stale stored value; "
        "cells tagged <span class=\"src\">(stored)</span> fell back to "
        "summary.json. All other values are per-condition averages.</p>",
        _core_metrics_table(cells),
        "<h2>4. Monitorability (CoT-only monitor)</h2>", _monitor_table(cells),
        "<h2>5. CoT-trace safety reasoning (indexed judge)</h2>",
        "<p>Trace parsed = traces the indexed judge returned valid JSON for; "
        "Risk/Policy/Intent/Refusal are span-category counts.</p>",
        _cot_trace_table(cells),
        "<h2>6. Pathway vector</h2>", _pathway_table(cells),
    ]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Direction A v5 — Metrics &amp; Status</title><style>{CSS}</style></head>
<body><main><header><h1>Direction A v5 — Metrics &amp; Judging Status</h1>
<p>Generated {html.escape(generated)}. Sources: full = n=100 judge tree, n25 = 25-row subset.</p>
</header>{''.join(body)}</main></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUN_ROOT / "metrics_status_report.html"))
    args = ap.parse_args()
    cells = _collect_cells()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(cells), encoding="utf-8")
    print(f"wrote {out_path}  ({len(cells)} cells across "
          f"{len({(c['model'], c['src']) for c in cells})} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
