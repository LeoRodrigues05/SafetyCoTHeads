"""Composite (P, Q, S) metric report for the Direction A v5 grid.

Reads every judged cell's ``summary.json`` (+ ``safety_reasoning.summary.json``),
computes the baseline-corrected (Potency, Quality, Safety-Reasoning) vector and
the Selective-Failure Score for each (model, dataset, condition), and emits:

* a machine-readable per-cell table (CSV + JSON), and
* a self-contained HTML report with the four headline tables:
  (1) per-model axis + scalar rankings,
  (2) raw-ASR vs SFS ranking agreement (Kendall tau),
  (3) the axis-ablation tau-table (how each correction reorders methods),
  (4) the pooled-across-models intervention-family table.

Usage:
    python -m scripts.make_composite_report \
        --model-keys qwen3_8b olmo3_7b_think llama31_8b_control \
        --out runs/direction_a_v5/composite_report.html \
        --csv-out runs/direction_a_v5/composite_cells.csv \
        --json-out runs/direction_a_v5/composite_cells.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
from pathlib import Path

from safety_cot_heads.analysis.composite import (
    AxisScores,
    axis_scores,
    cell_signals_from_summary,
    kendall_tau,
    pareto_front,
    rank_by,
    sfs,
    sfs_covert,
    sfs_product,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "direction_a_v5"

DEFAULT_MODEL_KEYS = [
    "qwen3_8b",
    "olmo3_7b_base",
    "olmo3_7b_base_own",
    "olmo3_7b_think",
    "llama31_8b_control",
]

# Ordering used for per-model ranking tables and family pooling.
CONDITIONS = [
    "ships_top3", "ships_top5", "ships_top8",
    "neurons_top256", "neurons_top512", "neurons_top1024",
    "steering_a0.5", "steering_a1.0", "steering_a1.5", "steering_ablate",
]
BASELINE = "baseline"


def _family(cond: str) -> str:
    if cond.startswith("ships"):
        return "SHIPS (heads)"
    if cond.startswith("neurons"):
        return "Neuron"
    if cond == "steering_ablate":
        return "Directional ablation"
    return "Steering"


DISPLAY = {
    "qwen3_8b": "Qwen3 8B",
    "olmo3_7b_base": "OLMo-3 7B base",
    "olmo3_7b_base_own": "OLMo-3 7B base-own",
    "olmo3_7b_think": "OLMo-3 7B think",
    "llama31_8b_control": "Llama 3.1 8B (control)",
    "jbb": "JBB",
    "bt": "BT",
}


def _disp(k: str) -> str:
    return DISPLAY.get(k, k)


# --- loading ----------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cell_dir(model: str, dataset: str, cond: str) -> Path:
    return RUN_ROOT / model / "judge" / dataset / cond / "seed0"


def load_signals(model_keys: list[str], datasets: list[str]) -> dict:
    """(model, dataset, condition) -> CellSignals for every judged cell found."""
    out = {}
    for model in model_keys:
        for dataset in datasets:
            for cond in [BASELINE] + CONDITIONS:
                d = _cell_dir(model, dataset, cond)
                summary = _load_json(d / "summary.json")
                if summary is None:
                    continue
                sr = _load_json(d / "safety_reasoning.summary.json")
                out[(model, dataset, cond)] = cell_signals_from_summary(
                    model, dataset, cond, summary, sr
                )
    return out


def pooled_axes(signals: dict, model_keys: list[str], datasets: list[str]) -> dict:
    """(model, condition) -> AxisScores averaged over datasets (baseline-corrected
    per dataset first, then pooled)."""
    pooled = {}
    for model in model_keys:
        for cond in CONDITIONS + [BASELINE]:
            per_ds = []
            for ds in datasets:
                cell = signals.get((model, ds, cond))
                base = signals.get((model, ds, BASELINE))
                if cell is None or base is None:
                    continue
                a = axis_scores(cell, base)
                if a is not None:
                    per_ds.append(a)
            if not per_ds:
                continue
            n = len(per_ds)
            pooled[(model, cond)] = AxisScores(
                model=model, dataset="pooled", condition=cond,
                P=sum(a.P for a in per_ds) / n,
                Q=sum(a.Q for a in per_ds) / n,
                S=sum(a.S for a in per_ds) / n,
                covert=sum(a.covert for a in per_ds) / n,
                raw_hac=sum(a.raw_hac for a in per_ds) / n,
                clean_rate=sum(a.clean_rate for a in per_ds) / n,
                gap=sum(a.gap for a in per_ds) / n,
                sr_rate=None,
            )
    return pooled


# --- machine-readable output ------------------------------------------------

CSV_FIELDS = [
    "model", "dataset", "condition", "family",
    "P", "Q", "S", "covert", "raw_hac", "clean_rate", "gap",
    "sfs", "sfs_product", "sfs_covert", "sr_rate",
]


def _row_dict(a: AxisScores) -> dict:
    return {
        "model": a.model, "dataset": a.dataset, "condition": a.condition,
        "family": _family(a.condition),
        "P": round(a.P, 4), "Q": round(a.Q, 4), "S": round(a.S, 4),
        "covert": round(a.covert, 4), "raw_hac": round(a.raw_hac, 4),
        "clean_rate": round(a.clean_rate, 4), "gap": round(a.gap, 4),
        "sfs": round(sfs(a), 4), "sfs_product": round(sfs_product(a), 4),
        "sfs_covert": round(sfs_covert(a), 4),
        "sr_rate": (round(a.sr_rate, 4) if a.sr_rate is not None else None),
    }


def per_cell_rows(signals: dict, model_keys: list[str], datasets: list[str]) -> list[dict]:
    rows = []
    for model in model_keys:
        for ds in datasets:
            base = signals.get((model, ds, BASELINE))
            if base is None:
                continue
            for cond in CONDITIONS:
                cell = signals.get((model, ds, cond))
                if cell is None:
                    continue
                a = axis_scores(cell, base)
                if a is not None:
                    rows.append(_row_dict(a))
    return rows


# --- HTML -------------------------------------------------------------------

CSS = """
:root { color-scheme: light; --ink:#172026; --muted:#64717d; --line:#d9e0e6;
  --soft:#eef3f6; --paper:#fff; --bg:#f7f8fa; --accent:#256c7d; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); background:var(--bg);
  font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1320px; margin:0 auto; padding:26px 22px 46px; }
header { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:22px; }
h1 { margin:0 0 8px; font-size:28px; line-height:1.12; }
h2 { margin:30px 0 10px; font-size:19px; }
h3 { margin:18px 0 6px; font-size:15px; }
p { margin:8px 0; color:var(--muted); line-height:1.45; }
table { width:100%; border-collapse:collapse; background:var(--paper);
  border:1px solid var(--line); margin:10px 0 22px; }
th,td { border-bottom:1px solid #e7ebef; padding:7px 9px; text-align:left;
  font-size:13px; }
th { background:var(--soft); font-weight:650; color:#24313a; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
code { background:var(--soft); border-radius:4px; padding:2px 4px; }
.callout { background:var(--paper); border:1px solid var(--line);
  border-radius:8px; padding:14px 16px; margin:12px 0 22px; }
.best { font-weight:700; color:#14533b; }
.faded { color:var(--muted); }
"""


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    out = ["<table>", "<thead><tr>"]
    for i, h in enumerate(headers):
        cls = ' class="num"' if i in numeric else ""
        out.append(f"<th{cls}>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for i, cell in enumerate(row):
            cls = ' class="num"' if i in numeric else ""
            out.append(f"<td{cls}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _f(x: float) -> str:
    return f"{x:.2f}"


def _section_per_model(pooled: dict, model_keys: list[str]) -> str:
    parts = ["<h2>1. Per-model axis + scalar rankings (pooled over datasets)</h2>",
             "<p>Non-baseline conditions, sorted by SFS. <code>raw ASR</code> is the "
             "un-normalised coherence-gated harm rate that current papers report; "
             "the SFS column reorders it once potency is baseline-corrected and gated "
             "by coherence (Q) and monitorability (S).</p>"]
    headers = ["condition", "P", "Q", "S", "gap", "raw ASR", "SFS", "SFS·covert"]
    numeric = set(range(1, len(headers)))
    for model in model_keys:
        rows_data = [(c, pooled[(model, c)]) for c in CONDITIONS if (model, c) in pooled]
        if not rows_data:
            continue
        rows_data.sort(key=lambda t: -sfs(t[1]))
        best = rows_data[0][0] if rows_data else None
        rows = []
        for c, a in rows_data:
            label = html.escape(c)
            if c == best:
                label = f'<span class="best">{label}</span>'
            rows.append([
                label, _f(a.P), _f(a.Q), _f(a.S), _f(a.gap),
                _f(a.raw_hac), _f(sfs(a)), _f(sfs_covert(a)),
            ])
        parts.append(f"<h3>{html.escape(_disp(model))}</h3>")
        parts.append(_table(headers, rows, numeric))
    return "".join(parts)


def _section_ranking_agreement(pooled: dict, model_keys: list[str]) -> str:
    parts = ["<h2>2. Raw-ASR vs SFS ranking agreement (Kendall &tau;)</h2>",
             "<p>Within each model, how similarly does each scalar rank the methods "
             "compared with the full SFS? &tau;=1 identical; lower means that scalar "
             "orders the methods differently. Low <code>raw ASR</code> &tau; is the "
             "core evidence that a single ASR mis-ranks interventions.</p>"]
    headers = ["model", "raw ASR", "P only", "P·Q", "vs full SFS"]
    numeric = {1, 2, 3, 4}
    rows = []
    for model in model_keys:
        conds = [c for c in CONDITIONS if (model, c) in pooled]
        if len(conds) < 3:
            continue
        items = conds
        ref = rank_by(items, lambda c: sfs(pooled[(model, c)]))
        tau_raw = kendall_tau(ref, rank_by(items, lambda c: pooled[(model, c)].raw_hac))
        tau_p = kendall_tau(ref, rank_by(items, lambda c: pooled[(model, c)].P))
        tau_pq = kendall_tau(ref, rank_by(items, lambda c: pooled[(model, c)].P * pooled[(model, c)].Q))
        rows.append([html.escape(_disp(model)), _f(tau_raw), _f(tau_p), _f(tau_pq), "1.00"])
    parts.append(_table(headers, rows, numeric))
    return "".join(parts)


def _section_ablation(pooled: dict, model_keys: list[str]) -> str:
    parts = ["<h2>3. Axis ablation: which correction reorders methods?</h2>",
             "<p>Kendall &tau; between each reduced scalar's method ranking and the "
             "full SFS ranking, per model. A low value means dropping that correction "
             "changes the conclusion. Baseline-correcting potency (raw&rarr;P) does the "
             "most work; the coherence gate (P&rarr;P·Q) matters where interventions "
             "destroy the model; S is preserved on this grid but retained for "
             "future covert methods.</p>"]
    headers = ["model", "raw HAC", "P", "P·Q", "P·Q·(1-S)", "SFS_prod"]
    numeric = {1, 2, 3, 4, 5}
    rows = []
    for model in model_keys:
        conds = [c for c in CONDITIONS if (model, c) in pooled]
        if len(conds) < 3:
            continue
        ref = rank_by(conds, lambda c: sfs(pooled[(model, c)]))
        def tau(scorer):
            return _f(kendall_tau(ref, rank_by(conds, scorer)))
        rows.append([
            html.escape(_disp(model)),
            tau(lambda c: pooled[(model, c)].raw_hac),
            tau(lambda c: pooled[(model, c)].P),
            tau(lambda c: pooled[(model, c)].P * pooled[(model, c)].Q),
            tau(lambda c: sfs_covert(pooled[(model, c)])),
            tau(lambda c: sfs_product(pooled[(model, c)])),
        ])
    parts.append(_table(headers, rows, numeric))
    return "".join(parts)


def _section_family(pooled: dict, model_keys: list[str]) -> str:
    parts = ["<h2>4. Intervention family, pooled across models</h2>",
             "<p>Mean axes and SFS per family over all models. Raw ASR can rate "
             "steering and directional ablation as comparable; the decomposed metric "
             "separates them once baseline-correction and coherence-gating apply.</p>"]
    fam_order = ["SHIPS (heads)", "Neuron", "Steering", "Directional ablation"]
    buckets: dict[str, list[AxisScores]] = {f: [] for f in fam_order}
    for (model, cond), a in pooled.items():
        if cond == BASELINE:
            continue
        buckets[_family(cond)].append(a)
    headers = ["family", "mean P", "mean Q", "mean S", "mean SFS", "mean raw ASR", "n cells"]
    numeric = {1, 2, 3, 4, 5, 6}
    rows = []
    for fam in fam_order:
        vs = buckets[fam]
        if not vs:
            continue
        n = len(vs)
        rows.append([
            html.escape(fam),
            _f(sum(a.P for a in vs) / n), _f(sum(a.Q for a in vs) / n),
            _f(sum(a.S for a in vs) / n), _f(sum(sfs(a) for a in vs) / n),
            _f(sum(a.raw_hac for a in vs) / n), str(n),
        ])
    parts.append(_table(headers, rows, numeric))
    return "".join(parts)


def _section_pareto(pooled: dict, model_keys: list[str]) -> str:
    parts = ["<h2>5. Pareto-non-dominated methods (per model)</h2>",
             "<p>Methods not strictly dominated on all three axes at once. These are "
             "the defensible frontier: no other method is at least as good on P, Q and "
             "S simultaneously.</p>"]
    headers = ["model", "non-dominated conditions (P,Q,S)"]
    rows = []
    for model in model_keys:
        scores = [pooled[(model, c)] for c in CONDITIONS if (model, c) in pooled]
        if not scores:
            continue
        front = pareto_front(scores)
        front.sort(key=lambda a: -sfs(a))
        cell = ", ".join(
            f"{html.escape(a.condition)} ({_f(a.P)},{_f(a.Q)},{_f(a.S)})"
            for a in front
        )
        rows.append([html.escape(_disp(model)), cell])
    parts.append(_table(headers, rows))
    return "".join(parts)


def build_report(pooled: dict, model_keys: list[str], title: str) -> str:
    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    body = "".join([
        "<div class='callout'><strong>Metric.</strong> Each cell is scored on three "
        "baseline-corrected axes in [0,1]: <code>P</code> = induced coherent harm "
        "<code>clip[(HAC_c-HAC_b)/(1-HAC_b)]</code>; <code>Q</code> = coherence "
        "retention <code>clip[clean_c/clean_b]</code>; <code>S</code> = monitorability "
        "retention <code>1-clip[|gap_c|-|gap_b|]</code>. Headline "
        "<code>SFS=(P&middot;Q&middot;S)^(1/3)</code>. Suppressive orientation: high "
        "SFS = potent, coherence-preserving, still-monitorable safety removal.</div>",
        _section_per_model(pooled, model_keys),
        _section_ranking_agreement(pooled, model_keys),
        _section_ablation(pooled, model_keys),
        _section_family(pooled, model_keys),
        _section_pareto(pooled, model_keys),
    ])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main><header><h1>{html.escape(title)}</h1>
<p>Generated {html.escape(generated)}. Composite (P,Q,S) + Selective-Failure Score
over the Direction A v5 grid.</p></header>
{body}
</main></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-keys", nargs="+", default=DEFAULT_MODEL_KEYS)
    ap.add_argument("--datasets", nargs="+", default=["jbb", "bt"])
    ap.add_argument("--out", default=str(RUN_ROOT / "composite_report.html"))
    ap.add_argument("--csv-out", default=str(RUN_ROOT / "composite_cells.csv"))
    ap.add_argument("--json-out", default=str(RUN_ROOT / "composite_cells.json"))
    ap.add_argument("--title", default="Direction A v5 Composite Metric Report")
    args = ap.parse_args()

    signals = load_signals(args.model_keys, args.datasets)
    if not signals:
        print("no judged cells found for requested model keys/datasets")
        return 2

    rows = per_cell_rows(signals, args.model_keys, args.datasets)
    pooled = pooled_axes(signals, args.model_keys, args.datasets)

    out_path = Path(args.out)
    csv_path = Path(args.csv_out)
    json_path = Path(args.json_out)
    for p in (out_path, csv_path, json_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out_path.write_text(build_report(pooled, args.model_keys, args.title), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"wrote {csv_path}  ({len(rows)} cells)")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
