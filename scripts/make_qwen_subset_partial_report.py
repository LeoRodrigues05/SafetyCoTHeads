"""Build a partial HTML report for the Qwen v5 subset judge run.

This is intentionally tolerant of in-flight jobs: missing summaries are shown
as incomplete, and safety-reasoning JSONL files are summarized even when the
final ``safety_reasoning.summary.json`` has not been written yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT / "runs" / "direction_a_v5" / "qwen3_8b" / "judge_subset_n25_fast"
)

CONDITION_ORDER = (
    "baseline",
    "neurons_top256",
    "neurons_top512",
    "neurons_top1024",
    "ships_top3",
    "ships_top5",
    "ships_top8",
    "steering_a0.5",
    "steering_a1.0",
    "steering_a1.5",
)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                break
    return rows


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


def num(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{digits}f}"


def cell_dir(base: Path, dataset: str, condition: str) -> Path:
    return base / dataset / condition / "seed0"


def subset_path(base: Path, dataset: str, condition: str) -> Path:
    return (
        base / "_subsets" / dataset / condition / "seed0"
        / f"completions_{condition}.jsonl"
    )


def summarize_reasoning_rows(rows: list[dict]) -> dict:
    parsed = [r for r in rows if isinstance(r.get("judge_flat"), dict)]
    if not parsed:
        return {
            "n": len(rows),
            "n_parsed": 0,
            "safety_reasoning_rate": None,
            "first_position_norm_mean": None,
            "extent_sentence_count_mean": None,
            "extent_fraction_mean": None,
            "first_section_hist": {},
            "span_category_hist": {},
        }
    has = []
    first_norm = []
    extent_counts = []
    extent_fracs = []
    first_section = {}
    categories = {}
    for row in parsed:
        flat = row.get("judge_flat") or {}
        has.append(bool(flat.get("has_safety_reasoning")))
        pos = flat.get("position") or {}
        extent = flat.get("extent") or {}
        if pos.get("first_global_index") is not None:
            denom = max(1, int(row.get("n_trace_segments") or 1) - 1)
            first_norm.append(float(pos["first_global_index"]) / denom)
        if pos.get("first_section"):
            sec = str(pos["first_section"])
            first_section[sec] = first_section.get(sec, 0) + 1
        if extent.get("sentence_count") is not None:
            extent_counts.append(float(extent["sentence_count"]))
        if extent.get("fraction_of_sentences") is not None:
            extent_fracs.append(float(extent["fraction_of_sentences"]))
        for span in flat.get("safety_reasoning_sentence_indexes") or []:
            cat = span.get("category") or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
    n = len(parsed)
    return {
        "n": len(rows),
        "n_parsed": n,
        "safety_reasoning_rate": sum(1 for x in has if x) / n,
        "first_position_norm_mean": mean(first_norm) if first_norm else None,
        "extent_sentence_count_mean": mean(extent_counts) if extent_counts else None,
        "extent_fraction_mean": mean(extent_fracs) if extent_fracs else None,
        "first_section_hist": first_section,
        "span_category_hist": categories,
    }


def load_cell(base: Path, dataset: str, condition: str) -> dict:
    out = cell_dir(base, dataset, condition)
    subset_rows = read_jsonl(subset_path(base, dataset, condition))
    n_expected = len(subset_rows)
    summary = read_json(out / "summary.json")
    sr_summary = read_json(out / "safety_reasoning.summary.json")
    sr_rows = read_jsonl(out / "judge_safety_reasoning_trace.jsonl")
    if sr_summary is None and sr_rows:
        sr_summary = summarize_reasoning_rows(sr_rows)

    basic = None
    monitor = None
    coherence = None
    if summary:
        basic = (summary.get("per_condition_basic") or {}).get(condition)
        monitor = (summary.get("monitorability") or {}).get("per_condition", {}).get(condition)
        coherence = summary.get("coherence")

    judged_path = out / f"judged_{condition}.jsonl"
    cot_path = out / "judge_cot_only.jsonl"
    judged_rows = read_jsonl(judged_path)
    cot_rows = read_jsonl(cot_path)

    return {
        "dataset": dataset,
        "condition": condition,
        "n_expected": n_expected,
        "standard_complete": summary is not None,
        "safety_reasoning_complete": (out / "safety_reasoning.summary.json").exists(),
        "n_judged": len(judged_rows),
        "n_cot_only": len(cot_rows),
        "n_safety_reasoning": len(sr_rows),
        "n_safety_reasoning_parsed": (sr_summary or {}).get("n_parsed"),
        "basic": basic or {},
        "monitor": monitor or {},
        "coherence": coherence or {},
        "safety_reasoning": sr_summary or {},
    }


def discover_cells(base: Path) -> list[tuple[str, str]]:
    cells = []
    subsets = base / "_subsets"
    for dataset_dir in sorted(p for p in subsets.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for cond_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            cells.append((dataset, cond_dir.name))
    return sorted(
        cells,
        key=lambda x: (
            0 if x[0] == "bt" else 1,
            CONDITION_ORDER.index(x[1]) if x[1] in CONDITION_ORDER else 99,
            x[1],
        ),
    )


def status_label(done: bool) -> str:
    return "complete" if done else "missing"


def td(x, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{x}</td>"


def render(rows: list[dict], base: Path) -> str:
    generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    standard_done = sum(1 for r in rows if r["standard_complete"])
    reasoning_done = sum(1 for r in rows if r["safety_reasoning_complete"])
    total = len(rows)
    parsed = sum(int(r.get("n_safety_reasoning_parsed") or 0) for r in rows)
    reason_rows = sum(int(r.get("n_safety_reasoning") or 0) for r in rows)
    css = """
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f8fa; color: #172026; }
main { max-width: 1360px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 8px; font-size: 26px; }
p { color: #5f6b76; line-height: 1.45; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }
.stat { background: #fff; border: 1px solid #dbe2e8; border-radius: 8px; padding: 12px; }
.label { color: #66737f; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.value { margin-top: 5px; font-size: 24px; font-weight: 750; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe2e8; margin: 16px 0 28px; }
th, td { padding: 8px 9px; border-bottom: 1px solid #e7ebef; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #eef3f6; position: sticky; top: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #13633d; font-weight: 650; }
.miss { color: #a15c20; font-weight: 650; }
.muted { color: #66737f; }
code { background: #eef3f6; padding: 2px 4px; border-radius: 4px; }
"""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Qwen3 subset partial judge report</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>Qwen3 subset partial judge report</h1>",
        f"<p>Generated {html.escape(generated)} from <code>{html.escape(str(base))}</code>.</p>",
        "<div class='grid'>",
        f"<div class='stat'><div class='label'>Cells</div><div class='value'>{total}</div><p>Expected dataset-condition cells.</p></div>",
        f"<div class='stat'><div class='label'>Standard metrics</div><div class='value'>{standard_done}/{total}</div><p>Safety, coherence, CoT-only summary.</p></div>",
        f"<div class='stat'><div class='label'>Safety reasoning</div><div class='value'>{reasoning_done}/{total}</div><p>Completed safety-reasoning summaries.</p></div>",
        f"<div class='stat'><div class='label'>Parsed reasoning rows</div><div class='value'>{parsed}/{reason_rows}</div><p>Parsed rows among written reasoning rows.</p></div>",
        "</div>",
        "<h2>Per-cell status and metrics</h2>",
        "<table><thead><tr>",
    ]
    headers = [
        "Dataset", "Condition", "Std", "Reason", "Rows",
        "Harmful", "Refusal", "Safety reasoning final", "Clean",
        "ASR final", "ASR CoT pred", "Gap",
        "Trace safety reasoning", "Trace parsed", "First pos", "Extent frac",
    ]
    parts.extend(f"<th>{h}</th>" for h in headers)
    parts.append("</tr></thead><tbody>")
    for r in rows:
        basic = r["basic"]
        coherence = r["coherence"]
        monitor = r["monitor"]
        sr = r["safety_reasoning"]
        std_cls = "ok" if r["standard_complete"] else "miss"
        sr_cls = "ok" if r["safety_reasoning_complete"] else "miss"
        parts.append("<tr>")
        parts.append(td(html.escape(r["dataset"])))
        parts.append(td(html.escape(r["condition"])))
        parts.append(td(status_label(r["standard_complete"]), std_cls))
        parts.append(td(status_label(r["safety_reasoning_complete"]), sr_cls))
        parts.append(td(f"{r['n_judged']}/{r['n_expected']}", "num"))
        parts.append(td(pct(basic.get("harmful_rate")), "num"))
        parts.append(td(pct(basic.get("refusal_rate")), "num"))
        parts.append(td(pct(basic.get("reasoning_rate")), "num"))
        parts.append(td(pct(coherence.get("clean_rate")), "num"))
        parts.append(td(pct(monitor.get("asr_final")), "num"))
        parts.append(td(pct(monitor.get("asr_cot_pred")), "num"))
        parts.append(td(pct(monitor.get("gap")), "num"))
        parts.append(td(pct(sr.get("safety_reasoning_rate")), "num"))
        parts.append(td(f"{sr.get('n_parsed', 0)}/{r['n_safety_reasoning']}", "num"))
        parts.append(td(num(sr.get("first_position_norm_mean")), "num"))
        parts.append(td(pct(sr.get("extent_fraction_mean")), "num"))
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Notes</h2>")
    parts.append(
        "<p>This report is intentionally partial. Missing cells were not on disk "
        "when the report was generated. The all-prefix pathway grid was skipped "
        "for this fast run, so pathway-vector metrics are absent by design.</p>"
    )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    base = args.base
    out = args.out or (base / "partial_report.html")
    json_out = args.json_out or (base / "partial_report.summary.json")
    rows = [load_cell(base, dataset, cond) for dataset, cond in discover_cells(base)]
    out.write_text(render(rows, base), encoding="utf-8")
    json_out.write_text(json.dumps({"base": str(base), "cells": rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
