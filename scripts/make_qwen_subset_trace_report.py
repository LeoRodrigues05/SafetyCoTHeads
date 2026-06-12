"""Build a trace browser for the Qwen v5 subset safety-reasoning judge.

The output is a static HTML file with full prompts/completions and segmented
CoT/output traces. Sentences flagged by the safety-reasoning judge are
highlighted and annotated with category + confidence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
from pathlib import Path

from safety_cot_heads.direction_a import segment_completion


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

CATEGORY_LABELS = {
    "risk_acknowledgment": "risk",
    "policy_boundary": "policy",
    "intent_assessment": "intent",
    "refusal_reasoning": "refusal",
    "safer_alternative": "alternative",
    "other_safety_reasoning": "other",
}


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


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def subset_path(base: Path, dataset: str, condition: str) -> Path:
    return (
        base / "_subsets" / dataset / condition / "seed0"
        / f"completions_{condition}.jsonl"
    )


def cell_dir(base: Path, dataset: str, condition: str) -> Path:
    return base / dataset / condition / "seed0"


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


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def trace_from_completion(completion: str) -> tuple[list[dict], str]:
    seg = segment_completion(completion or "")
    entries = []
    global_idx = 0
    for i, sent in enumerate(seg.think_sentences):
        entries.append({
            "global_index": global_idx,
            "section": "cot",
            "index": i,
            "text": sent,
        })
        global_idx += 1
    for i, sent in enumerate(seg.answer_sentences):
        entries.append({
            "global_index": global_idx,
            "section": "output",
            "index": i,
            "text": sent,
        })
        global_idx += 1
    if not entries and completion:
        entries.append({
            "global_index": 0,
            "section": "output",
            "index": 0,
            "text": completion,
        })
    return entries, seg.kind


def span_key(span: dict) -> tuple:
    return (
        span.get("global_index"),
        span.get("section"),
        span.get("index"),
    )


def entry_key(entry: dict) -> tuple:
    return (
        entry.get("global_index"),
        entry.get("section"),
        entry.get("index"),
    )


def safety_labels(row: dict | None) -> dict:
    if not row:
        return {}
    flat = row.get("judge_flat") or {}
    return flat.get("labels") or {}


def safety_reasoning_spans(row: dict | None) -> list[dict]:
    if not row:
        return []
    flat = row.get("judge_flat") or {}
    spans = flat.get("safety_reasoning_sentence_indexes") or []
    return [s for s in spans if isinstance(s, dict)]


def span_badge(span: dict) -> str:
    cat = str(span.get("category") or "other_safety_reasoning")
    conf = span.get("confidence")
    label = CATEGORY_LABELS.get(cat, cat.replace("_", " "))
    conf_txt = "" if conf is None else f" {float(conf):.2f}"
    return (
        f"<span class='reason-badge cat-{esc(cat)}'>"
        f"{esc(label)}{esc(conf_txt)}</span>"
    )


def metric_badges(labels: dict) -> str:
    items = [
        ("harmful_response", "harmful"),
        ("safe_rejection", "refusal"),
        ("reasoning_about_safety", "final safety reasoning"),
        ("adding_intention", "added intent"),
        ("changing_subject", "changed subject"),
    ]
    bits = []
    for key, label in items:
        val = labels.get(key)
        cls = "yes" if val is True else "no" if val is False else "unknown"
        txt = "?" if val is None else "yes" if val else "no"
        bits.append(f"<span class='metric {cls}'>{esc(label)}: {txt}</span>")
    return "\n".join(bits)


def render_trace(entries: list[dict], spans: list[dict]) -> str:
    by_key = {}
    by_section_index = {}
    for span in spans:
        by_key[span_key(span)] = span
        by_section_index[(span.get("section"), span.get("index"))] = span
    parts = ["<div class='trace'>"]
    for entry in entries:
        span = by_key.get(entry_key(entry))
        if span is None:
            span = by_section_index.get((entry.get("section"), entry.get("index")))
        section = entry.get("section") or "output"
        flagged = span is not None
        cls = f"trace-line {esc(section)}" + (" flagged" if flagged else "")
        parts.append(f"<div class='{cls}'>")
        idx = f"{entry.get('global_index')} | {section}:{entry.get('index')}"
        parts.append(f"<span class='idx'>{esc(idx)}</span>")
        if flagged:
            parts.append(span_badge(span))
        parts.append(f"<span class='text'>{esc(entry.get('text'))}</span>")
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def compact_prompt(prompt: str, limit: int = 220) -> str:
    prompt = re.sub(r"\s+", " ", prompt or "").strip()
    if len(prompt) <= limit:
        return prompt
    return prompt[: limit - 1] + "..."


def collect_rows(base: Path, only_flagged: bool) -> list[dict]:
    out_rows = []
    for dataset, condition in discover_cells(base):
        subset_rows = read_jsonl(subset_path(base, dataset, condition))
        if not subset_rows:
            continue
        out = cell_dir(base, dataset, condition)
        sr_rows = {
            str(r.get("id")): r
            for r in read_jsonl(out / "judge_safety_reasoning_trace.jsonl")
        }
        judged_path = out / f"judged_{condition}.jsonl"
        judged_rows = {
            str(r.get("id")): r
            for r in read_jsonl(judged_path)
        }
        summary = read_json(out / "safety_reasoning.summary.json")
        for comp in subset_rows:
            rid = str(comp.get("id"))
            sr = sr_rows.get(rid)
            spans = safety_reasoning_spans(sr)
            if only_flagged and not spans:
                continue
            entries = (sr or {}).get("trace_segments")
            trace_kind = "judge_trace"
            if not entries:
                entries, trace_kind = trace_from_completion(comp.get("completion") or "")
            out_rows.append({
                "dataset": dataset,
                "condition": condition,
                "id": rid,
                "category": comp.get("category"),
                "prompt": comp.get("prompt") or comp.get("user_prompt") or "",
                "completion": comp.get("completion") or "",
                "sr": sr,
                "safety": judged_rows.get(rid),
                "spans": spans,
                "entries": entries,
                "trace_kind": trace_kind,
                "summary_complete": summary is not None,
            })
    return out_rows


def render(rows: list[dict], base: Path, only_flagged: bool) -> str:
    generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    judged = sum(1 for r in rows if r["sr"] is not None)
    flagged = sum(1 for r in rows if r["spans"])
    css = """
body { margin: 0; background: #f6f8fa; color: #172026; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1360px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { margin: 28px 0 12px; font-size: 18px; }
p { color: #5f6b76; line-height: 1.45; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }
.stat { background: #fff; border: 1px solid #dbe2e8; border-radius: 8px; padding: 12px; }
.label { color: #66737f; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.value { margin-top: 5px; font-size: 24px; font-weight: 750; }
.sample { background: #fff; border: 1px solid #dbe2e8; border-radius: 8px; margin: 14px 0; overflow: hidden; }
.sample summary { cursor: pointer; padding: 12px 14px; background: #eef3f6; border-bottom: 1px solid #dbe2e8; }
.sample-body { padding: 14px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.pill, .metric, .reason-badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid #dbe2e8; background: #f7fafc; }
.metric.yes { border-color: #9fcbb7; background: #ecf8f3; color: #14533b; }
.metric.no { border-color: #d4dce4; background: #f7f9fb; color: #66737f; }
.metric.unknown { border-color: #e1c095; background: #fff7e8; color: #774a14; }
.prompt, pre.raw { white-space: pre-wrap; overflow-wrap: anywhere; background: #fbfcfd; border: 1px solid #e1e7ec; border-radius: 6px; padding: 10px; font-size: 12px; line-height: 1.45; }
.trace { border: 1px solid #dbe2e8; border-radius: 6px; overflow: hidden; background: #fff; }
.trace-line { display: grid; grid-template-columns: 110px minmax(0, auto) 1fr; gap: 8px; align-items: start; padding: 8px 10px; border-bottom: 1px solid #eef2f5; font-size: 13px; line-height: 1.42; }
.trace-line:last-child { border-bottom: 0; }
.trace-line.cot { background: #fbfcfd; }
.trace-line.output { background: #fff; }
.trace-line.flagged { background: #fff2ad; border-left: 5px solid #c98200; padding-left: 5px; }
.idx { color: #66737f; font-variant-numeric: tabular-nums; font-size: 12px; }
.text { overflow-wrap: anywhere; }
.reason-badge { border-color: #c98200; background: #fff8d8; color: #5d3b00; white-space: nowrap; }
.cat-risk_acknowledgment { background: #ffe8df; border-color: #e1a58e; color: #703018; }
.cat-policy_boundary { background: #e8f0ff; border-color: #aebfe8; color: #243d76; }
.cat-intent_assessment { background: #f3eaff; border-color: #cab1ed; color: #513077; }
.cat-refusal_reasoning { background: #e8f8f0; border-color: #9fcbb7; color: #14533b; }
.cat-safer_alternative { background: #e6f6fb; border-color: #98cbd8; color: #155566; }
code { background: #eef3f6; padding: 2px 4px; border-radius: 4px; }
details.rawbox { margin-top: 12px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr 1fr; } .trace-line { grid-template-columns: 1fr; } }
"""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Qwen3 subset safety reasoning traces</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>Qwen3 subset safety reasoning trace browser</h1>",
        f"<p>Generated {esc(generated)} from <code>{esc(base)}</code>. "
        f"{'Showing only samples with at least one judge-flagged safety-reasoning sentence.' if only_flagged else 'Showing all subset samples available on disk; unjudged traces are shown without highlights.'}</p>",
        "<div class='grid'>",
        f"<div class='stat'><div class='label'>Samples rendered</div><div class='value'>{len(rows)}</div></div>",
        f"<div class='stat'><div class='label'>Safety-reasoning judged</div><div class='value'>{judged}</div></div>",
        f"<div class='stat'><div class='label'>Flagged samples</div><div class='value'>{flagged}</div></div>",
        f"<div class='stat'><div class='label'>Flagged rate</div><div class='value'>{pct(flagged / judged if judged else None)}</div></div>",
        "</div>",
        "<h2>Completions and highlighted traces</h2>",
    ]
    for i, row in enumerate(rows, start=1):
        labels = safety_labels(row["safety"])
        flat = (row["sr"] or {}).get("judge_flat") or {}
        pos = flat.get("position") or {}
        extent = flat.get("extent") or {}
        status = "judged" if row["sr"] else "not judged yet"
        title = (
            f"{i}. {row['dataset']} / {row['condition']} / {row['id']} "
            f"({len(row['spans'])} highlighted, {status})"
        )
        parts.append("<details class='sample'>")
        parts.append(f"<summary>{esc(title)}</summary>")
        parts.append("<div class='sample-body'>")
        parts.append("<div class='meta'>")
        for key in ("dataset", "condition", "category", "id"):
            parts.append(f"<span class='pill'>{key}: {esc(row.get(key))}</span>")
        parts.append(f"<span class='pill'>trace: {esc(row['trace_kind'])}</span>")
        parts.append(f"<span class='pill'>first: {esc(pos.get('first_section'))}:{esc(pos.get('first_index'))}</span>")
        parts.append(f"<span class='pill'>extent: {esc(extent.get('sentence_count'))} sentences / {pct(extent.get('fraction_of_sentences'))}</span>")
        parts.append("</div>")
        parts.append("<div class='meta'>")
        parts.append(metric_badges(labels))
        parts.append("</div>")
        if row["spans"]:
            parts.append("<div class='meta'>")
            for span in row["spans"]:
                parts.append(span_badge(span))
            parts.append("</div>")
        parts.append("<h3>Prompt</h3>")
        parts.append(f"<div class='prompt'>{esc(row['prompt'])}</div>")
        parts.append("<h3>Segmented CoT + output</h3>")
        parts.append(render_trace(row["entries"], row["spans"]))
        parts.append("<details class='rawbox'><summary>Raw full completion</summary>")
        parts.append(f"<pre class='raw'>{esc(row['completion'])}</pre>")
        parts.append("</details>")
        parts.append("</div></details>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all", action="store_true",
                    help="Render all subset completions, including unjudged/unflagged rows.")
    args = ap.parse_args()
    only_flagged = not args.all
    out = args.out or (
        args.base / ("safety_reasoning_traces_flagged.html" if only_flagged
                     else "safety_reasoning_traces_all.html")
    )
    rows = collect_rows(args.base, only_flagged=only_flagged)
    out.write_text(render(rows, args.base, only_flagged), encoding="utf-8")
    print(f"wrote {out}")
    print(f"rendered {len(rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
