"""Build a wide side-by-side trace report for Qwen v5 interventions.

Rows are grouped by the same source query. Columns are interventions. Each cell
contains the full generated completion segmented into CoT/output sentences.
Safety-reasoning highlights are overlaid when the subset judge has produced
``judge_safety_reasoning_trace.jsonl`` for that query/intervention.
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
RUN_ROOT = ROOT / "runs" / "direction_a_v5"
DEFAULT_JUDGE_BASE = RUN_ROOT / "qwen3_8b" / "judge_subset_n25_fast"

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


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


def gen_path(model_key: str, dataset: str, condition: str) -> Path:
    return (
        RUN_ROOT / model_key / "gen" / dataset / condition / "seed0"
        / f"completions_{condition}.jsonl"
    )


def judge_dir(judge_base: Path, dataset: str, condition: str) -> Path:
    return judge_base / dataset / condition / "seed0"


def load_generation(model_key: str, dataset: str, conditions: list[str]) -> dict[str, dict[str, dict]]:
    out = {}
    for cond in conditions:
        rows = read_jsonl(gen_path(model_key, dataset, cond))
        out[cond] = {str(r.get("id")): r for r in rows}
    return out


def load_reasoning(judge_base: Path, dataset: str, conditions: list[str]) -> dict[str, dict[str, dict]]:
    out = {}
    for cond in conditions:
        rows = read_jsonl(judge_dir(judge_base, dataset, cond) / "judge_safety_reasoning_trace.jsonl")
        out[cond] = {str(r.get("id")): r for r in rows}
    return out


def load_safety(judge_base: Path, dataset: str, conditions: list[str]) -> dict[str, dict[str, dict]]:
    out = {}
    for cond in conditions:
        rows = read_jsonl(judge_dir(judge_base, dataset, cond) / f"judged_{cond}.jsonl")
        out[cond] = {str(r.get("id")): r for r in rows}
    return out


def condition_list(model_key: str, dataset: str) -> list[str]:
    root = RUN_ROOT / model_key / "gen" / dataset
    conds = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(
        conds,
        key=lambda c: (CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 99, c),
    )


def trace_from_completion(completion: str) -> list[dict]:
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
    return entries


def safety_labels(row: dict | None) -> dict:
    if not row:
        return {}
    flat = row.get("judge_flat") or {}
    return flat.get("labels") or {}


def reasoning_spans(row: dict | None) -> list[dict]:
    if not row:
        return []
    flat = row.get("judge_flat") or {}
    spans = flat.get("safety_reasoning_sentence_indexes") or []
    return [s for s in spans if isinstance(s, dict)]


def reasoning_complete(row: dict | None) -> bool:
    return isinstance((row or {}).get("judge_flat"), dict)


def span_key(span: dict) -> tuple:
    return (span.get("global_index"), span.get("section"), span.get("index"))


def entry_key(entry: dict) -> tuple:
    return (entry.get("global_index"), entry.get("section"), entry.get("index"))


def span_badge(span: dict) -> str:
    cat = str(span.get("category") or "other_safety_reasoning")
    label = CATEGORY_LABELS.get(cat, cat.replace("_", " "))
    conf = span.get("confidence")
    conf_txt = "" if conf is None else f" {float(conf):.2f}"
    return (
        f"<span class='reason-badge cat-{esc(cat)}'>"
        f"{esc(label)}{esc(conf_txt)}</span>"
    )


def metric_badges(labels: dict, judged: bool) -> str:
    if not judged:
        return "<span class='metric unknown'>standard judge: n/a</span>"
    items = [
        ("harmful_response", "harmful"),
        ("safe_rejection", "refusal"),
        ("reasoning_about_safety", "final safety reasoning"),
    ]
    bits = []
    for key, label in items:
        val = labels.get(key)
        cls = "yes" if val is True else "no" if val is False else "unknown"
        txt = "?" if val is None else "yes" if val else "no"
        bits.append(f"<span class='metric {cls}'>{esc(label)}: {txt}</span>")
    return "\n".join(bits)


def render_trace(completion: str, sr_row: dict | None) -> str:
    entries = (sr_row or {}).get("trace_segments")
    if not entries:
        entries = trace_from_completion(completion)
    spans = reasoning_spans(sr_row)
    by_key = {span_key(s): s for s in spans}
    by_section_index = {(s.get("section"), s.get("index")): s for s in spans}
    parts = ["<div class='trace'>"]
    for entry in entries:
        span = by_key.get(entry_key(entry))
        if span is None:
            span = by_section_index.get((entry.get("section"), entry.get("index")))
        section = entry.get("section") or "output"
        flagged = span is not None
        cls = f"trace-line {esc(section)}" + (" flagged" if flagged else "")
        idx = f"{entry.get('global_index')} | {section}:{entry.get('index')}"
        parts.append(f"<div class='{cls}'>")
        parts.append(f"<span class='idx'>{esc(idx)}</span>")
        if flagged:
            parts.append(span_badge(span))
        else:
            parts.append("<span></span>")
        parts.append(f"<span class='text'>{esc(entry.get('text'))}</span>")
        parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def render_completion_flow(completion: str, sr_row: dict | None) -> str:
    """Render the full completion as readable text, highlighting judged spans."""
    entries = (sr_row or {}).get("trace_segments")
    if not entries:
        entries = trace_from_completion(completion)
    spans = reasoning_spans(sr_row)
    by_key = {span_key(s): s for s in spans}
    by_section_index = {(s.get("section"), s.get("index")): s for s in spans}
    parts = ["<div class='completion-flow'>"]
    current_section = None
    for entry in entries:
        section = entry.get("section") or "output"
        if section != current_section:
            if current_section is not None:
                parts.append("</p>")
            current_section = section
            label = "CoT" if section == "cot" else "Output"
            parts.append(f"<p><span class='section-label'>{label}</span> ")
        span = by_key.get(entry_key(entry))
        if span is None:
            span = by_section_index.get((entry.get("section"), entry.get("index")))
        text = esc(entry.get("text"))
        idx = f"{entry.get('global_index')} | {section}:{entry.get('index')}"
        if span is not None:
            parts.append(
                f"<mark title='{esc(idx)}'>{text}"
                f"<span class='inline-badge'>{span_badge(span)}</span></mark> "
            )
        else:
            parts.append(f"<span title='{esc(idx)}'>{text}</span> ")
    if current_section is not None:
        parts.append("</p>")
    parts.append("</div>")
    return "\n".join(parts)


def compact_prompt(prompt: str, limit: int = 320) -> str:
    prompt = re.sub(r"\s+", " ", prompt or "").strip()
    if len(prompt) <= limit:
        return prompt
    return prompt[: limit - 1] + "..."


def query_ids_for_dataset(
    generation: dict[str, dict[str, dict]],
    reasoning: dict[str, dict[str, dict]],
    *,
    only_flagged: bool,
    require_baseline_reasoning: bool,
    max_queries: int | None,
) -> list[str]:
    if require_baseline_reasoning:
        ids = {
            rid for rid, row in reasoning.get("baseline", {}).items()
            if reasoning_complete(row)
        }
        if only_flagged:
            ids = {
                rid for rid in ids
                if reasoning_spans(reasoning.get("baseline", {}).get(rid))
            }
    elif only_flagged:
        ids = set()
        for rows in reasoning.values():
            for rid, row in rows.items():
                if reasoning_spans(row):
                    ids.add(rid)
    else:
        id_sets = [set(rows) for rows in generation.values() if rows]
        ids = set.intersection(*id_sets) if id_sets else set()
    def key(rid: str) -> tuple:
        return (rid.split("-000", 1)[0], rid)
    ordered = sorted(ids, key=key)
    if max_queries is not None:
        ordered = ordered[:max_queries]
    return ordered


def render_cell(cond: str, comp_row: dict | None, sr_row: dict | None, safety_row: dict | None) -> str:
    if not comp_row:
        return f"<div class='cell missing'><h3>{esc(cond)}</h3><p>missing generation</p></div>"
    spans = reasoning_spans(sr_row)
    flat = (sr_row or {}).get("judge_flat") or {}
    pos = flat.get("position") or {}
    extent = flat.get("extent") or {}
    judged_txt = "safety reasoning judged" if sr_row else "safety reasoning n/a"
    cls = "cell" + (" has-flags" if spans else "")
    parts = [f"<div class='{cls}'>"]
    parts.append(f"<h3>{esc(cond)}</h3>")
    parts.append("<div class='badges'>")
    parts.append(f"<span class='pill'>{judged_txt}</span>")
    parts.append(f"<span class='pill'>{len(spans)} highlighted</span>")
    if sr_row:
        parts.append(
            f"<span class='pill'>first {esc(pos.get('first_section'))}:"
            f"{esc(pos.get('first_index'))}</span>"
        )
        parts.append(
            f"<span class='pill'>extent {esc(extent.get('sentence_count'))} / "
            f"{pct(extent.get('fraction_of_sentences'))}</span>"
        )
    parts.append("</div>")
    parts.append("<div class='badges'>")
    parts.append(metric_badges(safety_labels(safety_row), judged=safety_row is not None))
    parts.append("</div>")
    if spans:
        parts.append("<div class='badges'>")
        for span in spans:
            parts.append(span_badge(span))
        parts.append("</div>")
    completion = comp_row.get("completion") or ""
    parts.append(render_completion_flow(completion, sr_row))
    parts.append("<details class='tracebox'><summary>Indexed sentence trace</summary>")
    parts.append(render_trace(completion, sr_row))
    parts.append("</details>")
    parts.append("<details class='rawbox'><summary>Raw full completion</summary>")
    parts.append(f"<pre>{esc(completion)}</pre>")
    parts.append("</details>")
    parts.append("</div>")
    return "\n".join(parts)


def render_dataset(
    *,
    dataset: str,
    conditions: list[str],
    generation: dict[str, dict[str, dict]],
    reasoning: dict[str, dict[str, dict]],
    safety: dict[str, dict[str, dict]],
    query_ids: list[str],
    only_judged_cells: bool,
) -> str:
    parts = [
        f"<section><h2>{esc(dataset.upper())}</h2>",
        f"<p>{len(query_ids)} grouped queries. Columns are interventions; scroll horizontally.</p>",
    ]
    for qn, rid in enumerate(query_ids, start=1):
        prompt = ""
        category = ""
        for cond in conditions:
            row = generation.get(cond, {}).get(rid)
            if row:
                prompt = row.get("prompt") or row.get("user_prompt") or ""
                category = row.get("category") or ""
                break
        visible_conditions = [
            cond for cond in conditions
            if generation.get(cond, {}).get(rid)
            and (
                not only_judged_cells
                or reasoning_complete(reasoning.get(cond, {}).get(rid))
            )
        ]
        flagged_count = sum(
            len(reasoning_spans(reasoning.get(cond, {}).get(rid)))
            for cond in visible_conditions
        )
        safety_rows = [safety.get(cond, {}).get(rid) for cond in visible_conditions]
        labels = [safety_labels(r) for r in safety_rows if r]
        harmful = sum(1 for lab in labels if lab.get("harmful_response") is True)
        refusal = sum(1 for lab in labels if lab.get("safe_rejection") is True)
        safety_reason = sum(1 for lab in labels if lab.get("reasoning_about_safety") is True)
        judged = len(labels)
        parts.append("<details class='query-row' open>")
        parts.append(
            f"<summary><span class='qnum'>{qn}</span> "
            f"<strong>{esc(rid)}</strong> "
            f"<span class='pill'>{esc(category or 'uncategorized')}</span> "
            f"<span class='pill hot'>{harmful}/{judged} harmful</span> "
            f"<span class='pill good'>{refusal}/{judged} refusal</span> "
            f"<span class='pill'>{safety_reason}/{judged} final safety reasoning</span> "
            f"<span class='pill'>{len(visible_conditions)} cells shown</span> "
            f"<span class='pill'>{flagged_count} highlighted spans</span>"
            "</summary>"
        )
        parts.append(f"<div class='prompt'><strong>Prompt</strong><br>{esc(prompt)}</div>")
        parts.append(
            f"<div class='wide-grid' style='grid-template-columns: repeat({len(visible_conditions)}, minmax(360px, 460px));'>"
        )
        for cond in visible_conditions:
            parts.append(render_cell(
                cond,
                generation.get(cond, {}).get(rid),
                reasoning.get(cond, {}).get(rid),
                safety.get(cond, {}).get(rid),
            ))
        parts.append("</div></details>")
    parts.append("</section>")
    return "\n".join(parts)


def render(
    *,
    model_key: str,
    judge_base: Path,
    datasets: list[str],
    max_queries: int | None,
    all_queries: bool,
    require_baseline_reasoning: bool,
    only_judged_cells: bool,
) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    css = """
body { margin: 0; background: #f6f8fa; color: #172026; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { padding: 24px; min-width: 1200px; }
h1 { margin: 0 0 8px; font-size: 26px; }
h2 { margin: 34px 0 10px; font-size: 20px; }
h3 { margin: 0 0 8px; font-size: 15px; }
p { color: #5f6b76; line-height: 1.45; }
code { background: #eef3f6; padding: 2px 4px; border-radius: 4px; }
.muted { color: #66737f; }
.query-row { background: #fff; border: 1px solid #dbe2e8; border-radius: 8px; margin: 16px 0; overflow: hidden; }
.query-row > summary { cursor: pointer; padding: 12px 14px; background: #eef3f6; border-bottom: 1px solid #dbe2e8; }
.qnum { display: inline-block; min-width: 28px; font-variant-numeric: tabular-nums; color: #66737f; }
.prompt-summary { color: #3f4e5a; }
.prompt { margin: 0; padding: 14px; background: #fff; border-bottom: 1px solid #dbe2e8; overflow-wrap: anywhere; }
.wide-grid { display: grid; gap: 0; overflow-x: auto; padding: 0; align-items: stretch; }
.cell { border-right: 1px solid #dbe2e8; background: #fff; min-height: 520px; max-height: 820px; overflow: auto; padding: 12px; }
.cell:last-child { border-right: 0; }
.cell.has-flags { border-color: #c98200; box-shadow: inset 0 0 0 2px #fff0ad; }
.cell.missing { color: #8a5d1b; background: #fff7e8; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
.pill, .metric, .reason-badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid #dbe2e8; background: #f7fafc; }
.pill.hot { border-color: #e0b68e; background: #fff7e8; color: #73420d; }
.pill.good { border-color: #9fcbb7; background: #ecf8f3; color: #14533b; }
.metric.yes { border-color: #9fcbb7; background: #ecf8f3; color: #14533b; }
.metric.no { border-color: #d4dce4; background: #f7f9fb; color: #66737f; }
.metric.unknown { border-color: #e1c095; background: #fff7e8; color: #774a14; }
.trace { border: 1px solid #dbe2e8; border-radius: 6px; overflow: hidden; background: #fff; margin-top: 8px; }
.trace-line { display: grid; grid-template-columns: 88px minmax(0, auto) 1fr; gap: 7px; align-items: start; padding: 7px 8px; border-bottom: 1px solid #eef2f5; font-size: 12px; line-height: 1.4; }
.trace-line:last-child { border-bottom: 0; }
.trace-line.cot { background: #fbfcfd; }
.trace-line.output { background: #fff; }
.trace-line.flagged { background: #fff2ad; border-left: 5px solid #c98200; padding-left: 3px; }
.idx { color: #66737f; font-variant-numeric: tabular-nums; font-size: 11px; }
.text { overflow-wrap: anywhere; }
.reason-badge { border-color: #c98200; background: #fff8d8; color: #5d3b00; white-space: nowrap; }
.cat-risk_acknowledgment { background: #ffe8df; border-color: #e1a58e; color: #703018; }
.cat-policy_boundary { background: #e8f0ff; border-color: #aebfe8; color: #243d76; }
.cat-intent_assessment { background: #f3eaff; border-color: #cab1ed; color: #513077; }
.cat-refusal_reasoning { background: #e8f8f0; border-color: #9fcbb7; color: #14533b; }
.cat-safer_alternative { background: #e6f6fb; border-color: #98cbd8; color: #155566; }
.completion-flow { margin-top: 10px; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; }
.completion-flow p { color: #172026; margin: 0 0 12px; }
.section-label { display: inline-block; font-weight: 700; color: #3f4e5a; margin-right: 6px; }
mark { background: #fff0a8; border-radius: 3px; padding: 1px 2px; }
.inline-badge { display: inline-block; margin-left: 4px; vertical-align: baseline; }
.inline-badge .reason-badge { font-size: 10px; padding: 0 5px; }
.tracebox, .rawbox { margin-top: 8px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #fbfcfd; border: 1px solid #e1e7ec; border-radius: 6px; padding: 8px; font-size: 11px; line-height: 1.4; }
"""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Qwen side-by-side safety reasoning traces</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>Qwen side-by-side safety reasoning traces</h1>",
        f"<p>Generated {esc(generated_at)}. Model key: <code>{esc(model_key)}</code>. "
        f"Safety-reasoning overlays from <code>{esc(judge_base)}</code>. "
        f"{'Showing all common generation queries.' if all_queries else 'Showing queries that have at least one judge-flagged safety-reasoning span in the current subset.'} "
        f"{'Requires baseline safety-reasoning judgment.' if require_baseline_reasoning else ''} "
        f"{'Cells without safety-reasoning judgment are hidden.' if only_judged_cells else ''}</p>",
    ]
    for dataset in datasets:
        conditions = condition_list(model_key, dataset)
        generation = load_generation(model_key, dataset, conditions)
        reasoning = load_reasoning(judge_base, dataset, conditions)
        safety = load_safety(judge_base, dataset, conditions)
        query_ids = query_ids_for_dataset(
            generation,
            reasoning,
            only_flagged=not all_queries,
            require_baseline_reasoning=require_baseline_reasoning,
            max_queries=max_queries,
        )
        parts.append(render_dataset(
            dataset=dataset,
            conditions=conditions,
            generation=generation,
            reasoning=reasoning,
            safety=safety,
            query_ids=query_ids,
            only_judged_cells=only_judged_cells,
        ))
    parts.append("</main></body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen3_8b")
    ap.add_argument("--judge-base", type=Path, default=DEFAULT_JUDGE_BASE)
    ap.add_argument("--datasets", nargs="+", default=["bt", "jbb"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all-queries", action="store_true")
    ap.add_argument("--require-baseline-reasoning", action="store_true",
                    help="Only include query ids whose baseline cell has a safety-reasoning judgment.")
    ap.add_argument("--only-judged-cells", action="store_true",
                    help="Hide intervention cells without safety-reasoning judgment for that query.")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()
    out = args.out
    if out is None:
        suffix = "all_queries" if args.all_queries else "flagged_queries"
        out = args.judge_base / f"side_by_side_safety_reasoning_{suffix}.html"
    out.write_text(
        render(
            model_key=args.model_key,
            judge_base=args.judge_base,
            datasets=args.datasets,
            max_queries=args.max_queries,
            all_queries=args.all_queries,
            require_baseline_reasoning=args.require_baseline_reasoning,
            only_judged_cells=args.only_judged_cells,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
