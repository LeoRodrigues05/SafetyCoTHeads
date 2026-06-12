"""Sampled side-by-side comparison report across ablation conditions.

Joins multiple (completions, judged) pairs on prompt ``id``, samples a
subset preferring prompts where many conditions produced a harmful
response, and writes an HTML + CSV report with a top-level summary and
per-prompt side-by-side cards.

Example (JBB, 25 prompts):

    python -m scripts.make_side_by_side_report \\
        --title "JBB side-by-side (Llama-3.1-8B)" \\
        --out-dir runs/analysis/v4_jbb_side_by_side \\
        --sample-size 25 \\
        --condition baseline runs/.../completions_baseline.jsonl runs/.../judged_baseline.jsonl \\
        --condition ships_top5 ...

Example (BT, ~100 prompts, balanced across categories):

    python -m scripts.make_side_by_side_report \\
        --title "BT side-by-side (Llama-3.1-8B)" \\
        --out-dir runs/analysis/v4_bt_side_by_side \\
        --sample-size 100 \\
        --stratify-by category \\
        --condition baseline ... --condition ships_top5 ...
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFETY_LABEL_KEYS = [
    "harmful_response",
    "safe_rejection",
    "reasoning_about_safety",
    "adding_intention",
    "changing_subject",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def word_count(text: str | None) -> int:
    return len((text or "").split())


def safe_div(num: float, den: float) -> float | None:
    return (num / den) if den else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def safety_labels(judge_row: dict[str, Any]) -> dict[str, Any]:
    flat = judge_row.get("judge_flat") or {}
    labels = flat.get("labels") if isinstance(flat.get("labels"), dict) else {}
    confs = flat.get("confidences") if isinstance(flat.get("confidences"), dict) else {}
    return {
        "labels": {k: labels.get(k) for k in SAFETY_LABEL_KEYS},
        "confidences": {k: confs.get(k) for k in SAFETY_LABEL_KEYS},
        "parse_status": judge_row.get("judge_parse_status"),
    }


def condition_summary(label: str, comp_rows: list[dict], judged_rows: list[dict]) -> dict[str, Any]:
    n = len(comp_rows)
    parsed = 0
    label_true = Counter()
    for r in judged_rows:
        if r.get("judge_parse_status") in ("ok", "recovered"):
            parsed += 1
        flat = r.get("judge_flat") or {}
        labs = flat.get("labels") if isinstance(flat.get("labels"), dict) else {}
        for k in SAFETY_LABEL_KEYS:
            if labs.get(k) is True:
                label_true[k] += 1
    lengths = [len(r.get("completion") or "") for r in comp_rows]
    words = [word_count(r.get("completion")) for r in comp_rows]
    return {
        "label": label,
        "n_completions": n,
        "n_judged": len(judged_rows),
        "n_parsed": parsed,
        "harmful_rate": safe_div(label_true["harmful_response"], len(judged_rows)),
        "safe_rejection_rate": safe_div(label_true["safe_rejection"], len(judged_rows)),
        "reasoning_about_safety_rate": safe_div(label_true["reasoning_about_safety"], len(judged_rows)),
        "adding_intention_rate": safe_div(label_true["adding_intention"], len(judged_rows)),
        "changing_subject_rate": safe_div(label_true["changing_subject"], len(judged_rows)),
        "completion_chars_median": statistics.median(lengths) if lengths else None,
        "completion_words_median": statistics.median(words) if words else None,
        "empty_completion_count": sum(1 for r in comp_rows if not (r.get("completion") or "").strip()),
    }


def build_joined_rows(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comp_by_label = {c["label"]: {r["id"]: r for r in c["completion_rows"]} for c in conditions}
    judge_by_label = {c["label"]: {r["id"]: r for r in c["judge_rows"]} for c in conditions}
    common_ids = set.intersection(*(set(v) for v in comp_by_label.values()))
    common_ids &= set.intersection(*(set(v) for v in judge_by_label.values()))
    rows: list[dict[str, Any]] = []
    primary = conditions[0]["label"]
    for row_id in sorted(common_ids):
        base = comp_by_label[primary][row_id]
        row: dict[str, Any] = {
            "id": row_id,
            "dataset": base.get("dataset"),
            "category": base.get("category"),
            "prompt": base.get("prompt", ""),
            "conditions": {},
        }
        for c in conditions:
            label = c["label"]
            comp = comp_by_label[label][row_id]
            judge = judge_by_label[label][row_id]
            sl = safety_labels(judge)
            row["conditions"][label] = {
                "completion": comp.get("completion", ""),
                "chars": len(comp.get("completion") or ""),
                "words": word_count(comp.get("completion")),
                "judge_parse_status": sl["parse_status"],
                **{k: sl["labels"][k] for k in SAFETY_LABEL_KEYS},
                "confidences": sl["confidences"],
            }
        rows.append(row)
    return rows


def annotate_unsafe_count(rows: list[dict[str, Any]], labels: list[str]) -> None:
    for row in rows:
        n_harm = sum(1 for label in labels if row["conditions"][label].get("harmful_response") is True)
        n_refuse = sum(1 for label in labels if row["conditions"][label].get("safe_rejection") is True)
        row["n_harmful_conditions"] = n_harm
        row["n_refusal_conditions"] = n_refuse


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    stratify_by: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not stratify_by:
        # Prefer rows with the largest number of harmful conditions, tie-break
        # by random ordering for reproducibility.
        scored = sorted(
            rows,
            key=lambda r: (-r["n_harmful_conditions"], rng.random()),
        )
        return scored[:sample_size]

    # Stratified: roughly equal counts per stratum, prefer-unsafe within each.
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get(stratify_by) or "uncategorized"].append(r)
    n_groups = len(groups)
    per_group = max(1, sample_size // n_groups)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = sorted(
            groups[key],
            key=lambda r: (-r["n_harmful_conditions"], rng.random()),
        )
        selected.extend(bucket[:per_group])
    # Fill remainder with the next-best leftovers across all groups.
    if len(selected) < sample_size:
        remaining = [r for r in rows if r not in selected]
        remaining.sort(key=lambda r: (-r["n_harmful_conditions"], rng.random()))
        selected.extend(remaining[: sample_size - len(selected)])
    return selected[:sample_size]


def label_chip(key: str, value: Any, conf: Any) -> str:
    if not isinstance(value, bool):
        return ""
    if key == "harmful_response":
        cls = "bad" if value else "neutral"
    elif key == "safe_rejection":
        cls = "good" if value else "neutral"
    elif value:
        cls = "warn"
    else:
        cls = "neutral"
    label_text = {
        "harmful_response": "harmful",
        "safe_rejection": "refusal",
        "reasoning_about_safety": "safety-reasoning",
        "adding_intention": "adds-intent",
        "changing_subject": "subject-change",
    }[key]
    conf_text = ""
    if isinstance(conf, (int, float)):
        conf_text = f" {conf:.2f}"
    return (
        f'<span class="chip {cls}">{html.escape(label_text)}'
        f': {str(value).lower()}{html.escape(conf_text)}</span>'
    )


def judge_chips_html(entry: dict[str, Any]) -> str:
    confs = entry.get("confidences") or {}
    return "".join(label_chip(k, entry.get(k), confs.get(k)) for k in SAFETY_LABEL_KEYS)


def summary_table_html(labels: list[str], summary: dict[str, Any], *, sample_summary: dict[str, Any]) -> str:
    headers = [
        "Condition", "N (full)", "Parsed", "Harmful", "Refusal", "Safety-reasoning",
        "Adds-intent", "Subject-change", "Median chars", "Empty",
        "Harmful (in sample)", "Refusal (in sample)",
    ]
    body = []
    for label in labels:
        s = summary[label]
        ss = sample_summary[label]
        body.append([
            label,
            s["n_completions"],
            s["n_parsed"],
            pct(s["harmful_rate"]),
            pct(s["safe_rejection_rate"]),
            pct(s["reasoning_about_safety_rate"]),
            pct(s["adding_intention_rate"]),
            pct(s["changing_subject_rate"]),
            fmt(s["completion_chars_median"]),
            s["empty_completion_count"],
            pct(ss["harmful_rate"]),
            pct(ss["safe_rejection_rate"]),
        ])
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
        for r in body
    )
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def category_balance_table_html(rows: list[dict[str, Any]]) -> str:
    cats = Counter(r.get("category") or "uncategorized" for r in rows)
    head = "<th>Category</th><th>Sampled prompts</th>"
    body_rows = "".join(
        f"<tr><td>{html.escape(c)}</td><td>{n}</td></tr>"
        for c, n in sorted(cats.items())
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>"


def write_html_report(
    path: Path,
    *,
    title: str,
    labels: list[str],
    sample_rows: list[dict[str, Any]],
    full_summary: dict[str, Any],
    sample_summary: dict[str, Any],
    generated_at: str,
    notes: list[str],
) -> None:
    style = """
body { margin: 24px; color: #1d252c; background: #f7f8fa; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
h1 { font-size: 26px; margin: 0 0 6px; }
h2 { font-size: 18px; margin-top: 28px; }
code { background: #eef1f4; padding: 2px 4px; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee4; margin: 12px 0 18px; }
th, td { border-bottom: 1px solid #e7ebef; padding: 8px; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #eef2f5; font-weight: 650; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 16px 0; }
input, select { font: inherit; padding: 8px 10px; border: 1px solid #c7ced6; border-radius: 4px; background: white; }
input { min-width: 360px; }
.prompt-row { background: white; border: 1px solid #d8dee4; border-radius: 6px; margin: 16px 0; overflow: hidden; }
.meta { background: #edf2f5; border-bottom: 1px solid #d8dee4; padding: 10px 12px; font-size: 13px; }
.prompt { padding: 12px; border-bottom: 1px solid #e7ebef; white-space: pre-wrap; }
.grid { display: grid; grid-template-columns: repeat(var(--cols), minmax(280px, 1fr)); overflow-x: auto; }
.cell { min-width: 280px; border-right: 1px solid #e7ebef; padding: 12px; }
.cell:last-child { border-right: 0; }
.cell h3 { margin: 0 0 8px; font-size: 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 8px; }
.chip { display: inline-block; border: 1px solid #b7c0c9; background: #fff; border-radius: 999px; padding: 2px 7px; font-size: 12px; }
.chip.good { border-color: #8fbf9d; background: #ecf7ee; }
.chip.warn { border-color: #d6b96c; background: #fff7df; }
.chip.bad { border-color: #d78a8a; background: #fff0f0; }
.chip.neutral { border-color: #c9d0d7; background: #f6f8fa; }
.response { white-space: pre-wrap; line-height: 1.38; font-size: 13px; max-height: 480px; overflow: auto; }
.notes { background: #fff8e6; border: 1px solid #e2d28a; border-radius: 6px; padding: 12px; }
@media (max-width: 900px) { input { min-width: 0; width: 100%; } .grid { grid-template-columns: 1fr; } .cell { border-right: 0; border-bottom: 1px solid #e7ebef; } }
"""
    categories = sorted({r.get("category") or "uncategorized" for r in sample_rows})
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        f"<style>{style}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>Generated {html.escape(generated_at)}. Each row pairs the same prompt id across all conditions. "
        f"Sampled <b>{len(sample_rows)}</b> prompts from the joined common-id pool, preferring prompts where "
        f"more conditions produced <code>harmful_response=true</code>.</p>",
    ]
    if notes:
        parts.append('<div class="notes"><b>Notes on this report</b><ul>')
        parts.extend(f"<li>{html.escape(n)}</li>" for n in notes)
        parts.append("</ul></div>")
    parts.extend([
        "<h2>Full-dataset and sampled summary (judge = Qwen2.5-32B-Instruct, 5-label CoT-safety schema)</h2>",
        summary_table_html(labels, full_summary, sample_summary=sample_summary),
        "<h2>Sample breakdown by category</h2>",
        category_balance_table_html(sample_rows),
        "<h2>Side-by-side responses</h2>",
        '<div class="controls"><input id="q" placeholder="Search prompt, id, or response text">',
        '<select id="category"><option value="">All categories</option>',
    ])
    for c in categories:
        parts.append(f'<option value="{html.escape(c, quote=True)}">{html.escape(c)}</option>')
    parts.append("</select></div>")
    parts.append(f'<div id="rows" style="--cols:{len(labels)}">')
    for row in sample_rows:
        category = row.get("category") or "uncategorized"
        search_blob = " ".join(
            [row["id"], category, row.get("prompt") or ""]
            + [row["conditions"][lbl].get("completion", "") for lbl in labels]
        ).lower()
        n_harm = row["n_harmful_conditions"]
        n_refuse = row["n_refusal_conditions"]
        flag = f"{n_harm}/{len(labels)} harmful · {n_refuse}/{len(labels)} refusal"
        parts.append(
            f'<section class="prompt-row" data-category="{html.escape(category, quote=True)}" '
            f'data-search="{html.escape(search_blob, quote=True)}">'
        )
        parts.append(
            f'<div class="meta"><b>{html.escape(row["id"])}</b> '
            f'<span class="chip neutral">{html.escape(category)}</span> '
            f'<span class="chip warn">{html.escape(flag)}</span></div>'
        )
        parts.append(f'<div class="prompt"><b>Prompt</b>\n{html.escape(row.get("prompt") or "")}</div>')
        parts.append('<div class="grid">')
        for label in labels:
            entry = row["conditions"][label]
            parts.append('<div class="cell">')
            parts.append(f"<h3>{html.escape(label)}</h3>")
            parts.append(f'<div class="chips">{judge_chips_html(entry)}</div>')
            parts.append(f'<div class="response">{html.escape(entry.get("completion", ""))}</div>')
            parts.append("</div>")
        parts.append("</div></section>")
    parts.append("</div>")
    parts.append("""
<script>
const q = document.getElementById('q');
const category = document.getElementById('category');
const rows = [...document.querySelectorAll('.prompt-row')];
function applyFilters() {
  const needle = q.value.trim().toLowerCase();
  const cat = category.value;
  rows.forEach(r => {
    const ok = r.dataset.search.includes(needle) && (!cat || r.dataset.category === cat);
    r.style.display = ok ? '' : 'none';
  });
}
q.addEventListener('input', applyFilters);
category.addEventListener('change', applyFilters);
</script>
</body></html>""")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, sample_rows: list[dict[str, Any]], labels: list[str]) -> None:
    fieldnames = ["id", "category", "prompt", "n_harmful_conditions", "n_refusal_conditions"]
    for label in labels:
        for k in SAFETY_LABEL_KEYS:
            fieldnames.append(f"{label}__{k}")
        fieldnames.append(f"{label}__chars")
        fieldnames.append(f"{label}__completion")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            out = {
                "id": row["id"],
                "category": row.get("category") or "",
                "prompt": row.get("prompt") or "",
                "n_harmful_conditions": row["n_harmful_conditions"],
                "n_refusal_conditions": row["n_refusal_conditions"],
            }
            for label in labels:
                entry = row["conditions"][label]
                for k in SAFETY_LABEL_KEYS:
                    out[f"{label}__{k}"] = entry.get(k)
                out[f"{label}__chars"] = entry.get("chars")
                out[f"{label}__completion"] = entry.get("completion", "")
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--title", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--condition", action="append", nargs=3, required=True,
        metavar=("LABEL", "COMPLETIONS", "JUDGED"),
        help="Triple: condition label, completions jsonl, judged jsonl. Repeat per condition.",
    )
    p.add_argument("--sample-size", type=int, default=25)
    p.add_argument("--stratify-by", default=None, help="Field name (e.g. 'category') for balanced sampling.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--note", action="append", default=[], help="Note text shown at top of report. Repeatable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conditions = []
    for label, comp_path, judge_path in args.condition:
        comp_rows = read_jsonl(Path(comp_path))
        judge_rows = read_jsonl(Path(judge_path))
        # Restrict completions to ids that exist in the judge file (handles
        # partially-judged conditions cleanly).
        judge_ids = {r["id"] for r in judge_rows}
        comp_rows = [r for r in comp_rows if r["id"] in judge_ids]
        conditions.append({
            "label": label,
            "completion_rows": comp_rows,
            "judge_rows": judge_rows,
        })

    labels = [c["label"] for c in conditions]

    full_summary = {c["label"]: condition_summary(c["label"], c["completion_rows"], c["judge_rows"]) for c in conditions}
    joined = build_joined_rows(conditions)
    annotate_unsafe_count(joined, labels)

    sample = sample_rows(
        joined,
        sample_size=args.sample_size,
        stratify_by=args.stratify_by,
        seed=args.seed,
    )

    # Sort sample rows: most-harmful first, then by category.
    sample.sort(key=lambda r: (-r["n_harmful_conditions"], r.get("category") or "", r["id"]))

    # Sampled-only summary (judged on the sample subset).
    sample_summary = {}
    for label in labels:
        comp_subset = [{"completion": r["conditions"][label]["completion"]} for r in sample]
        judge_subset = []
        for r in sample:
            entry = r["conditions"][label]
            judge_subset.append({
                "judge_parse_status": entry.get("judge_parse_status"),
                "judge_flat": {
                    "labels": {k: entry.get(k) for k in SAFETY_LABEL_KEYS},
                    "confidences": entry.get("confidences", {}),
                },
            })
        sample_summary[label] = condition_summary(label, comp_subset, judge_subset)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    html_path = args.out_dir / "side_by_side.html"
    csv_path = args.out_dir / "side_by_side.csv"
    json_path = args.out_dir / "side_by_side.summary.json"

    write_html_report(
        html_path,
        title=args.title,
        labels=labels,
        sample_rows=sample,
        full_summary=full_summary,
        sample_summary=sample_summary,
        generated_at=generated_at,
        notes=args.note,
    )
    write_csv(csv_path, sample, labels)

    json_path.write_text(json.dumps({
        "title": args.title,
        "generated_at": generated_at,
        "labels": labels,
        "sample_size_requested": args.sample_size,
        "sample_size_actual": len(sample),
        "stratify_by": args.stratify_by,
        "seed": args.seed,
        "n_common_ids": len(joined),
        "full_summary": full_summary,
        "sample_summary": sample_summary,
        "category_distribution_in_sample": dict(Counter(r.get("category") or "uncategorized" for r in sample)),
        "notes": args.note,
    }, indent=2), encoding="utf-8")

    print(f"wrote: {html_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
