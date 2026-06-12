"""Direction A v4 — consolidated JBB report (4-condition comparison).

Reads per-condition ``summary.json`` produced by ``run_v4_jbb_judge.py``
under ``--in-base/<tag>/seed<seed>/`` and produces a single markdown
report comparing:

* Basic safety: harmful_rate, refusal_rate, reasoning_rate, adding_intent,
  changing_subject (from the merged 5-label safety judge).
* Pathway vector: 8 means + dominant_pathway histogram per condition.
* Monitorability: ASR_final, ASR_cot_pred, gap per condition.

This is the headline table the v4 paper framing rests on: same conditions,
matched at whatever ASR each lands at, but compared along pathway and
monitorability axes that the legacy 5-metric report cannot show.

Usage:
    python -m scripts.make_v4_jbb_report \\
        --in-base runs/direction_a/19-v4-jbb-qwen3 \\
        --out runs/direction_a/19-v4-jbb-qwen3/v4_jbb_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PATHWAY_VECTOR_FIELDS = (
    "recognition_rate", "refusal_initiated", "refusal_suppressed",
    "rationalisation_rate", "execution_rate", "operational_detail_rate",
    "first_execution_idx_norm", "recognition_to_execution",
)
DOMINANT_PATHWAYS = (
    "recognition_loss", "refusal_loss", "rationalised_compliance",
    "direct_execution", "sanitised_compliance", "none",
)
BASIC_FIELDS = (
    "harmful_rate", "refusal_rate", "reasoning_rate",
    "adding_intent_rate", "changing_subject_rate",
)


def _pct(x):
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


def _num(x, fmt=".3f"):
    if x is None:
        return "n/a"
    return format(float(x), fmt)


def _load_summaries(in_base: Path, seed: int) -> list[dict]:
    rows: list[dict] = []
    for sub in sorted(in_base.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / f"seed{seed}" / "summary.json"
        if not p.exists():
            continue
        with p.open() as f:
            s = json.load(f)
        s["_dir"] = str(sub.name)
        rows.append(s)
    return rows


def _cond_of(summary: dict) -> str:
    return summary.get("condition") or summary.get("_dir")


def _basic_row(s: dict) -> dict:
    basic = (s.get("per_condition_basic") or {})
    # aggregate_safety keys by condition; usually one per summary
    if not basic:
        return {}
    # take the first (and typically only) entry
    return next(iter(basic.values()))


def _pathway_mean(s: dict) -> dict:
    pw = s.get("per_condition_pathway") or {}
    if not pw:
        return {}
    return next(iter(pw.values())).get("mean") or {}


def _pathway_hist(s: dict) -> dict:
    pw = s.get("per_condition_pathway") or {}
    if not pw:
        return {}
    return next(iter(pw.values())).get("dominant_pathway_hist") or {}


def _monitor(s: dict) -> dict:
    mon = (s.get("monitorability") or {}).get("per_condition") or {}
    if not mon:
        return {}
    return next(iter(mon.values()))


GIBBERISH_CLASSES = ("clean", "mild gibberish", "word salad", "noise")


def _table_coherence(summaries: list[dict]) -> str:
    head = ("| condition | n | clean_rate | harmful_among_clean | "
            "repeat3_mean | compression_ratio_mean | empty_rate | "
            + " | ".join(GIBBERISH_CLASSES) + " |")
    sep = "|" + "---|" * (7 + len(GIBBERISH_CLASSES))
    lines = [head, sep]
    for s in summaries:
        c = s.get("coherence") or {}
        hist = c.get("gibberish_hist") or {}
        n = c.get("n", 0) or 0
        cells = [
            _cond_of(s),
            str(n),
            _pct(c.get("clean_rate")),
            _pct(c.get("harmful_among_clean_rate")),
            _num(c.get("repeat3_mean")),
            _num(c.get("compression_ratio_mean")),
            _pct(c.get("empty_rate")),
        ]
        for k in GIBBERISH_CLASSES:
            v = hist.get(k, 0)
            cells.append(f"{v} ({100*v/max(1,n):.0f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_basic(summaries: list[dict]) -> str:
    head = "| condition | n | " + " | ".join(BASIC_FIELDS) + " |"
    sep = "|" + "---|" * (len(BASIC_FIELDS) + 2)
    lines = [head, sep]
    for s in summaries:
        b = _basic_row(s)
        cells = [_cond_of(s), str(b.get("n", "n/a"))]
        for f in BASIC_FIELDS:
            cells.append(_pct(b.get(f)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_pathway_mean(summaries: list[dict]) -> str:
    head = "| condition | n | " + " | ".join(PATHWAY_VECTOR_FIELDS) + " |"
    sep = "|" + "---|" * (len(PATHWAY_VECTOR_FIELDS) + 2)
    lines = [head, sep]
    for s in summaries:
        pw = (s.get("per_condition_pathway") or {})
        first = next(iter(pw.values()), {})
        means = first.get("mean") or {}
        cells = [_cond_of(s), str(first.get("n", "n/a"))]
        for f in PATHWAY_VECTOR_FIELDS:
            cells.append(_num(means.get(f)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_pathway_hist(summaries: list[dict]) -> str:
    head = "| condition | " + " | ".join(DOMINANT_PATHWAYS) + " |"
    sep = "|" + "---|" * (len(DOMINANT_PATHWAYS) + 1)
    lines = [head, sep]
    for s in summaries:
        hist = _pathway_hist(s)
        total = sum(hist.values()) or 1
        cells = [_cond_of(s)]
        for d in DOMINANT_PATHWAYS:
            c = hist.get(d, 0)
            cells.append(f"{c} ({100*c/total:.0f}%)")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_monitor(summaries: list[dict]) -> str:
    head = "| condition | n | ASR_final | ASR_cot_pred | gap (final-cot) |"
    sep = "|---|---|---|---|---|"
    lines = [head, sep]
    for s in summaries:
        m = _monitor(s)
        cells = [
            _cond_of(s),
            str(m.get("n", "n/a")),
            _pct(m.get("asr_final")),
            _pct(m.get("asr_cot_pred")),
            _num(m.get("gap"), fmt="+.3f"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _headline_asr(s: dict) -> float | None:
    """Headline ASR = harmful_among_clean if coherence ran, else harmful_rate."""
    coh = s.get("coherence") or {}
    if coh.get("harmful_among_clean_rate") is not None:
        return coh.get("harmful_among_clean_rate")
    return _basic_row(s).get("harmful_rate")


def _table_headline(summaries: list[dict]) -> str:
    head = ("| condition | n | clean_rate | ASR_clean (headline) | "
            "raw_harmful_rate | refusal_rate |")
    sep = "|---|---|---|---|---|---|"
    lines = [head, sep]
    for s in summaries:
        coh = s.get("coherence") or {}
        b = _basic_row(s)
        cells = [
            _cond_of(s),
            str(coh.get("n", b.get("n", "n/a"))),
            _pct(coh.get("clean_rate")),
            _pct(_headline_asr(s)),
            _pct(b.get("harmful_rate")),
            _pct(b.get("refusal_rate")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_iso_asr(summaries: list[dict], anchor: str | None) -> str:
    """Iso-ASR comparison.

    Picks the headline ASR (harmful_among_clean) of the anchor condition
    as the target, and reports |delta| for every other condition. Useful
    to read off which intervention is closest to the anchor's safety
    impact while preserving coherence.
    """
    if not anchor:
        # default: pick the steering condition if present, else max ASR_clean
        for s in summaries:
            if "steering" in _cond_of(s):
                anchor = _cond_of(s); break
    if not anchor:
        anchor = max(summaries,
                     key=lambda s: _headline_asr(s) or 0.0)
        anchor = _cond_of(anchor)
    target = next((_headline_asr(s) for s in summaries
                   if _cond_of(s) == anchor), None)
    head = ("| condition | clean_rate | ASR_clean | "
            f"|Δ vs {anchor}| | iso-ASR? |")
    sep = "|---|---|---|---|---|"
    lines = [head, sep]
    for s in summaries:
        coh = s.get("coherence") or {}
        asr = _headline_asr(s)
        if asr is None or target is None:
            delta = None; iso = "n/a"
        else:
            delta = abs(float(asr) - float(target))
            iso = "✓" if delta <= 0.05 else ""  # within 5pp
        cells = [
            _cond_of(s),
            _pct(coh.get("clean_rate")),
            _pct(asr),
            (_pct(delta) if delta is not None else "n/a"),
            iso,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section_interpretation() -> str:
    return (
        "**How to read the tables**\n\n"
        "- *Coherence* is the gate. `clean_rate` is the fraction of "
        "completions the gibberish detector labels `clean`; "
        "`harmful_among_clean` is the harmful_response rate restricted to "
        "those rows. If a condition's harmful_rate drops only because its "
        "`clean_rate` collapsed, the intervention is breaking the model, "
        "not removing the circuit. Compare `harmful_among_clean` across "
        "conditions to control for that.\n"
        "- *Basic safety* is the legacy 5-metric scoreboard (post-merge of the "
        "single-label safety judge). Treat this as a sanity check, not the "
        "headline.\n"
        "- *Pathway vector* and *Dominant pathway* describe **how** each "
        "condition's traces fail, not whether they fail. A condition with "
        "modest ASR but high `refusal_suppressed` / `rationalisation_rate` "
        "is failing *differently* from one with high ASR but high "
        "`operational_detail_rate`.\n"
        "- *Monitorability gap = ASR_final - ASR_cot_pred*. Positive means "
        "the CoT looks safer than the final answer — bad for CoT-monitoring. "
        "Near-zero means a CoT monitor would catch the attack. Compare the "
        "**gap across conditions**, not the absolute ASR.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Direction A v4 — JBB report")
    ap.add_argument("--iso-anchor", default=None,
                    help="Condition name to anchor iso-ASR comparison "
                         "(default: first condition containing 'steering').")
    args = ap.parse_args()

    summaries = _load_summaries(Path(args.in_base), args.seed)
    if not summaries:
        print(f"no summary.json files found under {args.in_base}")
        return 2

    judge = summaries[0].get("judge_model", "<unknown>")
    md = []
    md.append(f"# {args.title}")
    md.append("")
    md.append(f"- Judge: `{judge}` (NF4)")
    md.append(f"- Conditions: {len(summaries)}")
    md.append("- Headline ASR = `harmful_among_clean` "
              "(judge skipped on gibberish rows when `gate_safety_by_coherence=true`).")
    md.append(f"- Per-label single-shot judge: 5 safety + 12 pathway + 1 cot-only")
    md.append("")
    md.append("## 0. Headline (ASR_clean + coherence)")
    md.append("")
    md.append(_table_headline(summaries))
    md.append("")
    md.append("## 1. Iso-ASR comparison")
    md.append("")
    md.append(_table_iso_asr(summaries, args.iso_anchor))
    md.append("")
    md.append("## 2. Coherence gate (gibberish detector + diagnostics)")
    md.append("")
    md.append(_table_coherence(summaries))
    md.append("")
    md.append("## 3. Basic safety (5-metric sanity check)")
    md.append("")
    md.append(_table_basic(summaries))
    md.append("")
    md.append("## 4. Pathway vector (means)")
    md.append("")
    md.append(_table_pathway_mean(summaries))
    md.append("")
    md.append("## 5. Dominant pathway distribution")
    md.append("")
    md.append(_table_pathway_hist(summaries))
    md.append("")
    md.append("## 6. CoT-monitorability gap")
    md.append("")
    md.append(_table_monitor(summaries))
    md.append("")
    md.append("## 7. Interpretation")
    md.append("")
    md.append(_section_interpretation())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
