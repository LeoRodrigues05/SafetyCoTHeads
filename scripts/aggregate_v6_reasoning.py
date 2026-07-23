#!/usr/bin/env python3
"""Finding-2 aggregation from v6 pathway-prefix + indexed safety-reasoning outputs (P0.12).

Consumes ONLY v6 judge artifacts:
  * ``judge_pathway.jsonl``               (cumulative-prefix pathway labels)
  * ``judge_safety_reasoning_trace.jsonl``(indexed-sentence SR judgments)

For each explicit-trace cell it produces:
  * per-completion pathway vectors + a cell-level pathway summary / dominant
    pathway histogram (via the existing pathway_taxonomy aggregation);
  * safety-reasoning engagement rate, mean sentence count / coverage, and first
    safety-reasoning position, with denominators and parse rates.

Writes runs/direction_a_v6/reports/reasoning_metrics.{json,csv}. Nothing here
reads v5 pathway/SR files, so once v6 is declared final the paper's Finding 2
derives from v6 only.
"""

from __future__ import annotations

import argparse
import csv
from statistics import mean

import v6_common as C
from safety_cot_heads.direction_a.pathway_taxonomy import pathway_vector, summarise_pathways


def _v6(cell: C.Cell, name: str):
    return C.V6_ROOT / "judge" / cell.model / cell.dataset / cell.condition / cell.seed / name


def _pathway_cell(cell: C.Cell) -> dict | None:
    rows = C.read_jsonl(_v6(cell, "judge_pathway.jsonl"))
    if not rows:
        return None
    vectors = pathway_vector(rows)               # one per parent completion
    summary = summarise_pathways(vectors)
    return {"n_prefix_rows": len(rows), "n_completions": len(vectors), "summary": summary}


def _sr_cell(cell: C.Cell) -> dict | None:
    rows = C.read_jsonl(_v6(cell, "judge_safety_reasoning_trace.jsonl"))
    if not rows:
        return None
    n = 0
    n_has = 0
    n_failed = 0
    counts = []
    first_norm = []
    for r in rows:
        flat = r.get("judge_flat") or {}
        if r.get("judge_parse_status") == "failed" or not flat:
            n_failed += 1
            continue
        n += 1
        has = bool(flat.get("has_safety_reasoning"))
        n_has += int(has)
        ext = (flat.get("extent") or {})
        sc = ext.get("sentence_count")
        if isinstance(sc, (int, float)):
            counts.append(float(sc))
        pos = (flat.get("position") or {})
        fg = pos.get("first_global_index")
        nseg = r.get("n_trace_segments")
        if isinstance(fg, (int, float)) and isinstance(nseg, (int, float)) and nseg and nseg > 1:
            first_norm.append(float(fg) / (float(nseg) - 1))
    return {
        "n_judged": n, "n_failed_parse": n_failed,
        "safety_reasoning_rate": (n_has / n) if n else None,
        "mean_sentence_count": (mean(counts) if counts else None),
        "mean_first_position_norm": (mean(first_norm) if first_norm else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=["jbb", "bt"])
    ap.add_argument("--min-explicit-frac", type=float, default=0.5)
    args = ap.parse_args()

    scope = C.load_paper_scope()
    declared = set(scope["explicit_trace_models"])
    cells = C.discover_cells(args.models, args.datasets)

    # evidence-based explicit scope (mirror P0.8)
    from collections import defaultdict
    tot = defaultdict(int); exp = defaultdict(int)
    for c in cells:
        p = c.v6_parsed_dir() / "parsed_completions.jsonl"
        if not p.exists():
            continue
        for r in C.iter_jsonl(p):
            tot[c.model] += 1
            exp[c.model] += int(bool(r.get("has_explicit_trace")))
    explicit_models = {m for m in declared
                       if tot[m] and exp[m] / tot[m] >= args.min_explicit_frac}

    out_rows = []
    for c in cells:
        if c.model not in explicit_models:
            continue
        pw = _pathway_cell(c)
        sr = _sr_cell(c)
        if pw is None and sr is None:
            continue
        summ = (pw or {}).get("summary", {}) or {}
        out_rows.append({
            "model": c.model, "dataset": c.dataset, "condition": c.condition,
            "is_primary": c.model in set(scope["primary"]),
            "n_pathway_prefix_rows": (pw or {}).get("n_prefix_rows", 0),
            "n_pathway_completions": (pw or {}).get("n_completions", 0),
            "dominant_pathway_hist": summ.get("dominant_pathway_hist", {}),
            "pathway_summary": summ,
            "sr_n_judged": (sr or {}).get("n_judged", 0),
            "sr_n_failed_parse": (sr or {}).get("n_failed_parse", 0),
            "safety_reasoning_rate": (sr or {}).get("safety_reasoning_rate"),
            "sr_mean_sentence_count": (sr or {}).get("mean_sentence_count"),
            "sr_mean_first_position_norm": (sr or {}).get("mean_first_position_norm"),
        })

    out = C.V6_ROOT / "reports"
    C.write_json(out / "reasoning_metrics.json",
                 {"generated_at_utc": C.utcnow_iso(),
                  "explicit_models": sorted(explicit_models),
                  "source": "v6-only (judge_pathway + judge_safety_reasoning_trace)",
                  "rows": out_rows})
    cols = ["model", "dataset", "condition", "is_primary",
            "n_pathway_prefix_rows", "n_pathway_completions",
            "sr_n_judged", "sr_n_failed_parse", "safety_reasoning_rate",
            "sr_mean_sentence_count", "sr_mean_first_position_norm"]
    with open(out / "reasoning_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in out_rows:
            w.writerow([r.get(k) if not isinstance(r.get(k), float) else f"{r[k]:.4f}"
                        for k in cols])
    print(f"[reasoning] {len(out_rows)} explicit cells; models={sorted(explicit_models)}")
    print(f"[reasoning] wrote {out}/reasoning_metrics.json|csv")


if __name__ == "__main__":
    main()
