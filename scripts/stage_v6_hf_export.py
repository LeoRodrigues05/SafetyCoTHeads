#!/usr/bin/env python3
"""Stage a CORRECTED Hugging Face dataset export locally — DOES NOT PUBLISH.

Assembles the v6 corrected metrics + parsed answer/trace samples + a dataset
card into runs/direction_a_v6/hf_export_staged/. No network calls, no
huggingface_hub push. Publishing requires explicit human approval and a separate
invocation of scripts/upload_results_to_hf.py against this staged directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import v6_common as C


def main():
    stage = C.V6_ROOT / "hf_export_staged"
    stage.mkdir(parents=True, exist_ok=True)

    # 1. corrected per-cell metrics
    metrics = C.V6_ROOT / "reports" / "cell_metrics.json"
    if metrics.exists():
        shutil.copy(metrics, stage / "cell_metrics.json")
    for name in ("v5_vs_v6_metrics.csv", "explicit_trace_metrics.csv",
                 "prose_prefix_sensitivity.csv", "validation_summary.csv"):
        p = C.V6_ROOT / "reports" / name
        if p.exists():
            shutil.copy(p, stage / name)

    # 2. a small parsed-completions sample (answer/trace split) for inspection
    sample = []
    for cell in C.discover_cells(["olmo3_7b_think"], ["jbb"])[:2]:
        for r in C.read_jsonl(cell.v6_parsed_dir() / "parsed_completions.jsonl")[:20]:
            sample.append({k: r.get(k) for k in
                           ("id", "model", "dataset", "condition", "trace_kind",
                            "parse_status", "has_explicit_trace", "answer_text",
                            "trace_text")})
    C.write_jsonl(stage / "parsed_sample.jsonl", sample)

    # 3. dataset card (staged, not pushed)
    card = f"""---
license: cc-by-4.0
tags: [safety, chain-of-thought, monitorability, interventions]
---

# Direction A v6 (corrected) — STAGED, NOT PUBLISHED

Corrected evaluation of white-box safety interventions. This export is **staged
locally** and must not be pushed without explicit human approval.

Corrections vs v5:
- answer-level safety/coherence judged on the parsed final answer (not the CoT);
- trace-level judges see only the reasoning trace;
- monitorability computed from paired per-prompt outcomes (covert-failure U,
  over-warning O, agreement, trace FNR), replacing the marginal signed gap;
- explicit `<think>` traces separated from heuristic prose prefixes.

Files: cell_metrics.json, v5_vs_v6_metrics.csv, explicit_trace_metrics.csv,
prose_prefix_sensitivity.csv, validation_summary.csv, parsed_sample.jsonl.

Provenance: see runs/direction_a_v6/manifest/run_manifest.json.
"""
    (stage / "README.md").write_text(card)

    C.write_json(stage / "STAGING_INFO.json", {
        "staged_at_utc": C.utcnow_iso(),
        "published": False,
        "publish_requires": "explicit human approval + scripts/upload_results_to_hf.py",
        "note": "No network calls were made. This directory is a local staging area only.",
    })
    print(f"[hf-export] staged (NOT published) under {stage}")
    print("[hf-export] publishing requires explicit approval; nothing was uploaded.")


if __name__ == "__main__":
    main()
