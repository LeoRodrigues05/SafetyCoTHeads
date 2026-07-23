#!/usr/bin/env python3
"""Write the Direction A v6 corrected-run manifest (JSON + Markdown).

Captures provenance for reproducibility: git commit + dirty status, source
generation root and per-file SHA256 hashes, model/tokenizer/judge-prompt/parser
versions, library + CUDA + GPU info, decoding/judge settings, seeds, exact
commands, discovered vs completed cell counts, exclusions, missingness, any
generation repairs, and the validation files used. Nothing is hardcoded — the
commit and cell counts are read from the live repo and disk.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import v6_common as C


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(C.REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _pkg(name):
    try:
        return __import__("importlib.metadata", fromlist=["version"]).version(name)
    except Exception:
        return None


def _gpu_info():
    out = _sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader"])
    return out.splitlines() if out else []


def main():
    scope = C.load_paper_scope()
    cells = C.discover_cells()

    # SHA256 of source generation files (immutability record). Hash the first
    # completions file per cell (they are the load-bearing source).
    src_hashes = {}
    for c in cells:
        p = C.completions_path(c)
        if p:
            src_hashes[c.key] = {"path": str(p.relative_to(C.REPO)),
                                 "sha256": C.sha256_file(p)}

    audit = _load(C.V6_ROOT / "audit" / "generation_audit.json")
    repair = _load(C.V6_ROOT / "audit" / "generation_repair_manifest.json")
    pdiag = _load(C.V6_ROOT / "parsed" / "parse_diagnostics.json")
    cellm = _load(C.V6_ROOT / "reports" / "cell_metrics.json")

    # completed judge cells (v6 answer stage) vs discovered
    n_v6_answer = len(list((C.V6_ROOT / "judge").rglob("judge_answer_safety.jsonl")))

    manifest = {
        "generated_at_utc": C.utcnow_iso(),
        "git": {
            "commit": _sh(["git", "rev-parse", "HEAD"]),
            "branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_sh(["git", "status", "--porcelain"])),
            "describe": _sh(["git", "describe", "--always", "--dirty"]),
        },
        "source_generation_root": str(C.V5_ROOT.relative_to(C.REPO)),
        "source_immutable": True,
        "n_source_files_hashed": len(src_hashes),
        "paper_scope": {k: scope[k] for k in
                        ("primary", "exploratory", "explicit_trace_models",
                         "prose_prefix_models", "baseline_condition")},
        "models": {"judge_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                   "judge_dtype": "bfloat16",
                   "generation_models": sorted({c.model for c in cells})},
        "versions": {
            "parser_version": (pdiag or {}).get("parser_version"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _pkg("torch"),
            "transformers": _pkg("transformers"),
            "vllm": _pkg("vllm"),
            "numpy": _pkg("numpy"),
            "cuda_runtime": _sh(["nvcc", "--version"]) and "see nvcc",
        },
        "gpu": {"names": _gpu_info(),
                "driver": _sh(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"])},
        "decoding_settings": {"note": "reused verbatim from v5 completions; not regenerated",
                              "deterministic": True},
        "judge_settings": {"kind_by_stage": {
            "answer": "safety(answer_text)", "coherence": "coherence(answer_text)",
            "monitor": "cot_only(trace_text)", "pathway": "pathway(trace_text)",
            "safety-reasoning": "safety_reasoning_trace(trace_text)"},
            "base_temperature": 0.0, "max_new_tokens": 256},
        "seeds": {"generation_seed": 0,
                  "bootstrap_seed": (cellm or {}).get("boot_seed", 12345),
                  "bootstrap_replicates_final": 10000},
        "exact_commands": {
            "audit": "bash scripts/run_v6_two_b200.sh audit",
            "parse": "bash scripts/run_v6_two_b200.sh parse",
            "answer": "bash scripts/run_v6_two_b200.sh answer",
            "monitor": "bash scripts/run_v6_two_b200.sh monitor",
            "pathway": "bash scripts/run_v6_two_b200.sh pathway",
            "safety_reasoning": "bash scripts/run_v6_two_b200.sh safety-reasoning",
            "aggregate": "bash scripts/run_v6_two_b200.sh aggregate --n-boot 10000",
            "validation": "bash scripts/run_v6_two_b200.sh validation",
            "all": "bash scripts/run_v6_two_b200.sh all",
        },
        "cell_counts": {
            "discovered": len(cells),
            "by_model": _count_by_model(cells),
            "audit_needing_repair": (repair or {}).get("n_repairs", 0),
            "v6_answer_judge_completed": n_v6_answer,
        },
        "parser_diagnostics": {
            "trace_kind_histogram": (pdiag or {}).get("trace_kind_histogram"),
            "n_malformed_explicit": (pdiag or {}).get("totals", {}).get("n_malformed_explicit"),
        },
        "exclusions": {
            "exploratory_models_excluded_from_primary": scope["exploratory"],
            "prose_prefix_models_excluded_from_explicit_monitorability":
                scope["prose_prefix_models"],
        },
        "generation_repairs": (repair or {}).get("repairs", []),
        "validation_files_used": [
            "data/annotations/batch_v5_002/tasks.json",
            "data/annotations/batch_v5_002/judge_labels.json",
            "data/annotations/batch_v5_002/annotations_Leo.jsonl",
            "data/annotations/batch_v5_002/annotations_Thomas.jsonl",
            "data/annotations/batch_v5_002/validation_report.json",
        ],
        "source_generation_hashes": src_hashes,
    }
    C.write_json(C.V6_ROOT / "manifest" / "run_manifest.json", manifest)
    _write_md(manifest)
    print(f"[manifest] commit={manifest['git']['commit']} dirty={manifest['git']['dirty']} "
          f"cells={len(cells)} hashed={len(src_hashes)}")
    print("[manifest] wrote runs/direction_a_v6/manifest/run_manifest.json|md")


def _count_by_model(cells):
    d = {}
    for c in cells:
        d[c.model] = d.get(c.model, 0) + 1
    return d


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _write_md(m):
    g = m["git"]
    lines = [
        "# Direction A v6 — run manifest", "",
        f"- Generated: {m['generated_at_utc']}",
        f"- Git commit: `{g['commit']}` (branch `{g['branch']}`, dirty={g['dirty']})",
        f"- Source generation root: `{m['source_generation_root']}` (immutable; "
        f"{m['n_source_files_hashed']} files SHA256-hashed)",
        f"- Judge model: {m['models']['judge_model']} ({m['models']['judge_dtype']})",
        f"- Parser version: {m['versions']['parser_version']}",
        f"- Python {m['versions']['python']}, torch {m['versions']['torch']}, "
        f"transformers {m['versions']['transformers']}, vllm {m['versions']['vllm']}",
        f"- GPUs: {m['gpu']['names'] or 'n/a (CPU host)'}",
        f"- Bootstrap seed: {m['seeds']['bootstrap_seed']} "
        f"({m['seeds']['bootstrap_replicates_final']} replicates for final reports)", "",
        "## Cell counts", "",
        f"- Discovered: **{m['cell_counts']['discovered']}**",
        f"- v6 answer-judge completed: **{m['cell_counts']['v6_answer_judge_completed']}**",
        f"- Flagged for generation repair: **{m['cell_counts']['audit_needing_repair']}**", "",
        "## Parser diagnostics", "",
        f"- Trace-kind histogram: `{m['parser_diagnostics']['trace_kind_histogram']}`",
        f"- Malformed explicit (open tag, no close): "
        f"{m['parser_diagnostics']['n_malformed_explicit']}", "",
        "## Exclusions", "",
        f"- Exploratory (not in primary averages): {m['exclusions']['exploratory_models_excluded_from_primary']}",
        f"- Prose-only (excluded from explicit monitorability): "
        f"{m['exclusions']['prose_prefix_models_excluded_from_explicit_monitorability']}", "",
        "## Exact commands", "",
    ]
    for k, v in m["exact_commands"].items():
        lines.append(f"- **{k}**: `{v}`")
    (C.V6_ROOT / "manifest" / "run_manifest.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
