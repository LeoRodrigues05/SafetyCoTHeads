"""Judge a small Direction A v5 subset with standard and safety-reasoning metrics.

Default target: Qwen3-8B completions, 40 JBB + 40 BT rows per condition.
Outputs are written to a separate subset tree so full-run generation and judge
outputs are not modified.

Usage:
    python -m scripts.run_v5_qwen_subset_judging --dry-run
    python -m scripts.run_v5_qwen_subset_judging
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg  # noqa: E402
from run_v4_jbb_judge import _process_condition  # noqa: E402

from safety_cot_heads.direction_a import segment_completion  # noqa: E402
from safety_cot_heads.judging import JudgeConfig, judge_rows  # noqa: E402
from safety_cot_heads.models import load_model  # noqa: E402
from safety_cot_heads.utils import (  # noqa: E402
    ensure_dir,
    json_dump,
    jsonl_read,
    jsonl_write,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(jsonl_read(path))


def _completion_file(model_key: str, dkey: str, cond: str) -> Path | None:
    seed_dir = (
        ROOT / "runs" / "direction_a_v5" / model_key
        / "gen" / dkey / cond / "seed0"
    )
    preferred = seed_dir / f"completions_{cond}.jsonl"
    if preferred.exists():
        return preferred
    matches = sorted(seed_dir.glob("completions_*.jsonl"))
    return matches[0] if matches else None


def _iter_generation_cells(model_key: str, datasets: set[str],
                           conditions: set[str] | None):
    cfg_root = (
        ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr"
        / model_key / "gen"
    )
    for dset_dir in sorted(p for p in cfg_root.iterdir() if p.is_dir()):
        dkey = dset_dir.name
        if dkey not in datasets:
            continue
        for cfg_path in sorted(dset_dir.glob("*.yaml")):
            cond = cfg_path.stem
            if conditions and cond not in conditions:
                continue
            comp = _completion_file(model_key, dkey, cond)
            if comp is not None:
                yield dkey, cond, comp


def _stable_key(seed: int, *parts: str) -> str:
    text = "||".join([str(seed), *map(str, parts)])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _subset_rows(rows: list[dict], n: int, *, seed: int,
                 model_key: str, dkey: str, cond: str) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda r: _stable_key(seed, model_key, dkey, cond, str(r.get("id"))),
    )
    return ordered[: min(n, len(ordered))]


def _write_subset(
    *,
    model_key: str,
    dkey: str,
    cond: str,
    source_path: Path,
    out_base: Path,
    n_per_dataset: int,
    seed: int,
    write: bool = True,
) -> dict:
    rows = _jsonl_rows(source_path)
    subset = _subset_rows(
        rows,
        n_per_dataset,
        seed=seed,
        model_key=model_key,
        dkey=dkey,
        cond=cond,
    )
    out_dir = out_base / "_subsets" / dkey / cond / "seed0"
    out_path = out_dir / f"completions_{cond}.jsonl"
    if write:
        ensure_dir(out_dir)
        jsonl_write(out_path, subset)
    return {
        "model_key": model_key,
        "dataset": dkey,
        "condition": cond,
        "source_path": str(source_path),
        "subset_path": str(out_path),
        "n_source": len(rows),
        "n_subset": len(subset),
        "ids": [r.get("id") for r in subset],
    }


def _indexed_trace(row: dict) -> dict:
    seg = segment_completion(row.get("completion") or "")
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
    if not entries and row.get("completion"):
        entries.append({
            "global_index": 0,
            "section": "output",
            "index": 0,
            "text": row.get("completion") or "",
        })
    indexed = "\n".join(
        "[global={global_index} section={section} index={index}] {text}".format(**e)
        for e in entries
    )
    return {
        "indexed_text": indexed,
        "segments": entries,
        "n_cot_sentences": len(seg.think_sentences),
        "n_output_sentences": len(seg.answer_sentences),
        "n_trace_segments": len(entries),
        "segments_kind": seg.kind,
    }


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    valid_rows = []
    malformed = False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    malformed = True
                    break
                valid_rows.append(row)
                if row.get("id") is not None:
                    out.add(str(row.get("id")))
    except Exception:
        malformed = True
        valid_rows = []
        out = set()
    if malformed:
        jsonl_write(path, valid_rows)
    return out


def _reasoning_inputs(completions: list[dict]) -> list[dict]:
    out = []
    for row in completions:
        tr = _indexed_trace(row)
        out.append({
            "id": row["id"],
            "parent_id": row["id"],
            "prompt": row.get("prompt") or row.get("user_prompt") or "",
            "completion": tr["indexed_text"],
            "condition": row.get("condition"),
            "model": row.get("model"),
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "seed": row.get("seed"),
            "n_trace_segments": tr["n_trace_segments"],
            "n_cot_sentences": tr["n_cot_sentences"],
            "n_output_sentences": tr["n_output_sentences"],
            "trace_segments": tr["segments"],
        })
    return out


def _jcfg(cfg, *, kind: str, max_new_tokens: int) -> JudgeConfig:
    return JudgeConfig(
        kind=kind,
        max_new_tokens=int(max_new_tokens),
        base_temperature=float(cfg.get("base_temperature", 0.0)),
        retry_temperature=float(cfg.get("retry_temperature", 0.3)),
        max_retries=int(cfg.get("max_retries", 2)),
        seed=int(cfg.get("seed", 0)),
        batch_size=int(cfg.get("batch_size", 8)),
        use_chat_template=bool(cfg.get("use_chat_template", True)),
    )


def _load_judge(cfg):
    judge = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        attach_controllers=False,
    )
    hf_map = getattr(judge.model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        devices = sorted({str(v) for v in hf_map.values()})
        map_summary = ",".join(devices[:6])
        if len(devices) > 6:
            map_summary += f",...+{len(devices) - 6}"
    else:
        map_summary = str(hf_map)
    print(f"judge loaded: first_param_device={judge.device} hf_device_map={map_summary}")
    return judge


def _safety_reasoning_summary(rows: list[dict]) -> dict:
    parsed = [r for r in rows if r.get("judge_flat")]
    if not parsed:
        return {"n": len(rows), "n_parsed": 0}
    has = []
    first_norm = []
    extent_counts = []
    extent_fracs = []
    by_category: dict[str, int] = {}
    first_section: dict[str, int] = {}
    for row in parsed:
        flat = row.get("judge_flat") or {}
        has.append(bool(flat.get("has_safety_reasoning")))
        pos = flat.get("position") or {}
        extent = flat.get("extent") or {}
        if pos.get("first_global_index") is not None:
            denom = max(1, int(row.get("n_trace_segments", 1)) - 1)
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
            by_category[cat] = by_category.get(cat, 0) + 1
    n = len(parsed)
    return {
        "n": len(rows),
        "n_parsed": n,
        "safety_reasoning_rate": sum(1 for x in has if x) / n,
        "first_position_norm_mean": mean(first_norm) if first_norm else None,
        "extent_sentence_count_mean": mean(extent_counts) if extent_counts else None,
        "extent_fraction_mean": mean(extent_fracs) if extent_fracs else None,
        "first_section_hist": first_section,
        "span_category_hist": by_category,
    }


def _run_reasoning_judge(judge, cfg, completions_path: Path, out_dir: Path,
                         *, max_new_tokens: int) -> dict:
    ensure_dir(out_dir)
    out_path = out_dir / "judge_safety_reasoning_trace.jsonl"
    completions = _jsonl_rows(completions_path)
    inputs = _reasoning_inputs(completions)
    seen = _existing_ids(out_path)
    remaining = [r for r in inputs if str(r["id"]) not in seen]
    if remaining:
        judge_rows(
            judge,
            remaining,
            _jcfg(cfg, kind="safety_reasoning_trace", max_new_tokens=max_new_tokens),
            out_path=str(out_path),
        )
    rows = _jsonl_rows(out_path)
    summary = _safety_reasoning_summary(rows)
    json_dump(out_dir / "safety_reasoning.summary.json", summary)
    return summary


def _print_plan(subsets: list[dict], out_base: Path) -> None:
    print(f"subset output: {out_base}")
    print("dataset condition             source subset subset_file")
    print("------- --------------------- ------ ------ -----------")
    for s in subsets:
        print(
            f"{s['dataset']:<7} {s['condition']:<21} "
            f"{s['n_source']:>6} {s['n_subset']:>6} {s['subset_path']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="qwen3_8b")
    ap.add_argument("--datasets", nargs="+", default=["jbb", "bt"])
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--n-per-dataset", type=int, default=40,
                    help="Rows per dataset per condition.")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out-base", default=None)
    ap.add_argument("--skip-standard-metrics", action="store_true")
    ap.add_argument("--skip-safety-reasoning", action="store_true")
    ap.add_argument("--skip-safety", action="store_true",
                    help="Within standard metrics, skip the 5 completion-level safety labels.")
    ap.add_argument("--skip-pathway", action="store_true",
                    help="Within standard metrics, skip the expensive all-prefix pathway labels.")
    ap.add_argument("--skip-cot-only", action="store_true",
                    help="Within standard metrics, skip the CoT-only monitorability judge.")
    ap.add_argument("--skip-coherence", action="store_true",
                    help="Within standard metrics, skip the gibberish/coherence classifier.")
    ap.add_argument("--safety-reasoning-max-new-tokens", type=int, default=768)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_dir = (
        ROOT / "configs" / "experiments" / "direction_a_v5_iso_asr"
        / args.model_key
    )
    judge_cfg = load_cfg(cfg_dir / "judge.yaml")
    set_seed(int(judge_cfg.get("seed", 0)))

    out_base = Path(args.out_base) if args.out_base else (
        ROOT / "runs" / "direction_a_v5" / args.model_key
        / f"judge_subset_n{args.n_per_dataset}"
    )
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    if not args.dry_run:
        ensure_dir(out_base)

    datasets = set(args.datasets)
    conditions = set(args.conditions) if args.conditions else None
    subsets = []
    for dkey, cond, source_path in _iter_generation_cells(
        args.model_key,
        datasets,
        conditions,
    ):
        subsets.append(_write_subset(
            model_key=args.model_key,
            dkey=dkey,
            cond=cond,
            source_path=source_path,
            out_base=out_base,
            n_per_dataset=args.n_per_dataset,
            seed=args.sample_seed,
            write=not args.dry_run,
        ))

    if not subsets:
        print("no completion cells found")
        return 2
    _print_plan(subsets, out_base)
    if args.dry_run:
        print(
            f"DRY-RUN complete: selected {len(subsets)} cells; "
            "no subset files or judge outputs were written."
        )
        return 0
    json_dump(out_base / "subset_manifest.json", {
        "model_key": args.model_key,
        "n_per_dataset": args.n_per_dataset,
        "sample_seed": args.sample_seed,
        "subsets": subsets,
    })

    judge = _load_judge(judge_cfg)
    standard_summaries = []
    reasoning_summaries = []
    for subset in subsets:
        dkey = subset["dataset"]
        cond = subset["condition"]
        comp = Path(subset["subset_path"])
        out_dir = out_base / dkey / cond / "seed0"
        spec = {
            "tag": f"{dkey}/{cond}",
            "cond": cond,
            "completions": str(comp),
        }
        if not args.skip_standard_metrics:
            s = _process_condition(
                judge,
                judge_cfg,
                spec,
                out_dir,
                skip_safety=args.skip_safety,
                skip_pathway=args.skip_pathway,
                skip_cot_only=args.skip_cot_only,
                skip_coherence=args.skip_coherence,
            )
            standard_summaries.append({
                "dataset": dkey,
                "condition": cond,
                "summary": s,
            })
        if not args.skip_safety_reasoning:
            rs = _run_reasoning_judge(
                judge,
                judge_cfg,
                comp,
                out_dir,
                max_new_tokens=args.safety_reasoning_max_new_tokens,
            )
            reasoning_summaries.append({
                "dataset": dkey,
                "condition": cond,
                "summary": rs,
            })

    json_dump(out_base / "qwen_subset_judge.summary.json", {
        "model_key": args.model_key,
        "judge_config": cfg_to_dict(judge_cfg),
        "n_cells": len(subsets),
        "standard_metrics": standard_summaries,
        "safety_reasoning": reasoning_summaries,
    })
    print(f"wrote subset judge outputs under {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
