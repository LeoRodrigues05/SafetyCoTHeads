#!/usr/bin/env python3
"""Finish reasoning traces that were cut off by the original ``max_new_tokens``.

Why: OLMo-3-Think (2048-token cap) leaves 4-12% of traces unterminated at most
conditions and 29-41% under steering a=1.0; Qwen3-8B (1024-token cap) leaves
4-10%. An unterminated trace has no final answer, so it can contribute neither
a harmfulness label nor a trace/answer pair. Rather than regenerate whole cells
with a larger cap, this continues ONLY the truncated rows from where they
stopped: under greedy decoding the continuation of a prefix equals what one
longer run would have produced (up to bf16 batch-composition effects, which the
original runs share), and the intervention is rebuilt from the row's own config
via ``run_generation.build_interventions`` so it is identical.

Selection (per row): the trace is open (the prompt pre-fills ``<think>`` or the
completion emitted it), there is no ``</think>``, and the completion used the
whole original token budget. Rows whose tail is already a verbatim repetition
loop (repeat3 >= --degenerate-repeat3 over the last 300 words) are left as-is
and marked ``continuation_skipped="degenerate_loop"``: greedy decoding does not
leave such a loop, and those rows fail the coherence gate either way.

Output: ``runs/direction_a_v6/gen_continued/<model>/<ds>/<cond>/seed0/``
holding ALL rows of the cell (unchanged rows verbatim), which
``v6_common.completions_path`` then prefers over the immutable v5 file. A cell
with nothing to continue gets no file. Resume-safe at cell granularity.

Usage (one process per GPU; cells are split deterministically by --shard):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/generation/continue_truncated_generations.py \
      --models olmo3_7b_think --budget olmo3_7b_think=8192 --shard 0/2
  .venv/bin/python scripts/generation/continue_truncated_generations.py --models olmo3_7b_think \
      qwen3_8b --plan-only          # CPU: count rows, no model
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import v6_common as C  # noqa: E402
from safety_cot_heads.analysis.coherence import ngram_repeat_fraction  # noqa: E402
from safety_cot_heads.direction_a_v6.parsing import (  # noqa: E402
    _THINK_CLOSE_RE, _THINK_OPEN_RE, prompt_prefills_trace)
from safety_cot_heads.direction_a_v6.sharding import shard_of  # noqa: E402

#: default total budgets (original caps: OLMo-3-Think 2048, Qwen3/R1 1024)
DEFAULT_BUDGET = {"olmo3_7b_think": 8192, "qwen3_8b": 4096, "r1_distill_qwen_7b": 4096}


def trace_is_open(row: dict) -> bool:
    comp = row.get("completion") or ""
    if _THINK_CLOSE_RE.search(comp):
        return False
    return prompt_prefills_trace(row.get("rendered_prompt")) or bool(_THINK_OPEN_RE.search(comp))


def is_degenerate(text: str, threshold: float) -> bool:
    tail = " ".join((text or "").split()[-300:])
    return ngram_repeat_fraction(tail, 3) >= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=["jbb", "bt"])
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="restrict to these conditions (default: every cell on disk)")
    ap.add_argument("--exclude-prefix", nargs="*",
                    default=["heads_amplify", "neurons_amplify", "steering_defend", "defend_prompt"],
                    help="skip conditions with these prefixes (default: the defence-side "
                         "Exp-5 cells, which the paper does not report); pass nothing to include")
    ap.add_argument("--budget", nargs="*", default=[],
                    help="model=total_max_new_tokens overrides, e.g. olmo3_7b_think=8192")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--degenerate-repeat3", type=float, default=0.5)
    ap.add_argument("--shard", default="0/1", help="i/n: process cells with shard_of==i")
    ap.add_argument("--plan-only", action="store_true", help="count work; load no model")
    args = ap.parse_args()

    budget = dict(DEFAULT_BUDGET)
    for kv in args.budget:
        k, v = kv.split("=")
        budget[k] = int(v)
    si, sn = (int(x) for x in args.shard.split("/"))

    from _cli import load_cfg
    tot = Counter()
    for model in args.models:
        cells = [c for c in C.discover_cells([model], args.datasets)
                 if (args.conditions is None or c.condition in args.conditions)
                 and not any(c.condition.startswith(p) for p in (args.exclude_prefix or []))
                 and shard_of(c.key, sn) == si]
        lm = None
        lm_sig = None
        for cell in cells:
            out_dir = C.continued_gen_dir(cell)
            src = sorted(cell.gen_dir().glob("completions*.jsonl"))
            if not src:
                continue
            rows = C.read_jsonl(src[0])
            cap = int((rows[0].get("decoding") or {}).get("max_new_tokens", 0)) if rows else 0
            total = budget.get(model)
            if not rows or total is None or total <= cap:
                continue
            done_manifest = out_dir / "continuation_manifest.json"
            if done_manifest.exists():
                import json
                m = json.loads(done_manifest.read_text())
                if m.get("total_max_new_tokens") == total and m.get("source_sha256") == C.sha256_file(src[0]):
                    tot["cells_already_done"] += 1
                    continue
            open_rows = [i for i, r in enumerate(rows) if trace_is_open(r)]
            if not open_rows:
                continue
            degenerate = [i for i in open_rows
                          if is_degenerate(rows[i].get("completion"), args.degenerate_repeat3)]
            cand = [i for i in open_rows if i not in set(degenerate)]
            tot["cells"] += 1
            tot["open_rows"] += len(open_rows)
            tot["degenerate_skipped"] += len(degenerate)
            if args.plan_only:
                tot["rows_to_continue(upper bound)"] += len(cand)
                print(f"  {cell.key:55s} open={len(open_rows):3d} degenerate={len(degenerate):3d} "
                      f"to_continue<={len(cand):3d} cap={cap}->{total}")
                continue

            cfg_path = rows[0].get("config_path")
            cfg = load_cfg(cfg_path, [])
            sig = (cfg.model.name, cfg.model.get("dtype", "auto"), cfg.model.get("attn_implementation"))
            if lm is None:
                from safety_cot_heads.models import load_model
                lm = load_model(cfg.model.name, dtype=sig[1], attn_implementation=sig[2],
                                device_map=cfg.model.get("device_map"),
                                trust_remote_code=bool(cfg.model.get("trust_remote_code", False)))
                lm_sig = sig
            elif sig != lm_sig:
                raise RuntimeError(f"{cell.key}: model settings {sig} differ from loaded {lm_sig}")

            # only rows that really used the whole budget were cut off
            tok = lm.tokenizer
            cand = [i for i in cand
                    if len(tok(rows[i]["completion"], add_special_tokens=False)["input_ids"]) >= cap - 8]
            if not cand:
                continue
            from run_generation import build_interventions
            from safety_cot_heads.generation import DecodingConfig, continue_texts
            iv = build_interventions(cfg, lm, config_path=str(cfg_path))
            dec = dict(cfg.get("decoding") or {})
            dec["max_new_tokens"] = total - cap
            decoding = DecodingConfig(**dec)
            texts = [rows[i]["rendered_prompt"] + rows[i]["completion"] for i in cand]
            new = continue_texts(lm, texts, decoding, mask_cfg=iv["mask_cfg"],
                                 neuron_cfg=iv["neuron_cfg"], steering_cfg=iv["steering_cfg"],
                                 batch_size=args.batch_size)
            out_rows = [dict(r) for r in rows]
            n_term = 0
            for i, t in zip(cand, new):
                orig = rows[i]["completion"]
                term = bool(_THINK_CLOSE_RE.search(t))
                n_term += int(term)
                out_rows[i]["completion"] = orig + t
                out_rows[i]["continued"] = True
                out_rows[i]["continuation"] = {
                    "orig_max_new_tokens": cap, "total_max_new_tokens": total,
                    "orig_completion_sha256": hashlib.sha256(orig.encode()).hexdigest(),
                    "new_chars": len(t), "trace_terminated": term,
                    "batch_size": args.batch_size,
                }
            for i in degenerate:
                out_rows[i]["continued"] = False
                out_rows[i]["continuation_skipped"] = "degenerate_loop"
            C.write_jsonl(out_dir / src[0].name, out_rows)
            C.write_json(done_manifest, {
                "cell": cell.key, "generated_at_utc": C.utcnow_iso(),
                "source": str(src[0].relative_to(C.REPO)), "source_sha256": C.sha256_file(src[0]),
                "orig_max_new_tokens": cap, "total_max_new_tokens": total,
                "n_rows": len(rows), "n_open": len(open_rows), "n_continued": len(cand),
                "n_terminated_after_continuation": n_term,
                "n_degenerate_skipped": len(degenerate),
                "degenerate_repeat3": args.degenerate_repeat3, "batch_size": args.batch_size,
                "config_path": cfg_path,
            })
            tot["continued"] += len(cand)
            tot["terminated"] += n_term
            print(f"  {cell.key}: continued {len(cand)}, now terminated {n_term}, "
                  f"degenerate left as-is {len(degenerate)}", flush=True)
    print("[continue] " + ", ".join(f"{k}={v}" for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
