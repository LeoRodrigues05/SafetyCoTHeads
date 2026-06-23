"""Evaluate a fine-tuned pathway judge against the Qwen3-30B baseline.

Runs both judges on the held-out HarmThoughts test split (produced by
prepare_harmthoughts_training_data.py) and reports:
  - Per-label binary F1, precision, recall
  - Overall agreement rate and Cohen's kappa
  - Pathway vector Spearman correlation (if --completions provided)

Usage:
    # Basic: compare on held-out HarmThoughts test set
    python scripts/eval_pathway_judge.py \
        --test-data  data/pathway_judge_test.jsonl \
        --finetuned-model models/pathway_judge_14b_merged \
        --baseline-config configs/experiments/direction_a_v5_iso_asr/olmo3_7b_base/judge.yaml \
        --out eval_pathway_judge_results.json

    # With pathway vector correlation on actual experiment completions:
    python scripts/eval_pathway_judge.py \
        --test-data  data/pathway_judge_test.jsonl \
        --finetuned-model models/pathway_judge_14b_merged \
        --baseline-config configs/experiments/direction_a_v5_iso_asr/olmo3_7b_base/judge.yaml \
        --completions runs/direction_a_v5/olmo3_7b_base/gen/bt/baseline/seed0/completions_baseline.jsonl \
        --out eval_pathway_judge_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_cot_heads.utils import get_logger, jsonl_read  # noqa: E402

log = get_logger(__name__)


def _load_test_examples(path: Path) -> list[dict]:
    rows = list(jsonl_read(path))
    return [r for r in rows if "user_prompt" in r and "target_json" in r]


def _run_judge_on_examples(
    examples: list[dict],
    model_path: str,
    *,
    batch_size: int = 16,
) -> list[bool]:
    """Run a HuggingFace model on judge prompts; return list of label_present bools."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from safety_cot_heads.judging.parse import parse_judge_json, normalize_single_label

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def format_prompt(prompt: str) -> str:
        if getattr(tok, "chat_template", None) is None:
            return prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    results: list[bool] = []
    try:
        for i in range(0, len(examples), batch_size):
            batch = examples[i: i + batch_size]
            prompts = [format_prompt(ex["user_prompt"]) for ex in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            for _j, ids in enumerate(out_ids):
                n_prompt = inputs["input_ids"].shape[1]
                raw = tok.decode(ids[n_prompt:], skip_special_tokens=True)
                parsed = parse_judge_json(raw)
                data = (
                    parsed.get("data")
                    if parsed.get("parse_status") in ("ok", "recovered")
                    else None
                )
                norm = normalize_single_label(data)
                results.append(bool((norm or {}).get("label_present", False)))
            if (i // batch_size) % 10 == 0:
                log.info("  judged %d/%d", min(i + batch_size, len(examples)), len(examples))
    finally:
        tok.padding_side = old_padding_side
    return results


def _metrics(gold: list[bool], pred: list[bool], label: str) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(not g and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = sum(not g and not p for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    agree = (tp + tn) / len(gold) if gold else 0.0
    # Cohen's kappa
    p_e = ((tp + fp) / len(gold)) * ((tp + fn) / len(gold)) + \
          ((tn + fn) / len(gold)) * ((tn + fp) / len(gold))
    kappa = (agree - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
    return {
        "label": label,
        "n": len(gold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "agreement": round(agree, 4),
        "cohen_kappa": round(kappa, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate fine-tuned vs baseline pathway judge")
    ap.add_argument("--test-data", type=Path, required=True,
                    help="Held-out test JSONL from prepare_harmthoughts_training_data.py")
    ap.add_argument("--finetuned-model", required=True,
                    help="Path to merged fine-tuned model directory")
    ap.add_argument("--baseline-config", type=Path, default=None,
                    help="Judge YAML config for the Qwen3-30B baseline (optional; "
                         "if omitted, only the fine-tuned model is evaluated)")
    ap.add_argument("--completions", type=Path, default=None,
                    help="Completions JSONL for pathway vector correlation check (optional)")
    ap.add_argument("--out", type=Path, default=Path("eval_pathway_judge_results.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N examples (for quick testing)")
    args = ap.parse_args()

    examples = _load_test_examples(args.test_data)
    log.info("loaded %d test examples", len(examples))
    if args.limit:
        examples = examples[:args.limit]
        log.info("limited to %d examples", len(examples))

    gold_labels: list[bool] = [ex["target_json"]["label_present"] for ex in examples]
    repo_labels: list[str] = []
    for ex in examples:
        # Extract repo label from user_prompt (line starting with "LABEL:")
        for line in ex["user_prompt"].splitlines():
            if line.startswith("LABEL:"):
                repo_labels.append(line.split(":", 1)[1].strip())
                break
        else:
            repo_labels.append("unknown")

    log.info("running fine-tuned model: %s", args.finetuned_model)
    ft_preds = _run_judge_on_examples(
        examples, args.finetuned_model, batch_size=args.batch_size)

    baseline_preds: list[bool] | None = None
    if args.baseline_config is not None:
        import yaml
        with args.baseline_config.open() as f:
            cfg = yaml.safe_load(f)
        baseline_model = cfg["model"]["name"]
        log.info("running baseline judge: %s (from %s)", baseline_model, args.baseline_config)
        baseline_preds = _run_judge_on_examples(
            examples, baseline_model, batch_size=args.batch_size)

    # Compute per-label metrics
    label_to_gold: dict[str, list[bool]] = defaultdict(list)
    label_to_ft: dict[str, list[bool]] = defaultdict(list)
    label_to_bl: dict[str, list[bool]] = defaultdict(list)
    for i, lbl in enumerate(repo_labels):
        label_to_gold[lbl].append(gold_labels[i])
        label_to_ft[lbl].append(ft_preds[i])
        if baseline_preds is not None:
            label_to_bl[lbl].append(baseline_preds[i])

    results: dict = {"per_label": {}}
    for lbl in sorted(label_to_gold):
        g = label_to_gold[lbl]
        ft_m = _metrics(g, label_to_ft[lbl], lbl)
        entry: dict = {"finetuned": ft_m}
        if label_to_bl:
            bl_m = _metrics(g, label_to_bl[lbl], lbl)
            entry["baseline"] = bl_m
            entry["f1_ratio"] = (
                round(ft_m["f1"] / bl_m["f1"], 3) if bl_m["f1"] > 0 else None)
        results["per_label"][lbl] = entry

    # Overall agreement
    ft_overall = _metrics(gold_labels, ft_preds, "overall")
    results["overall_finetuned"] = ft_overall
    if baseline_preds is not None:
        bl_overall = _metrics(gold_labels, baseline_preds, "overall")
        results["overall_baseline"] = bl_overall
        results["overall_f1_ratio"] = (
            round(ft_overall["f1"] / bl_overall["f1"], 3)
            if bl_overall["f1"] > 0 else None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("results written to %s", args.out)

    # Print summary
    print("\n=== Evaluation Summary ===")
    for lbl, entry in results["per_label"].items():
        ft = entry["finetuned"]
        ratio_str = (f"  ratio={entry['f1_ratio']:.3f}"
                     if entry.get("f1_ratio") is not None else "")
        print(f"  {lbl:<35} ft_F1={ft['f1']:.3f}  kappa={ft['cohen_kappa']:.3f}{ratio_str}")
    print(f"\n  Overall fine-tuned F1: {ft_overall['f1']:.3f}  "
          f"kappa={ft_overall['cohen_kappa']:.3f}")
    if baseline_preds is not None:
        print(f"  Overall baseline F1:   {results['overall_baseline']['f1']:.3f}")
        print(f"  F1 ratio (ft/baseline): {results.get('overall_f1_ratio')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
