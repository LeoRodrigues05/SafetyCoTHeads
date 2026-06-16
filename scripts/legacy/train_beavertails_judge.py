"""Fine-tune a small open-weight LM as a BeaverTails dual-score judge (LoRA).

REQUIRES A DEDICATED GPU. Do not invoke without one — it will silently load
multi-GB weights into RAM and crash the login node.

The training data is constructed by reading each BeaverTails per-category CSV
(see ``data/raw/beavertails/*.csv``), turning the raw ``(prompt, response,
is_safe)`` triples into supervised JSON targets that match the schema in
``safety_cot_heads.judging.BEAVERTAILS_JUDGE_PROMPT``.

The intent of this script is calibration / cost-reduction: once the Qwen2.5-32B
4-bit judge has produced a few thousand labels, those labels can be distilled
into a 7B-class judge via LoRA so that the per-experiment judge pass becomes
much cheaper. It is OPTIONAL — the main pipeline works with the Qwen judge
alone.

Usage (GPU node):
    python scripts/train_beavertails_judge.py \
        --base-model meta-llama/Llama-2-7b-chat-hf \
        --teacher-labels runs/08-beaver-baseline/judge_beavertails.jsonl \
                         runs/09-beaver-safety-ablation/judge_beavertails.jsonl \
        --out-dir runs/judge-finetune/llama2-7b-beavertails-lora \
        --epochs 1 --lr 1.0e-4 --batch-size 2

Dry run (CPU-safe):
    python scripts/train_beavertails_judge.py --dry-run \
        --base-model meta-llama/Llama-2-7b-chat-hf \
        --teacher-labels runs/08-beaver-baseline/judge_beavertails.jsonl \
        --out-dir runs/judge-finetune/llama2-7b-beavertails-lora
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_cot_heads.data import load_beavertails_judge_examples
from safety_cot_heads.judging import BEAVERTAILS_JUDGE_PROMPT
from safety_cot_heads.utils import ensure_dir, get_logger, jsonl_read

log = get_logger(__name__)


def _row_to_target(prompt: str, response: str, *, coherence: int,
                   malicious_intent: int, is_safe: bool,
                   primary_category: str,
                   coherence_rationale: str = "",
                   intent_rationale: str = "") -> dict:
    return {
        "user_prompt": BEAVERTAILS_JUDGE_PROMPT.format(
            prompt=prompt, response=response),
        "target_json": {
            "coherence": int(coherence),
            "malicious_intent": int(malicious_intent),
            "is_safe": bool(is_safe),
            "primary_category": primary_category,
            "coherence_rationale": coherence_rationale,
            "intent_rationale": intent_rationale,
        },
    }


def build_examples_from_teacher(teacher_paths: list[Path]) -> list[dict]:
    """Pull (prompt, response, judge_flat) from previously-judged JSONLs."""
    out: list[dict] = []
    for p in teacher_paths:
        for row in jsonl_read(p):
            flat = row.get("judge_flat")
            if not isinstance(flat, dict):
                continue
            scores = flat.get("scores") or {}
            if not (isinstance(scores.get("coherence"), int) and
                    isinstance(scores.get("malicious_intent"), int)):
                continue
            out.append(_row_to_target(
                prompt=row.get("prompt", ""),
                response=row.get("completion", ""),
                coherence=scores["coherence"],
                malicious_intent=scores["malicious_intent"],
                is_safe=bool(flat.get("is_safe", False)),
                primary_category=flat.get("primary_category") or "none",
                coherence_rationale=(flat.get("rationales") or {}).get("coherence", ""),
                intent_rationale=(flat.get("rationales") or {}).get("intent", ""),
            ))
    return out


def build_examples_from_raw() -> list[dict]:
    """Bootstrap pre-teacher: use the BeaverTails ``is_safe`` label to derive
    coarse 2/8 dual-scores. Useful only as a warm start."""
    rows = load_beavertails_judge_examples(n_per_category=50, seed=0,
                                            balance_safe=True)
    return [_row_to_target(
        prompt=r["prompt"], response=r["response"],
        coherence=r["coherence"], malicious_intent=r["malicious_intent"],
        is_safe=r["is_safe"], primary_category=r["primary_category"],
        coherence_rationale=r["coherence_rationale"],
        intent_rationale=r["intent_rationale"],
    ) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--teacher-labels", nargs="*", type=Path, default=[])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the dataset, write a plan JSON, do not load the model.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    examples = (build_examples_from_teacher(args.teacher_labels)
                if args.teacher_labels else build_examples_from_raw())
    log.info("built %d (prompt, target_json) supervised examples", len(examples))

    plan = {
        "base_model": args.base_model,
        "n_examples": len(examples),
        "teacher_label_sources": [str(p) for p in args.teacher_labels],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "train_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8")
    (args.out_dir / "train_examples_preview.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in examples[:8]),
        encoding="utf-8")

    if args.dry_run:
        log.info("DRY-RUN: wrote plan to %s and 8 preview examples. "
                 "Skipping model load.", args.out_dir / "train_plan.json")
        return 0

    # Heavy imports gated behind the non-dry-run path so that --dry-run works
    # on CPU-only login nodes without optional GPU deps installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _format(ex):
        user = ex["user_prompt"]
        target = json.dumps(ex["target_json"], ensure_ascii=False)
        text = (f"<s>[INST] {user} [/INST] {target} </s>")
        ids = tok(text, truncation=True, max_length=2048)
        ids["labels"] = ids["input_ids"].copy()
        return ids

    ds = Dataset.from_list(examples).map(_format, remove_columns=["user_prompt", "target_json"])
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.out_dir))
    log.info("saved LoRA adapter to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
