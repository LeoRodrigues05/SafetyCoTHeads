"""LoRA fine-tune a Qwen3-14B pathway judge on HarmThoughts training data.

Training data is produced by scripts/prepare_harmthoughts_training_data.py.
Each example is a (SINGLE_LABEL_JUDGE_PROMPT, target_json) pair.

The LoRA adapter is trained to output strict JSON for one pathway label at a
time, matching the inference-time format exactly.  After training, the adapter
can be merged into the base model for deployment.

Multi-GPU (DDP):
    torchrun --nproc_per_node=N scripts/train_pathway_judge.py \
        --base-model Qwen/Qwen3-14B \
        --training-data data/pathway_judge_train.jsonl \
        --out-dir runs/pathway_judge_14b_lora

Single GPU:
    python scripts/train_pathway_judge.py \
        --base-model Qwen/Qwen3-14B \
        --training-data data/pathway_judge_train.jsonl \
        --out-dir runs/pathway_judge_14b_lora

Dry run (no GPU, verifies data loading and prints plan):
    python scripts/train_pathway_judge.py \
        --base-model Qwen/Qwen3-14B \
        --training-data data/pathway_judge_train.jsonl \
        --out-dir runs/pathway_judge_14b_lora \
        --dry-run

Merge LoRA adapter into full model after training:
    python scripts/train_pathway_judge.py \
        --merge-only \
        --adapter-dir runs/pathway_judge_14b_lora \
        --out-dir models/pathway_judge_14b_merged
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_cot_heads.utils import ensure_dir, get_logger, jsonl_read  # noqa: E402

log = get_logger(__name__)


def _load_examples(path: Path) -> list[dict]:
    rows = list(jsonl_read(path))
    valid = [r for r in rows if "user_prompt" in r and "target_json" in r]
    if len(valid) < len(rows):
        log.warning("skipped %d rows missing user_prompt/target_json", len(rows) - len(valid))
    return valid


def _format_example(ex: dict, tokenizer) -> dict:
    """Format one (user_prompt, target_json) pair into causal-LM tokens.

    Loss is computed only on the target JSON tokens (prompt tokens masked to -100).
    """
    prompt_text = ex["user_prompt"]
    target_text = json.dumps(ex["target_json"], ensure_ascii=False)

    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    # Tokenize the full sequence (prompt + target + EOS)
    full_text = tokenizer.decode(prompt_ids) + target_text + tokenizer.eos_token
    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
    )
    # Mask prompt tokens so loss is computed only on target JSON
    labels = list(full_ids["input_ids"])
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100

    return {
        "input_ids": full_ids["input_ids"],
        "attention_mask": full_ids["attention_mask"],
        "labels": labels,
    }


def do_merge(adapter_dir: Path, out_dir: Path) -> None:
    """Merge a LoRA adapter into the base model and save to out_dir."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading adapter config from %s", adapter_dir)
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"no adapter_config.json found in {adapter_dir}")
    with adapter_cfg_path.open() as f:
        base_model_name = json.load(f).get("base_model_name_or_path")
    log.info("base model: %s", base_model_name)

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    log.info("merging LoRA weights ...")
    model = model.merge_and_unload()

    ensure_dir(out_dir)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    log.info("merged model saved to %s", out_dir)


def do_train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    examples = _load_examples(args.training_data)
    log.info("loaded %d training examples from %s", len(examples), args.training_data)
    if args.max_train_samples is not None:
        examples = examples[: int(args.max_train_samples)]
        log.info("limited to %d training examples", len(examples))

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    plan = {
        "base_model": args.base_model,
        "n_examples": len(examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_steps": args.max_steps,
        "max_train_samples": args.max_train_samples,
        "out_dir": str(args.out_dir),
    }
    rank = int(os.environ.get("RANK", "0"))
    is_main_process = rank == 0
    ensure_dir(args.out_dir)
    if is_main_process:
        (args.out_dir / "train_plan.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8")
        log.info("training plan: %s", json.dumps(plan))

        # Write 8-row preview for inspection
        preview_rows = [
            {"user_prompt": ex["user_prompt"][:200] + "...", "target_json": ex["target_json"]}
            for ex in examples[:8]
        ]
        (args.out_dir / "train_examples_preview.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in preview_rows),
            encoding="utf-8")

    if args.dry_run:
        if is_main_process:
            log.info("DRY-RUN: wrote plan and preview. Skipping model load.")
        return

    log.info("tokenising %d examples ...", len(examples))
    ds = Dataset.from_list(examples).map(
        lambda ex: _format_example(ex, tok),
        remove_columns=["user_prompt", "target_json"],
        desc="tokenising",
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if world_size > 1 else "auto"
    log.info("loading model with device_map=%s", device_map)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,           # keep only 2 adapter checkpoints
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to=[],
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    # Save only the LoRA adapter (not the full merged model).  In DDP, only
    # rank 0 should write final artifacts to avoid concurrent file writes.
    if trainer.is_world_process_zero():
        unwrapped = trainer.accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(args.out_dir))
        tok.save_pretrained(str(args.out_dir))
        log.info("LoRA adapter saved to %s", args.out_dir)
        log.info("To merge: python scripts/train_pathway_judge.py --merge-only "
                 "--adapter-dir %s --out-dir models/pathway_judge_14b_merged", args.out_dir)
    trainer.accelerator.wait_for_everyone()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fine-tune Qwen3-14B as a pathway judge via LoRA")
    ap.add_argument("--base-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--training-data", type=Path,
                    default=Path("data/pathway_judge_train.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Per-device batch size (use 4 per A100 80GB)")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch_size × accum)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable gradient checkpointing; use --no-gradient-checkpointing on high-VRAM GPUs")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Use only the first N training examples for smoke tests")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Override training length for smoke tests; -1 means use epochs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Tokenise data, write plan JSON, do not load the model")
    ap.add_argument("--merge-only", action="store_true",
                    help="Merge LoRA adapter into base model (skips training)")
    ap.add_argument("--adapter-dir", type=Path,
                    help="LoRA adapter directory (for --merge-only)")
    args = ap.parse_args()

    if args.merge_only:
        if args.adapter_dir is None or args.out_dir is None:
            ap.error("--merge-only requires --adapter-dir and --out-dir")
        do_merge(args.adapter_dir, args.out_dir)
    else:
        if not args.training_data.exists():
            ap.error(f"training data not found: {args.training_data}")
        do_train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
