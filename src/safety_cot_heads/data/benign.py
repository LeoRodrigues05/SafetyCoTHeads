"""Benign quality datasets (Alpaca, MMLU, GSM8K, WikiText-103).

All loaders here use the HF ``datasets`` library; the heavy datasets are
downloaded lazily and cached in the standard HF cache.  Each loader returns
the canonical ``{"id", "prompt"}`` dict so they slot into the same generation
and judging pipeline as the harmful datasets.
"""

from __future__ import annotations
from typing import Optional


def load_alpaca(n: Optional[int] = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        prompt = ex["instruction"]
        if ex.get("input"):
            prompt = f"{prompt}\n{ex['input']}"
        out.append({"id": f"alpaca-{i:05d}", "prompt": prompt, "dataset": "alpaca"})
    return out


def load_mmlu(n: Optional[int] = None, subject: str | None = None) -> list[dict]:
    from datasets import load_dataset
    name = subject or "all"
    ds = load_dataset("cais/mmlu", name, split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        choices = "\n".join(f"({c}) {x}" for c, x in zip("ABCD", ex["choices"]))
        prompt = f"{ex['question']}\n{choices}\nAnswer:"
        out.append({"id": f"mmlu-{i:05d}", "prompt": prompt, "dataset": "mmlu",
                    "answer": "ABCD"[ex["answer"]]})
    return out


def load_gsm8k(n: Optional[int] = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for i, ex in enumerate(ds):
        out.append({"id": f"gsm8k-{i:05d}", "prompt": ex["question"],
                    "dataset": "gsm8k", "answer": ex["answer"]})
    return out


def load_wikitext(n: Optional[int] = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [t for t in ds["text"] if t.strip()]
    if n is not None:
        texts = texts[:n]
    return [{"id": f"wikitext-{i:05d}", "prompt": t, "dataset": "wikitext-103"}
            for i, t in enumerate(texts)]
