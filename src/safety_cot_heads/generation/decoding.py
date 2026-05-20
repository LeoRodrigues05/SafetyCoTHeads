"""Decoding configs."""

from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    seed: int = 0

    def to_hf_kwargs(self) -> dict:
        kw = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_return_sequences": self.num_return_sequences,
        }
        if self.do_sample:
            kw.update(temperature=self.temperature, top_p=self.top_p)
            if self.top_k > 0:
                kw["top_k"] = self.top_k
        if self.repetition_penalty != 1.0:
            kw["repetition_penalty"] = self.repetition_penalty
        return kw

    def asdict(self) -> dict:
        return asdict(self)
