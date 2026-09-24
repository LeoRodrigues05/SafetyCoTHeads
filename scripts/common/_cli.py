"""Common helpers for CLI scripts: OmegaConf loading + dry-run plumbing."""
from __future__ import annotations
from pathlib import Path

from omegaconf import OmegaConf


def load_cfg(path: str | Path, overrides: list[str] | None = None):
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def cfg_to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)
