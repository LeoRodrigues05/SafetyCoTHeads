"""Device resolution — no hardcoded ``cuda:0`` anywhere else in the package."""
from __future__ import annotations
import os
import torch


def resolve_device(explicit: str | None = None) -> torch.device:
    """Resolve a device string.

    Precedence: explicit arg > ``SAFETY_COT_DEVICE`` env > first visible CUDA > CPU.
    Honours ``CUDA_VISIBLE_DEVICES`` by deferring to torch's view of devices.
    """
    if explicit is not None:
        return torch.device(explicit)
    env = os.environ.get("SAFETY_COT_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
