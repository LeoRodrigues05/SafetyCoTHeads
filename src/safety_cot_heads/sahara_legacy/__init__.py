"""Sahara attribution — reformatted from ydyjya/SafetyHeadAttribution.

The SVD-based subspace-similarity routine is vendored verbatim under
``_legacy/sha/sahara_svd.py``; this package wraps the attribution loop
around it with the new hook-based ``HeadMaskController``.
"""
from .sahara import SaharaConfig, safety_head_attribution, get_last_hidden_states

__all__ = ["SaharaConfig", "safety_head_attribution", "get_last_hidden_states"]
