"""SHIPS attribution — reformatted from ydyjya/SafetyHeadAttribution.

The underlying algorithm and the KL-divergence / sort utilities are upstream
(see ``_legacy/sha/pd_diff.py`` and ``_legacy/sha/ships_utils.py``); this
package wraps them with the new hook-based ``HeadMaskController``.
"""
from .ships import SHIPS, SHIPSConfig, aggregate_dataset_ranking

__all__ = ["SHIPS", "SHIPSConfig", "aggregate_dataset_ranking"]
