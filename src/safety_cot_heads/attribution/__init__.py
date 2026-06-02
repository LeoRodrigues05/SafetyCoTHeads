"""Attribution — new (non-upstream) methods designed for this repository.

Re-exports SHIPS and Sahara from their dedicated ``ships_legacy`` and
``sahara_legacy`` top-level packages so existing call sites continue to
work via ``from safety_cot_heads.attribution import ...``.
"""
from ..ships_legacy import SHIPS, SHIPSConfig, aggregate_dataset_ranking
from ..sahara_legacy import SaharaConfig, safety_head_attribution, get_last_hidden_states
from .coherency import CoherencyConfig, coherency_attribution
from .quality_heads import QualityConfig, quality_attribution
from .random_heads import uniform_random, layer_matched, activation_magnitude_matched
from .template_anchoring import (
    TemplateAnchoringConfig,
    compute_head_template_anchoring,
    residualize_on_template_anchoring,
)
from .neuron_attribution import NeuronAttributionConfig, neuron_attribution
from .directions import (
    DirectionExtractionConfig,
    compute_refusal_directions,
    save_directions_npz,
)

__all__ = [
    "SHIPS", "SHIPSConfig", "aggregate_dataset_ranking",
    "SaharaConfig", "safety_head_attribution", "get_last_hidden_states",
    "CoherencyConfig", "coherency_attribution",
    "QualityConfig", "quality_attribution",
    "uniform_random", "layer_matched", "activation_magnitude_matched",
    "TemplateAnchoringConfig",
    "compute_head_template_anchoring",
    "residualize_on_template_anchoring",
    "NeuronAttributionConfig", "neuron_attribution",
    "DirectionExtractionConfig", "compute_refusal_directions", "save_directions_npz",
]
