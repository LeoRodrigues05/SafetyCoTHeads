"""Direction A — failure-mode atlas, SHIPS slice.

Module layout:

- :mod:`segmentation` — sentence + ``<think>`` block segmentation, cumulative
  prefix builder.
- :mod:`trajectory_metrics` — the 7 trajectory metrics computed from
  per-prefix judge labels (Llama-prose and R1-Distill variants).
"""

from .segmentation import (
    Segments, segment_completion, build_prefix_rows,
)
from .trajectory_metrics import (
    trajectory_vector, METRIC_FIELDS, FIELD_DESCRIPTIONS,
)
from .pathway_taxonomy import (
    DOMINANT_PATHWAYS, PATHWAY_VECTOR_FIELDS,
    pathway_vector, summarise_pathways,
)
from .monitorability import (
    build_cot_only_inputs, compute_monitorability_gap,
)

__all__ = [
    "Segments", "segment_completion", "build_prefix_rows",
    "trajectory_vector", "METRIC_FIELDS", "FIELD_DESCRIPTIONS",
    "PATHWAY_VECTOR_FIELDS", "DOMINANT_PATHWAYS",
    "pathway_vector", "summarise_pathways",
    "build_cot_only_inputs", "compute_monitorability_gap",
]
