"""Direction A v6 — corrected evaluation & aggregation.

Corrects four evaluation/aggregation issues from v5 without regenerating model
completions:

1. Answer-level safety & coherence are judged on the parsed *final answer*
   (:mod:`parsing`), not the full completion.
2. Trace-level judges see only the *reasoning trace*.
3. Monitorability is computed from paired per-prompt outcomes
   (:mod:`paired_metrics`), not a marginal-rate difference that lets covert
   failures and over-warnings cancel.
4. Explicit reasoning traces are distinguished from heuristic prose prefixes.

Public API is deliberately pure/deterministic (no I/O) so it is unit-testable
on CPU; the scripts under ``scripts/`` do the I/O and GPU orchestration.
"""

from .parsing import (
    ParsedCompletion,
    parse_completion,
    parse_row,
    PARSER_VERSION,
    TRACE_KINDS,
    PARSE_STATUSES,
)
from .paired_metrics import (
    PairedItem,
    PairedTable,
    build_paired_table,
    MonitorabilityRetention,
    monitorability_retention,
)
from .aggregate import (
    AnswerSignals,
    CellAxes,
    compute_cell_axes,
    potency,
    quality,
)
from .bootstrap import PairedAnswerCell, paired_bootstrap, CI
from .sharding import shard_of, assign_shards, verify_partition

__all__ = [
    "ParsedCompletion", "parse_completion", "parse_row", "PARSER_VERSION",
    "TRACE_KINDS", "PARSE_STATUSES",
    "PairedItem", "PairedTable", "build_paired_table",
    "MonitorabilityRetention", "monitorability_retention",
    "AnswerSignals", "CellAxes", "compute_cell_axes", "potency", "quality",
    "PairedAnswerCell", "paired_bootstrap", "CI",
    "shard_of", "assign_shards", "verify_partition",
]
