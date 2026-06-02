"""Template-anchoring diagnostic for attention heads (Phase 1 / P1.5).

Implements the rho_tpl statistic from the Direction A plan (§4.3 / §8.3):
the fraction of a head's attention mass placed on tokens belonging to the
prompt template region (e.g. ``## Query:`` / ``## Answer:`` boilerplate)
rather than the user-provided content. Heads with high rho_tpl behave
template-anchored: their ablation effects may be confounded with general
prompt-format sensitivity rather than safety-specific recognition.

Used as a robustness post-process on SHIPS / Sahara head rankings: regress
the discovery score on rho_tpl and report both the raw ranking and the
ranking residualised on template anchoring.

Only Llama/Mistral/Qwen2-style decoders with ``model.model.layers[i].self_attn``
are supported (same constraint as :class:`HeadMaskController`).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from ..models.custom_llama import num_layers_and_heads


@dataclass
class TemplateAnchoringConfig:
    """Configuration for per-head template-anchoring measurement.

    The template regions are defined by *substring spans* inside the rendered
    prompt; we tokenise the prompt and mark every token whose character span
    overlaps any template span. Everything else (query content + chat tokens
    not listed in ``extra_template_substrings``) is treated as non-template.

    Attributes
    ----------
    prompt_template
        Same format string used by SHIPS, with ``{q}`` marking the user-content
        slot, e.g. ``"## Query:{q}\\n## Answer:"``. We render this for every
        prompt and tokenise the rendered string.
    extra_template_substrings
        Additional substrings to count as template (e.g. system-prompt
        boilerplate, ``<|begin_of_text|>``-style chat scaffolding). Each
        substring is matched case-sensitively; all occurrences are anchored.
    max_prompt_len
        Truncate rendered prompts to this many tokens (left-truncation). Set
        to ``None`` to disable.
    normalise
        If ``True``, normalise per-(layer, head) by the within-prompt query-token
        count so that prompts of different lengths contribute comparably.
        Default ``True`` (recommended; reduces length confound).
    """

    prompt_template: str = "## Query:{q}\n## Answer:"
    extra_template_substrings: Sequence[str] = field(default_factory=tuple)
    max_prompt_len: int | None = 1024
    normalise: bool = True


def _template_char_spans(rendered: str, query: str, cfg: TemplateAnchoringConfig) -> list[tuple[int, int]]:
    """Return character spans (start, end_exclusive) of template regions."""
    spans: list[tuple[int, int]] = []
    # The template prefix/suffix come from cfg.prompt_template with ``{q}``
    # spliced out. We locate ``query`` in ``rendered`` and mark everything
    # except that slice as template.
    qstart = rendered.find(query)
    if qstart >= 0:
        if qstart > 0:
            spans.append((0, qstart))
        qend = qstart + len(query)
        if qend < len(rendered):
            spans.append((qend, len(rendered)))
    else:
        # Query not found verbatim (chat templating ate whitespace etc.);
        # mark nothing from the prompt-template path and rely on extras.
        pass
    for sub in cfg.extra_template_substrings:
        if not sub:
            continue
        start = 0
        while True:
            idx = rendered.find(sub, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(sub)))
            start = idx + len(sub)
    return spans


def _token_template_mask(
    tokenizer,
    rendered: str,
    query: str,
    cfg: TemplateAnchoringConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise ``rendered`` and return ``(input_ids, is_template_mask)``.

    ``is_template_mask`` is a 1-D bool tensor over tokens; ``True`` where the
    token's character span lies inside any template region.
    """
    enc = tokenizer(rendered, return_offsets_mapping=True,
                    add_special_tokens=True, return_tensors="pt",
                    truncation=cfg.max_prompt_len is not None,
                    max_length=cfg.max_prompt_len)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    spans = _template_char_spans(rendered, query, cfg)
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    for ti, (a, b) in enumerate(offsets):
        if b <= a:
            # special tokens have (0, 0) offsets; treat as template scaffolding
            mask[ti] = True
            continue
        for sa, sb in spans:
            if a >= sa and b <= sb:
                mask[ti] = True
                break
    return input_ids, mask


@torch.no_grad()
def compute_head_template_anchoring(
    model: nn.Module,
    tokenizer,
    prompts: Iterable[Mapping],
    cfg: TemplateAnchoringConfig | None = None,
) -> dict[tuple[int, int], float]:
    """Per-(layer, head) mean attention mass on template tokens.

    Parameters
    ----------
    model
        Decoder LM (Llama / Mistral / Qwen2). Must be loaded with an attention
        implementation that supports ``output_attentions=True`` (``eager`` is
        safe; flash-attn-2 does not return weights).
    tokenizer
        Matching tokenizer with ``return_offsets_mapping`` support (any HF
        fast tokenizer).
    prompts
        Iterable of dicts with at least ``{"prompt": str}``; missing field
        falls back to ``"text"``.
    cfg
        Anchoring config; defaults to the SHIPS ``## Query:{q}\\n## Answer:``
        template with no extras.

    Returns
    -------
    dict
        ``{(layer_idx, head_idx): rho_tpl}`` averaged over prompts. ``rho_tpl``
        is in [0, 1]: 1 means the head attends exclusively to template tokens,
        0 means it never does.
    """
    cfg = cfg or TemplateAnchoringConfig()
    n_layers, n_heads, _ = num_layers_and_heads(model)
    device = next(model.parameters()).device

    sums = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    counts = torch.zeros(n_layers, n_heads, dtype=torch.float64)

    for row in prompts:
        query = str(row.get("prompt") or row.get("text") or "")
        if not query:
            continue
        rendered = cfg.prompt_template.format(q=query)
        input_ids, is_tpl = _token_template_mask(tokenizer, rendered, query, cfg)
        if is_tpl.numel() == 0 or not bool((~is_tpl).any()):
            # All-template prompt (query not found inside rendered text) —
            # nothing meaningful to measure; skip.
            continue
        input_ids = input_ids.to(device).unsqueeze(0)
        attn_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_attentions=True, use_cache=False, return_dict=True)
        # out.attentions: tuple of len n_layers, each (1, n_heads, q_len, k_len)
        is_tpl_dev = is_tpl.to(device)
        # For each query position, total attention on template keys.
        # Average across query positions, but exclude query positions that
        # are themselves templated (so we measure how content reads template,
        # not how template reads template).
        non_tpl_q = (~is_tpl_dev).nonzero(as_tuple=False).squeeze(-1)
        if non_tpl_q.numel() == 0:
            continue
        denom_pos = float(non_tpl_q.numel())
        for layer_idx, attn in enumerate(out.attentions):
            # attn: (1, heads, q_len, k_len) — restrict to non-template q rows.
            a = attn[0].index_select(1, non_tpl_q)            # (heads, n_q_nt, k_len)
            tpl_mass = a[:, :, is_tpl_dev].sum(dim=-1)        # (heads, n_q_nt)
            per_head = tpl_mass.mean(dim=-1).double().cpu()   # (heads,)
            sums[layer_idx] += per_head
            counts[layer_idx] += 1.0
        # free for next prompt
        del out

    out_dict: dict[tuple[int, int], float] = {}
    for li in range(n_layers):
        if counts[li, 0] <= 0:
            continue
        per_head = (sums[li] / counts[li]).tolist()
        for hi, v in enumerate(per_head):
            out_dict[(li, hi)] = float(v)
    return out_dict


def residualize_on_template_anchoring(
    ranking: Sequence[Mapping],
    anchoring: Mapping[tuple[int, int], float],
    score_key: str = "score",
) -> list[dict]:
    """OLS-residualise a head ranking on rho_tpl.

    Parameters
    ----------
    ranking
        Iterable of dicts with at least ``{layer, head, <score_key>}``.
    anchoring
        Mapping ``(layer, head) -> rho_tpl`` (output of
        :func:`compute_head_template_anchoring`).
    score_key
        Field name carrying the discovery score (default ``"score"``).

    Returns
    -------
    list[dict]
        Copy of ``ranking`` with three added fields per row:
        ``rho_tpl``, ``score_resid`` (raw score minus OLS prediction from
        rho_tpl), ``rank_resid`` (1-based rank by descending residual,
        ties broken by original order).

    Notes
    -----
    Uses plain least-squares with intercept; no scipy/statsmodels dependency.
    Heads missing from ``anchoring`` get ``rho_tpl=None`` and are excluded
    from the OLS fit but appear in the output with ``score_resid=None``.
    """
    rows = [dict(r) for r in ranking]
    xs: list[float] = []
    ys: list[float] = []
    idx_fit: list[int] = []
    for i, r in enumerate(rows):
        key = (int(r["layer"]), int(r["head"]))
        rho = anchoring.get(key)
        r["rho_tpl"] = rho
        if rho is None or score_key not in r:
            continue
        xs.append(float(rho))
        ys.append(float(r[score_key]))
        idx_fit.append(i)

    if len(idx_fit) < 2:
        for r in rows:
            r["score_resid"] = None
            r["rank_resid"] = None
        return rows

    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    slope = (cov / var).item() if var.item() > 0 else 0.0
    intercept = (y_mean - slope * x_mean).item()

    for r in rows:
        rho = r.get("rho_tpl")
        if rho is None or score_key not in r:
            r["score_resid"] = None
            continue
        pred = intercept + slope * float(rho)
        r["score_resid"] = float(r[score_key]) - pred

    ranked = sorted(
        [(i, r["score_resid"]) for i, r in enumerate(rows) if r["score_resid"] is not None],
        key=lambda t: (-t[1], t[0]),
    )
    rank_map = {i: pos + 1 for pos, (i, _) in enumerate(ranked)}
    for i, r in enumerate(rows):
        r["rank_resid"] = rank_map.get(i)
    return rows


__all__ = [
    "TemplateAnchoringConfig",
    "compute_head_template_anchoring",
    "residualize_on_template_anchoring",
]
