"""Weighted sub-score aggregation (NOD-24).

NOD-24 spec (Linear):
- AGGREGATE = sum(weight_i * score_i)
- Weights must sum to 1.0 (validated by ScoringConfig)

Risk levels:
- [0.0, 0.25) -> low
- [0.25, 0.50) -> medium
- [0.50, 0.75) -> high
- [0.75, 1.0] -> critical
"""

from __future__ import annotations

from workflow_eval.types import RiskLevel, ScoringConfig, SubScore

_WEIGHT_FIELDS: dict[str, str] = {
    "fan_out": "fan_out",
    "chain_depth": "chain_depth",
    "irreversibility": "irreversibility",
    "centrality": "centrality",
    "spectral": "spectral",
    "compositional": "compositional",
}


def aggregate(sub_scores: tuple[SubScore, ...], config: ScoringConfig) -> float:
    """Compute weighted sum of sub-scores."""
    weights = {name: getattr(config, field) for name, field in _WEIGHT_FIELDS.items()}
    return float(sum(weights[s.name] * s.score for s in sub_scores))


def apply_weights(
    sub_scores: tuple[SubScore, ...], config: ScoringConfig,
) -> tuple[SubScore, ...]:
    """Return new SubScore instances with weights populated from config."""
    weights = {name: getattr(config, field) for name, field in _WEIGHT_FIELDS.items()}
    return tuple(s.model_copy(update={"weight": weights[s.name]}) for s in sub_scores)


def classify_risk(score: float) -> RiskLevel:
    """Map aggregate score to discrete risk level."""
    if score >= 0.75:
        return RiskLevel.CRITICAL
    if score >= 0.50:
        return RiskLevel.HIGH
    if score >= 0.25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
