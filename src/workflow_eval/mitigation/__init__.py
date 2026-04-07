"""Mitigation — rule-based recommendation engine."""

from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.mitigation.models import MitigationStrategy
from workflow_eval.mitigation.strategies import (
    MitigateCredentialAccess,
    MitigateExternalOps,
    MitigateHighFanOut,
    MitigateHighRiskExternal,
    MitigateIrreversibleOps,
    MitigateUncertainPredecessors,
    MitigateUserFacingOps,
    get_default_strategies,
)

__all__ = [
    "MitigationEngine",
    "MitigateCredentialAccess",
    "MitigateExternalOps",
    "MitigateHighFanOut",
    "MitigateHighRiskExternal",
    "MitigateIrreversibleOps",
    "MitigateUncertainPredecessors",
    "MitigateUserFacingOps",
    "MitigationStrategy",
    "get_default_strategies",
]
