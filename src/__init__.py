"""rewardcap - constraint-projection layer for monitorability-preserving
reward composition in LLM post-training.

See README.md for the framing and docs/theory.md for the formal version.
"""

from src.composition import (
    # legacy
    ComponentType,
    CompositionMonitor,
    CompositionSpec,
    Compositor,
    DifferentialCap,
    Gate,
    GatedCompositor,
    Monitor,
    RewardComponent,
    RewardResult,
    Scorer,
    WeightedSumCompositor,
    run_bandit,
    spec_from_components,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Compositor",
    "CompositionSpec",
    "DifferentialCap",
    "Gate",
    "Monitor",
    "RewardResult",
    "Scorer",
    "ComponentType",
    "CompositionMonitor",
    "GatedCompositor",
    "RewardComponent",
    "WeightedSumCompositor",
    "run_bandit",
    "spec_from_components",
]
