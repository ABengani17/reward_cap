"""Legacy 1D-bandit scenarios.

Three pairs of broken/fixed `RewardComponent` lists, each scenario
demonstrating one historical composition failure mode. Retained from the
original prototype; the canonical example in the new framework is the 2D
toy in `src.scenarios.cot_drift`.

References for each scenario are in the docstrings; cited from
`docs/theory.md` Section 5.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from src.composition import ComponentType, RewardComponent, ValidatorFn

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Scenario 1: Priority inversion
#
# Based on the Med-RLVR direct-answer-hacking behavior (Zhang et al.,
# 2025, arXiv:2502.19655). A format reward and a correctness reward are
# summed. The format reward has a sharp peak that creates a local
# maximum; the policy converges there and never reaches the correctness
# peak.
# -----------------------------------------------------------------------


def priority_inversion_broken() -> list[RewardComponent]:
    """Format and correctness as additive scorers (the standard approach)."""

    def format_fn(a: float) -> float:
        return float(1.5 * np.exp(-30.0 * (a - 0.3) ** 2))

    def correct_fn(a: float) -> float:
        return float(np.exp(-8.0 * (a - 0.8) ** 2))

    return [
        RewardComponent(
            name="format",
            fn=format_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
    ]


def priority_inversion_fixed() -> list[RewardComponent]:
    """Format as a gate, correctness as the only scorer."""

    def format_fn(a: float) -> float:
        return float(1.0 / (1.0 + np.exp(-20.0 * (a - 0.15))))

    def correct_fn(a: float) -> float:
        return float(np.exp(-8.0 * (a - 0.8) ** 2))

    return [
        RewardComponent(
            name="format",
            fn=format_fn,
            component_type=ComponentType.GATE,
            gate_threshold=0.5,
        ),
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
    ]


# -----------------------------------------------------------------------
# Scenario 2: Gradient dominance
#
# Inspired by DeepSeek-R1 length gaming (DeepSeek-AI, 2025) and
# Wen et al. (2024)'s "U-Sophistry" finding. Three components with
# different variance profiles; the style judge (high variance, steep
# gradient) dominates updates even though its nominal weight is lowest.
# -----------------------------------------------------------------------


def gradient_dominance_broken(
    rng: np.random.RandomState | None = None,
) -> list[RewardComponent]:
    """Three scorers with unequal variance. Style dominates."""
    if rng is None:
        rng = np.random.RandomState(42)
    _rng = rng

    def correct_fn(a: float) -> float:
        return float(a > 0.4) * float(np.exp(-5.0 * (a - 0.7) ** 2))

    def style_fn(a: float) -> float:
        base = float(np.exp(-15.0 * (a - 0.5) ** 2))
        return float(np.clip(base + _rng.normal(0, 0.15), 0.0, 1.0))

    def safety_fn(a: float) -> float:
        return 0.0 if a < 0.05 else 0.9 + 0.1 * a

    return [
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
        RewardComponent(
            name="style",
            fn=style_fn,
            component_type=ComponentType.SCORER,
            weight=0.3,
            independent_of=["correctness"],
        ),
        RewardComponent(
            name="safety",
            fn=safety_fn,
            component_type=ComponentType.SCORER,
            weight=0.5,
        ),
    ]


def gradient_dominance_fixed(
    rng: np.random.RandomState | None = None,
) -> list[RewardComponent]:
    """Same rewards with tighter caps and safety as a gate."""
    if rng is None:
        rng = np.random.RandomState(42)
    _rng = rng

    def correct_fn(a: float) -> float:
        return float(a > 0.4) * float(np.exp(-5.0 * (a - 0.7) ** 2))

    def style_fn(a: float) -> float:
        base = float(np.exp(-15.0 * (a - 0.5) ** 2))
        return float(np.clip(base + _rng.normal(0, 0.15), 0.0, 1.0))

    def safety_fn(a: float) -> float:
        return 0.0 if a < 0.05 else 0.9 + 0.1 * a

    return [
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
            influence_cap=1.5,
        ),
        RewardComponent(
            name="style",
            fn=style_fn,
            component_type=ComponentType.SCORER,
            weight=0.3,
            influence_cap=1.2,
            independent_of=["correctness"],
        ),
        RewardComponent(
            name="safety",
            fn=safety_fn,
            component_type=ComponentType.GATE,
            gate_threshold=0.5,
        ),
    ]


# -----------------------------------------------------------------------
# Scenario 3: Signal degradation
#
# Based on Wen et al. (2024), arXiv:2409.12822: models drift beyond the
# RLHF evaluator's detection boundary over training. A safety classifier
# calibrated at step 0 becomes increasingly permissive.
# -----------------------------------------------------------------------


def signal_degradation_broken(
    degradation_rate: float = 0.0003,
) -> tuple[list[RewardComponent], dict[str, int]]:
    """Safety classifier with no health monitoring."""
    counter: dict[str, int] = {"step": 0}

    def correct_fn(a: float) -> float:
        return float(np.exp(-5.0 * (a - 0.7) ** 2))

    def safety_fn(a: float) -> float:
        acc = max(0.4, 0.95 - degradation_rate * counter["step"])
        return 0.0 if a < 0.5 * acc else 0.9

    def validator(scores: Sequence[float], step: int) -> bool:
        if len(scores) < 50:
            return True
        recent = scores[-50:]
        pass_rate = sum(1 for s in recent if s > 0.8) / len(recent)
        return pass_rate < 0.95

    val: ValidatorFn = validator
    components = [
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
        RewardComponent(
            name="safety",
            fn=safety_fn,
            component_type=ComponentType.SCORER,
            weight=0.5,
            validator=val,
        ),
    ]
    return components, counter


def signal_degradation_fixed(
    degradation_rate: float = 0.0003,
) -> tuple[list[RewardComponent], dict[str, int]]:
    """Same classifier but as a gate with health monitoring."""
    counter: dict[str, int] = {"step": 0}

    def correct_fn(a: float) -> float:
        return float(np.exp(-5.0 * (a - 0.7) ** 2))

    def safety_fn(a: float) -> float:
        acc = max(0.4, 0.95 - degradation_rate * counter["step"])
        return 0.0 if a < 0.5 * acc else 0.9

    def validator(scores: Sequence[float], step: int) -> bool:
        if len(scores) < 50:
            return True
        recent = scores[-50:]
        pass_rate = sum(1 for s in recent if s > 0.8) / len(recent)
        return pass_rate < 0.95

    val: ValidatorFn = validator
    components = [
        RewardComponent(
            name="correctness",
            fn=correct_fn,
            component_type=ComponentType.SCORER,
            weight=1.0,
        ),
        RewardComponent(
            name="safety",
            fn=safety_fn,
            component_type=ComponentType.GATE,
            gate_threshold=0.3,
            validator=val,
        ),
    ]
    return components, counter


__all__ = [
    "priority_inversion_broken",
    "priority_inversion_fixed",
    "gradient_dominance_broken",
    "gradient_dominance_fixed",
    "signal_degradation_broken",
    "signal_degradation_fixed",
]
