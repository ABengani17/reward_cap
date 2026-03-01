"""
Simulated reward scenarios that reproduce documented composition failures.

Each function returns a pair: the broken configuration (how production
pipelines actually do it) and the fixed configuration (with gating
and contribution bounds applied). The scenarios are parameterized to
make the failure modes visible in a 1D bandit setting.

References for each scenario are in the docstrings and in the notebooks.
"""

from __future__ import annotations
import numpy as np
from src.composition import RewardComponent, ComponentType


# -----------------------------------------------------------------------
# Scenario 1: Priority inversion
#
# Based on the Med-RLVR "Direct Answer Hacker" (Zhang et al., 2025).
# A format reward and a correctness reward are summed. The format
# reward has a sharp peak that creates a local maximum in the combined
# landscape. The policy converges to the format-satisfying region and
# never reaches the correctness peak.
#
# The fix: treat format as a gate instead of an additive score.
# -----------------------------------------------------------------------

def priority_inversion_broken() -> list[RewardComponent]:
    """Format and correctness as additive scorers (the standard approach).

    The combined landscape peaks at ~0.3 (format dominates) instead
    of 0.8 (where correctness peaks). The policy gets trapped.
    """
    def format_fn(a: float) -> float:
        return 1.5 * np.exp(-30.0 * (a - 0.3) ** 2)

    def correct_fn(a: float) -> float:
        return np.exp(-8.0 * (a - 0.8) ** 2)

    return [
        RewardComponent(name="format", fn=format_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
    ]


def priority_inversion_fixed() -> list[RewardComponent]:
    """Format as a gate, correctness as the only scorer.

    Once format is satisfied (action > 0.15), it contributes nothing
    to the gradient. The optimizer only sees the correctness landscape.
    """
    def format_fn(a: float) -> float:
        return 1.0 / (1.0 + np.exp(-20.0 * (a - 0.15)))

    def correct_fn(a: float) -> float:
        return np.exp(-8.0 * (a - 0.8) ** 2)

    return [
        RewardComponent(name="format", fn=format_fn,
                        component_type=ComponentType.GATE, gate_threshold=0.5),
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
    ]


# -----------------------------------------------------------------------
# Scenario 2: Gradient dominance
#
# Inspired by DeepSeek-R1 length gaming and similar reports of
# high-variance LLM judges pulling training toward their preferences.
#
# Three components with different variance profiles. The style judge
# (high variance, steep gradient) dominates gradient updates even
# though its nominal weight is lowest. The policy converges toward
# the style peak instead of the correctness peak.
#
# The fix: cap each component's contribution and make safety a gate.
# -----------------------------------------------------------------------

def gradient_dominance_broken(rng: np.random.RandomState = None
                              ) -> list[RewardComponent]:
    """Three scorers with unequal variance. Style dominates."""
    if rng is None:
        rng = np.random.RandomState(42)

    def correct_fn(a: float) -> float:
        return float(a > 0.4) * np.exp(-5.0 * (a - 0.7) ** 2)

    def style_fn(a: float) -> float:
        base = np.exp(-15.0 * (a - 0.5) ** 2)
        return np.clip(base + rng.normal(0, 0.15), 0.0, 1.0)

    def safety_fn(a: float) -> float:
        return 0.0 if a < 0.05 else 0.9 + 0.1 * a

    return [
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
        RewardComponent(name="style", fn=style_fn,
                        component_type=ComponentType.SCORER, weight=0.3,
                        independent_of=["correctness"]),
        RewardComponent(name="safety", fn=safety_fn,
                        component_type=ComponentType.SCORER, weight=0.5),
    ]


def gradient_dominance_fixed(rng: np.random.RandomState = None
                             ) -> list[RewardComponent]:
    """Same rewards with tighter caps and safety as a gate."""
    if rng is None:
        rng = np.random.RandomState(42)

    def correct_fn(a: float) -> float:
        return float(a > 0.4) * np.exp(-5.0 * (a - 0.7) ** 2)

    def style_fn(a: float) -> float:
        base = np.exp(-15.0 * (a - 0.5) ** 2)
        return np.clip(base + rng.normal(0, 0.15), 0.0, 1.0)

    def safety_fn(a: float) -> float:
        return 0.0 if a < 0.05 else 0.9 + 0.1 * a

    return [
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0,
                        influence_cap=1.5),
        RewardComponent(name="style", fn=style_fn,
                        component_type=ComponentType.SCORER, weight=0.3,
                        influence_cap=1.2, independent_of=["correctness"]),
        RewardComponent(name="safety", fn=safety_fn,
                        component_type=ComponentType.GATE, gate_threshold=0.5),
    ]


# -----------------------------------------------------------------------
# Scenario 3: Signal degradation
#
# Based on Wen et al. (2024) showing that models learn to produce
# outputs beyond RLHF evaluator detection boundaries over training.
#
# A safety classifier calibrated at step 0 becomes increasingly
# permissive as the policy drifts. The composition keeps reporting
# high safety while actual detection accuracy drops from 0.95 to 0.4.
#
# The fix: attach a validator that checks classifier health and flags
# when it starts rubber-stamping everything.
# -----------------------------------------------------------------------

def signal_degradation_broken(degradation_rate: float = 0.0003):
    """Safety classifier with no health monitoring."""
    counter = {"step": 0}

    def correct_fn(a: float) -> float:
        return np.exp(-5.0 * (a - 0.7) ** 2)

    def safety_fn(a: float) -> float:
        acc = max(0.4, 0.95 - degradation_rate * counter["step"])
        return 0.0 if a < 0.5 * acc else 0.9

    def validator(scores: list[float], step: int) -> bool:
        if len(scores) < 50:
            return True
        recent = scores[-50:]
        pass_rate = sum(1 for s in recent if s > 0.8) / len(recent)
        return pass_rate < 0.95

    components = [
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
        RewardComponent(name="safety", fn=safety_fn,
                        component_type=ComponentType.SCORER, weight=0.5,
                        validator=validator),
    ]
    return components, counter


def signal_degradation_fixed(degradation_rate: float = 0.0003):
    """Same classifier but as a gate with health monitoring."""
    counter = {"step": 0}

    def correct_fn(a: float) -> float:
        return np.exp(-5.0 * (a - 0.7) ** 2)

    def safety_fn(a: float) -> float:
        acc = max(0.4, 0.95 - degradation_rate * counter["step"])
        return 0.0 if a < 0.5 * acc else 0.9

    def validator(scores: list[float], step: int) -> bool:
        if len(scores) < 50:
            return True
        recent = scores[-50:]
        pass_rate = sum(1 for s in recent if s > 0.8) / len(recent)
        return pass_rate < 0.95

    components = [
        RewardComponent(name="correctness", fn=correct_fn,
                        component_type=ComponentType.SCORER, weight=1.0),
        RewardComponent(name="safety", fn=safety_fn,
                        component_type=ComponentType.GATE, gate_threshold=0.3,
                        validator=validator),
    ]
    return components, counter
