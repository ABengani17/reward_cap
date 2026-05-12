"""Reward composition scenarios.

The legacy 1D-bandit scenarios live in `src.scenarios.legacy`. They are
re-exported at the top level of this package so the original tests and
notebook 01 continue to import them as `from src.scenarios import ...`.

The 2D `(action_quality, cot_content)` toy that is the canonical example
in the README and paper lives in `src.scenarios.cot_drift`.
"""

from src.scenarios.legacy import (
    gradient_dominance_broken,
    gradient_dominance_fixed,
    priority_inversion_broken,
    priority_inversion_fixed,
    signal_degradation_broken,
    signal_degradation_fixed,
)

__all__ = [
    "priority_inversion_broken",
    "priority_inversion_fixed",
    "gradient_dominance_broken",
    "gradient_dominance_fixed",
    "signal_degradation_broken",
    "signal_degradation_fixed",
]
