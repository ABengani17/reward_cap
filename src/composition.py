"""Reward composition as a constraint-projection layer.

Implements the formal object (G, S, c, m) from docs/theory.md: a tuple of
gates, scorers, per-scorer caps, and per-scorer monitors. The composed
reward is

    R(x) = [all gates pass] * sum_s min(c_s, s(x))

with an optional DifferentialCap on CoT-reading scorers that decomposes
s_aware = s_blind + (s_aware - s_blind) and clips the second term to a
monitorability budget delta.

The module exposes two layers:

  * New API (preferred): Gate, Scorer, DifferentialCap, Monitor,
    CompositionSpec, Compositor. Strictly typed; built around a
    declarative spec.
  * Legacy API: RewardComponent, ComponentType, GatedCompositor,
    WeightedSumCompositor, CompositionMonitor, run_bandit. Preserved so
    that the original 1D-bandit scenarios and tests continue to run
    unchanged. All legacy paths internally convert to the new API.

Public functions are type-annotated for mypy --strict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Actions can be scalars (1D bandit) or vectors (2D toy and beyond).
# Callable types accept Any so that scenario authors are free to write
# scorers narrowly typed on float or on ndarray; the compositor never
# inspects the action shape, it just hands it through.
Action = float | npt.NDArray[np.floating[Any]]
ScalarFn = Callable[..., float]
PredicateFn = Callable[..., float]
ValidatorFn = Callable[[Sequence[float], int], bool]


# ---------------------------------------------------------------------------
# New typed primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """A binary prerequisite.

    `predicate(x)` returns a real number. The gate is satisfied when the
    returned value is at least `threshold`. Failure of any gate
    short-circuits the composed reward to `gate_penalty`.
    """

    name: str
    predicate: PredicateFn
    threshold: float = 0.5


@dataclass(frozen=True)
class Scorer:
    """A scalar quality signal with a hard upper cap.

    The cap `c_s` is the maximum contribution this scorer can make to the
    composed reward per sample. See docs/theory.md Section 2 for how to
    pick `cap` from a policy-drift budget.
    """

    name: str
    fn: ScalarFn
    cap: float = 1.0
    reads_cot: bool = False
    validator: ValidatorFn | None = None


@dataclass(frozen=True)
class DifferentialCap:
    """A CoT-reading scorer with a monitorability budget.

    Composes a CoT-aware scorer `fn_aware` and a CoT-blind companion
    `fn_blind` into

        min(cap, fn_blind(x) + clip(fn_aware(x) - fn_blind(x), -delta, delta))

    where delta is the per-sample monitorability budget. See docs/theory.md
    Section 3 for the soundness statement.
    """

    name: str
    fn_aware: ScalarFn
    fn_blind: ScalarFn
    cap: float = 1.0
    delta: float = 0.1
    reads_cot: bool = True
    validator: ValidatorFn | None = None


@dataclass(frozen=True)
class Monitor:
    """A health predicate attached to a named scorer.

    The predicate takes the rolling window of recent scores and the current
    step, and returns True when the scorer is still healthy. Falsy returns
    are surfaced as violations.
    """

    name: str
    target: str
    predicate: ValidatorFn


ScorerLike = Scorer | DifferentialCap


@dataclass
class CompositionSpec:
    """Declarative composition spec consumed by Compositor and audit."""

    gates: list[Gate] = field(default_factory=list)
    scorers: list[ScorerLike] = field(default_factory=list)
    monitors: list[Monitor] = field(default_factory=list)
    reward_bounds: tuple[float, float] = (-1.0, 1.0)
    gate_penalty: float = -1.0

    @property
    def scorer_names(self) -> list[str]:
        return [s.name for s in self.scorers]

    @property
    def gate_names(self) -> list[str]:
        return [g.name for g in self.gates]

    @property
    def cap_sum(self) -> float:
        return float(sum(s.cap for s in self.scorers))

    @property
    def gate_magnitude(self) -> float:
        return float(abs(self.gate_penalty))


@dataclass
class RewardResult:
    """Return type of `Compositor.compose`."""

    reward: float
    scores: dict[str, float]
    capped: dict[str, float]
    gates_pass: bool
    violations: list[dict[str, Any]]
    step: int


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------


def _eval_scorer(s: ScorerLike, action: Action) -> tuple[float, float]:
    """Return (raw_score, capped_score) for either scorer flavor."""
    if isinstance(s, DifferentialCap):
        v_aware = float(s.fn_aware(action))
        v_blind = float(s.fn_blind(action))
        residual = float(np.clip(v_aware - v_blind, -s.delta, s.delta))
        raw = v_blind + residual
        capped = float(min(s.cap, raw))
        return raw, capped
    raw = float(s.fn(action))
    capped = float(min(s.cap, raw))
    return raw, capped


class Compositor:
    """Composes a stack of scorers under the spec into a single reward.

    See docs/theory.md Section 1 for the formal definition. The compositor
    is stateless across calls; the optional `MonitorState` it owns tracks
    rolling windows of scores for health-checking.
    """

    def __init__(self, spec: CompositionSpec) -> None:
        self.spec = spec
        self._history: dict[str, list[float]] = {
            s.name: [] for s in spec.scorers
        }
        self._step: int = 0

    def _record(self, scores: Mapping[str, float], window: int = 200) -> None:
        for name, value in scores.items():
            buf = self._history.setdefault(name, [])
            buf.append(value)
            if len(buf) > window:
                buf.pop(0)

    def _check_monitors(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for mon in self.spec.monitors:
            buf = self._history.get(mon.target, [])
            if not mon.predicate(buf, self._step):
                out.append(
                    {
                        "type": "monitor",
                        "monitor": mon.name,
                        "target": mon.target,
                        "step": self._step,
                    }
                )
        for s in self.spec.scorers:
            if s.validator is None:
                continue
            buf = self._history.get(s.name, [])
            if not s.validator(buf, self._step):
                out.append(
                    {
                        "type": "degraded",
                        "scorer": s.name,
                        "step": self._step,
                    }
                )
        return out

    def compose(self, action: Action, step: int | None = None) -> RewardResult:
        """Apply the composition function R(x) to a single action."""
        if step is not None:
            self._step = step
        else:
            self._step += 1

        gate_scores: dict[str, float] = {}
        for g in self.spec.gates:
            gate_scores[g.name] = float(g.predicate(action))
        gates_pass = all(
            gate_scores[g.name] >= g.threshold for g in self.spec.gates
        )

        raw_scores: dict[str, float] = {}
        capped_scores: dict[str, float] = {}
        for s in self.spec.scorers:
            raw, capped = _eval_scorer(s, action)
            raw_scores[s.name] = raw
            capped_scores[s.name] = capped

        if not gates_pass:
            reward = self.spec.gate_penalty
        else:
            reward = float(sum(capped_scores.values()))

        lo, hi = self.spec.reward_bounds
        reward = float(np.clip(reward, lo, hi))

        self._record(capped_scores)
        violations = self._check_monitors()

        return RewardResult(
            reward=reward,
            scores={**gate_scores, **raw_scores},
            capped=capped_scores,
            gates_pass=gates_pass,
            violations=violations,
            step=self._step,
        )


# ---------------------------------------------------------------------------
# Legacy primitives kept for backward compatibility
# ---------------------------------------------------------------------------


class ComponentType(Enum):
    GATE = "gate"
    SCORER = "scorer"


@dataclass
class RewardComponent:
    """One reward signal in the legacy 1D-bandit API.

    A gate is a binary prerequisite. A scorer is a scalar quality signal.
    Retained so existing scenarios and tests continue to import this name;
    new code should use `Gate`, `Scorer`, or `DifferentialCap` directly.
    """

    name: str
    fn: ScalarFn
    component_type: ComponentType = ComponentType.SCORER
    weight: float = 1.0
    influence_cap: float = 2.0
    independent_of: list[str] = field(default_factory=list)
    validator: ValidatorFn | None = None
    gate_threshold: float = 0.5


def spec_from_components(
    components: Sequence[RewardComponent],
    reward_bounds: tuple[float, float] = (-1.0, 1.0),
    gate_penalty: float = -1.0,
) -> CompositionSpec:
    """Convert a legacy component list to a CompositionSpec."""
    gates: list[Gate] = []
    scorers: list[ScorerLike] = []
    for c in components:
        if c.component_type is ComponentType.GATE:
            gates.append(
                Gate(name=c.name, predicate=c.fn, threshold=c.gate_threshold)
            )
        else:
            def _scaled(a: Any, *, _c: RewardComponent = c) -> float:
                return float(_c.weight) * float(_c.fn(a))

            scorers.append(
                Scorer(
                    name=c.name,
                    fn=_scaled,
                    cap=float(c.weight) * float(c.influence_cap),
                    validator=c.validator,
                )
            )
    return CompositionSpec(
        gates=gates,
        scorers=scorers,
        reward_bounds=reward_bounds,
        gate_penalty=gate_penalty,
    )


class CompositionMonitor:
    """Legacy monitor on a list of RewardComponents.

    Watches three patterns over a rolling window of recent scores:
        1. Contribution dominance (a scorer exceeding its nominal share).
        2. Correlation between scorers declared independent.
        3. Validator-predicate failure (signal degradation).
    """

    def __init__(
        self,
        components: Sequence[RewardComponent],
        window: int = 200,
    ) -> None:
        self.components: dict[str, RewardComponent] = {
            c.name: c for c in components
        }
        self.window = window
        self.history: dict[str, list[float]] = {c.name: [] for c in components}
        self.violations: list[dict[str, Any]] = []
        self.step = 0

    def record(self, scores: Mapping[str, float]) -> list[dict[str, Any]]:
        self.step += 1
        new_violations: list[dict[str, Any]] = []

        for name, score in scores.items():
            buf = self.history[name]
            buf.append(score)
            if len(buf) > self.window:
                buf.pop(0)

        new_violations.extend(self._check_dominance(scores))
        new_violations.extend(self._check_correlation())
        new_violations.extend(self._check_validators())

        self.violations.extend(new_violations)
        return new_violations

    def _check_dominance(
        self, scores: Mapping[str, float]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        scorers = [
            c
            for c in self.components.values()
            if c.component_type is ComponentType.SCORER
        ]
        total = sum(
            abs(scores.get(c.name, 0.0) * c.weight) for c in scorers
        )
        if total < 1e-10:
            return out

        weight_sum = sum(c.weight for c in scorers)
        for comp in scorers:
            actual = abs(scores.get(comp.name, 0.0) * comp.weight) / total
            nominal = comp.weight / weight_sum
            if actual > nominal * comp.influence_cap:
                out.append(
                    {
                        "type": "dominance",
                        "component": comp.name,
                        "step": self.step,
                        "actual_share": round(actual, 4),
                        "nominal_share": round(nominal, 4),
                    }
                )
        return out

    def _check_correlation(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for comp in self.components.values():
            for other_name in comp.independent_of:
                if other_name not in self.history:
                    continue
                h1, h2 = self.history[comp.name], self.history[other_name]
                n = min(len(h1), len(h2))
                if n < 30:
                    continue
                a1 = np.asarray(h1[-n:], dtype=float)
                a2 = np.asarray(h2[-n:], dtype=float)
                if float(a1.std()) < 1e-10 or float(a2.std()) < 1e-10:
                    continue
                r = float(np.corrcoef(a1, a2)[0, 1])
                if abs(r) > 0.5:
                    out.append(
                        {
                            "type": "correlation",
                            "components": (comp.name, other_name),
                            "step": self.step,
                            "r": round(r, 4),
                        }
                    )
        return out

    def _check_validators(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for comp in self.components.values():
            if comp.validator is None:
                continue
            if not comp.validator(self.history[comp.name], self.step):
                out.append(
                    {
                        "type": "degraded",
                        "component": comp.name,
                        "step": self.step,
                    }
                )
        return out

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for v in self.violations:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
        return {
            "steps": self.step,
            "total_violations": len(self.violations),
            "by_type": counts,
            "violations": self.violations,
        }


class GatedCompositor:
    """Legacy gated compositor on RewardComponent lists.

    Same semantics as the original prototype: gates short-circuit to a
    penalty; scorers aggregate as a bounded weighted mean of clipped
    contributions, then the total is clipped to `reward_bounds`. Used by
    1D-bandit scenarios; new work should prefer `Compositor`.
    """

    def __init__(
        self,
        components: Sequence[RewardComponent],
        reward_bounds: tuple[float, float] = (-1.0, 1.0),
        gate_penalty: float = -1.0,
        monitor: bool = True,
    ) -> None:
        self.gates = [
            c for c in components if c.component_type is ComponentType.GATE
        ]
        self.scorers = [
            c for c in components if c.component_type is ComponentType.SCORER
        ]
        self.reward_min, self.reward_max = reward_bounds
        self.gate_penalty = gate_penalty
        self.monitor: CompositionMonitor | None = (
            CompositionMonitor(list(components)) if monitor else None
        )

    def compose(self, action: float, step: int = 0) -> dict[str, Any]:
        scores: dict[str, float] = {}
        for comp in [*self.gates, *self.scorers]:
            scores[comp.name] = float(comp.fn(action))

        gates_pass = all(
            scores[g.name] >= g.gate_threshold for g in self.gates
        )

        reward: float
        if not gates_pass:
            reward = self.gate_penalty
        else:
            total_weight = sum(c.weight for c in self.scorers)
            if total_weight < 1e-10:
                reward = 0.0
            else:
                agg = 0.0
                for comp in self.scorers:
                    raw = scores[comp.name] * comp.weight
                    cap = comp.weight * comp.influence_cap
                    agg += float(np.clip(raw, -cap, cap))
                reward = agg / total_weight

        reward = float(np.clip(reward, self.reward_min, self.reward_max))

        violations: list[dict[str, Any]] = []
        if self.monitor is not None:
            violations = self.monitor.record(scores)

        return {
            "reward": reward,
            "scores": scores,
            "gates_pass": gates_pass,
            "violations": violations,
            "step": step,
        }


class WeightedSumCompositor:
    """Legacy weighted-sum compositor with no constraints.

    The unstructured baseline. Used in notebook 01 to demonstrate
    failure modes; not recommended for any real pipeline.
    """

    def __init__(self, components: Sequence[RewardComponent]) -> None:
        self.components = list(components)

    def compose(self, action: float, step: int = 0) -> dict[str, Any]:
        scores: dict[str, float] = {
            c.name: float(c.fn(action)) for c in self.components
        }
        reward = sum(
            scores[c.name] * c.weight for c in self.components
        )
        return {
            "reward": float(reward),
            "scores": scores,
            "gates_pass": True,
            "violations": [],
            "step": step,
        }


# ---------------------------------------------------------------------------
# 1D bandit training loop (legacy entry point used by tests and notebooks)
# ---------------------------------------------------------------------------


def run_bandit(
    compositor: GatedCompositor | WeightedSumCompositor,
    n_steps: int = 2000,
    n_samples: int = 50,
    lr: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """REINFORCE on a 1D Gaussian policy against a legacy compositor.

    Deliberately minimal: composition dynamics do not depend on model
    complexity, so the bandit makes the reward landscape visible. The 2D
    canonical example lives in `src.scenarios.cot_drift`.
    """
    rng = np.random.RandomState(seed)
    mu: float = 0.5
    sigma: float = 0.15

    if isinstance(compositor, GatedCompositor):
        all_components: list[RewardComponent] = [
            *compositor.gates,
            *compositor.scorers,
        ]
    else:
        all_components = list(compositor.components)

    history: dict[str, Any] = {
        "step": [],
        "mean_reward": [],
        "policy_mu": [],
        "scores": {c.name: [] for c in all_components},
        "violations": [],
    }

    for step in range(n_steps):
        actions = np.clip(
            rng.normal(mu, sigma, size=n_samples), 0.0, 1.0
        )
        rewards = np.zeros(n_samples)
        step_scores: dict[str, list[float]] = {
            c.name: [] for c in all_components
        }

        for i, a in enumerate(actions):
            result = compositor.compose(float(a), step=step)
            rewards[i] = result["reward"]
            for c in all_components:
                step_scores[c.name].append(
                    float(result["scores"].get(c.name, 0.0))
                )
            history["violations"].extend(result["violations"])

        advantages = rewards - rewards.mean()
        grad = float(
            np.sum(advantages * (actions - mu)) / (n_samples * sigma**2)
        )
        mu = float(np.clip(mu + lr * grad, 0.0, 1.0))

        history["step"].append(step)
        history["mean_reward"].append(float(rewards.mean()))
        history["policy_mu"].append(mu)
        for c in all_components:
            history["scores"][c.name].append(
                float(np.mean(step_scores[c.name]))
            )

    return history


__all__ = [
    # new API
    "Action",
    "ScalarFn",
    "PredicateFn",
    "ValidatorFn",
    "Gate",
    "Scorer",
    "DifferentialCap",
    "Monitor",
    "ScorerLike",
    "CompositionSpec",
    "RewardResult",
    "Compositor",
    # legacy API
    "ComponentType",
    "RewardComponent",
    "CompositionMonitor",
    "GatedCompositor",
    "WeightedSumCompositor",
    "run_bandit",
    "spec_from_components",
]
