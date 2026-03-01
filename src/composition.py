"""
Reward composition with structural constraints.

Two compositors and a monitor. The gated compositor separates binary
prerequisites (format checks, safety filters) from scalar quality
signals and bounds each component's gradient contribution. The
weighted-sum compositor is the standard baseline with no constraints.
The monitor tracks component health online during training.

See docs/spec.pdf for the formal treatment. This module is the
implementation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class ComponentType(Enum):
    GATE = "gate"
    SCORER = "scorer"


@dataclass
class RewardComponent:
    """One reward signal in a composition pipeline.

    A gate is a binary prerequisite (format compliance, safety check).
    A scorer is a scalar quality signal (correctness, style, helpfulness).
    The distinction matters because additive composition of gates and
    scorers creates priority inversion. See notebooks/01 for the demo.
    """
    name: str
    fn: Callable
    component_type: ComponentType = ComponentType.SCORER
    weight: float = 1.0
    influence_cap: float = 2.0
    independent_of: list[str] = field(default_factory=list)
    validator: Optional[Callable] = None
    gate_threshold: float = 0.5


class CompositionMonitor:
    """Tracks reward component behavior during training.

    Watches for three failure patterns:
      1. A component's effective gradient contribution exceeding its
         nominal share (contribution dominance)
      2. Components declared independent becoming correlated over
         training (shared bias / double-counting)
      3. A component's validation predicate failing (signal degradation,
         typically from distribution shift)

    All checks run on a rolling window of recent scores.
    """

    def __init__(self, components: list[RewardComponent], window: int = 200):
        self.components = {c.name: c for c in components}
        self.window = window
        self.history: dict[str, list[float]] = {c.name: [] for c in components}
        self.violations: list[dict] = []
        self.step = 0

    def record(self, scores: dict[str, float]) -> list[dict]:
        """Log scores from one training step. Returns any new violations."""
        self.step += 1
        new_violations = []

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

    def _check_dominance(self, scores: dict[str, float]) -> list[dict]:
        """Flag when a component contributes more than its cap allows."""
        out = []
        scorers = [c for c in self.components.values()
                    if c.component_type == ComponentType.SCORER]
        total = sum(abs(scores.get(c.name, 0.0) * c.weight) for c in scorers)
        if total < 1e-10:
            return out

        weight_sum = sum(c.weight for c in scorers)
        for comp in scorers:
            actual = abs(scores.get(comp.name, 0.0) * comp.weight) / total
            nominal = comp.weight / weight_sum
            if actual > nominal * comp.influence_cap:
                out.append({
                    "type": "dominance",
                    "component": comp.name,
                    "step": self.step,
                    "actual_share": round(actual, 4),
                    "nominal_share": round(nominal, 4),
                })
        return out

    def _check_correlation(self) -> list[dict]:
        """Flag correlated scores between components declared independent."""
        out = []
        for comp in self.components.values():
            for other_name in comp.independent_of:
                if other_name not in self.history:
                    continue
                h1, h2 = self.history[comp.name], self.history[other_name]
                n = min(len(h1), len(h2))
                if n < 30:
                    continue
                a1, a2 = np.array(h1[-n:]), np.array(h2[-n:])
                if a1.std() < 1e-10 or a2.std() < 1e-10:
                    continue
                r = np.corrcoef(a1, a2)[0, 1]
                if abs(r) > 0.5:
                    out.append({
                        "type": "correlation",
                        "components": (comp.name, other_name),
                        "step": self.step,
                        "r": round(r, 4),
                    })
        return out

    def _check_validators(self) -> list[dict]:
        """Run each component's health check."""
        out = []
        for comp in self.components.values():
            if comp.validator is None:
                continue
            if not comp.validator(self.history[comp.name], self.step):
                out.append({
                    "type": "degraded",
                    "component": comp.name,
                    "step": self.step,
                })
        return out

    def summary(self) -> dict:
        counts = {}
        for v in self.violations:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
        return {
            "steps": self.step,
            "total_violations": len(self.violations),
            "by_type": counts,
            "violations": self.violations,
        }


class GatedCompositor:
    """Composes rewards by separating gates from scorers.

    Gates are evaluated first. If any gate fails, the step gets a
    fixed penalty and scorers are skipped. If all gates pass, scorers
    are aggregated as a bounded weighted mean.

    Each scorer's contribution is clipped to (weight * influence_cap)
    so that high-variance components cannot dominate the gradient
    regardless of their actual output values.

    The total reward is clipped to [reward_min, reward_max].
    """

    def __init__(
        self,
        components: list[RewardComponent],
        reward_bounds: tuple[float, float] = (-1.0, 1.0),
        gate_penalty: float = -1.0,
        monitor: bool = True,
    ):
        self.gates = [c for c in components if c.component_type == ComponentType.GATE]
        self.scorers = [c for c in components if c.component_type == ComponentType.SCORER]
        self.reward_min, self.reward_max = reward_bounds
        self.gate_penalty = gate_penalty
        self.monitor = CompositionMonitor(components) if monitor else None

    def compose(self, action: float, step: int = 0) -> dict:
        scores = {}
        for comp in self.gates + self.scorers:
            scores[comp.name] = comp.fn(action)

        gates_pass = all(scores[g.name] >= g.gate_threshold for g in self.gates)

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
                    agg += np.clip(raw, -cap, cap)
                reward = agg / total_weight

        reward = float(np.clip(reward, self.reward_min, self.reward_max))

        violations = []
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
    """Standard weighted sum. No gates, no bounds, no monitoring.

    This is what most RL training pipelines use today. It exists here
    as the baseline.
    """

    def __init__(self, components: list[RewardComponent]):
        self.components = components

    def compose(self, action: float, step: int = 0) -> dict:
        scores = {c.name: c.fn(action) for c in self.components}
        reward = sum(scores[c.name] * c.weight for c in self.components)
        return {
            "reward": reward,
            "scores": scores,
            "gates_pass": True,
            "violations": [],
            "step": step,
        }


def run_bandit(compositor, n_steps: int = 2000, n_samples: int = 50,
               lr: float = 0.05, seed: int = 42) -> dict:
    """Train a 1D Gaussian bandit against a compositor.

    Simple REINFORCE on a single continuous action in [0, 1]. This is
    deliberately minimal. Composition dynamics do not depend on model
    complexity. We use a bandit because it makes the reward landscape
    visible and the failure modes unambiguous. See the notebooks for
    the full experiments.
    """
    rng = np.random.RandomState(seed)
    mu, sigma = 0.5, 0.15

    all_components = (
        compositor.gates + compositor.scorers
        if hasattr(compositor, "gates")
        else compositor.components
    )
    history = {
        "step": [], "mean_reward": [], "policy_mu": [],
        "scores": {c.name: [] for c in all_components},
        "violations": [],
    }

    for step in range(n_steps):
        actions = np.clip(rng.normal(mu, sigma, size=n_samples), 0.0, 1.0)
        rewards = np.zeros(n_samples)
        step_scores = {c.name: [] for c in all_components}

        for i, a in enumerate(actions):
            result = compositor.compose(a, step=step)
            rewards[i] = result["reward"]
            for c in all_components:
                step_scores[c.name].append(result["scores"].get(c.name, 0.0))
            history["violations"].extend(result["violations"])

        advantages = rewards - rewards.mean()
        grad = np.sum(advantages * (actions - mu)) / (n_samples * sigma ** 2)
        mu = np.clip(mu + lr * grad, 0.0, 1.0)

        history["step"].append(step)
        history["mean_reward"].append(rewards.mean())
        history["policy_mu"].append(mu)
        for c in all_components:
            history["scores"][c.name].append(np.mean(step_scores[c.name]))

    return history
