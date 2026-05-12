"""The 2D `(action_quality, cot_content)` canonical example.

This is the running example used in the README, theory doc, paper, and
notebooks 02 and 04. The action space is `[0, 1]^2`:

    action[0]  -- action_quality, what the user actually cares about
    action[1]  -- cot_content,    what shows up in the model's chain of
                                  thought

A *ground-truth* task reward depends only on action_quality (peaked at
0.8). A *CoT-aware* scorer (the LLM judge with CoT access) depends on
both axes: it likes high quality but also assigns a spurious bonus to a
specific CoT pattern (peaked at cot_content = 0.6). A *CoT-blind*
scorer (an exact-match correctness check) depends only on
action_quality.

Under a vanilla weighted sum, the policy drifts toward the CoT-content
ridge because that surface has a steeper gradient than the
action-quality surface. Under a `DifferentialCap`, the policy update is
restricted to the s_blind direction up to a budget `delta`, which keeps
the CoT distribution from collapsing onto the spurious ridge while
still letting the aware scorer contribute when it agrees with the
blind one.

Five composition schemes are exposed: weighted_sum, gated, capped,
gated_capped, gated_capped_differential. The training loop and the
plotting hooks live alongside the scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.composition import (
    CompositionSpec,
    Compositor,
    DifferentialCap,
    Gate,
    Scorer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground truth and scorers
# ---------------------------------------------------------------------------


def task_reward(a: npt.NDArray[np.floating[Any]]) -> float:
    """What we actually want the policy to optimize.

    Peaked at action_quality = 0.8; flat in the cot_content axis.
    """
    q = float(a[0])
    return float(np.exp(-8.0 * (q - 0.8) ** 2))


def s_blind(a: npt.NDArray[np.floating[Any]]) -> float:
    """A CoT-blind scorer: exact-match-style correctness on quality.

    Same as task reward in this toy; in real settings it is a weaker but
    unbiased proxy (programmatic correctness check on the final answer).
    """
    q = float(a[0])
    return float(np.exp(-8.0 * (q - 0.8) ** 2))


def s_aware(a: npt.NDArray[np.floating[Any]]) -> float:
    """A CoT-reading scorer.

    Likes high action_quality, but also assigns a spurious +0.9 bonus to a
    specific CoT pattern (cot_content ~ 0.75) with a much steeper gradient
    than the quality term. The bonus is the information-only-in-CoT signal
    that gradient descent will hunt for if it is left unbounded.

    The CoT-ridge gradient is intentionally larger than the
    action_quality gradient near the policy's initial position so that
    under a weighted sum the optimizer is dominated by the CoT signal in
    early training, producing a clearly drifted final policy.
    """
    q = float(a[0])
    c = float(a[1])
    quality_term = float(np.exp(-6.0 * (q - 0.8) ** 2))
    cot_ridge = 0.9 * float(np.exp(-40.0 * (c - 0.75) ** 2))
    return quality_term + cot_ridge


def format_gate(a: npt.NDArray[np.floating[Any]]) -> float:
    """A coarse format gate: the answer must not be empty.

    Implemented as a sigmoid on action_quality so the gate is orthogonal
    to the CoT axis (a format check on the final answer says nothing
    about CoT content). Returns a soft pass value in [0, 1].
    """
    q = float(a[0])
    return 1.0 / (1.0 + float(np.exp(-15.0 * (q - 0.05))))


# ---------------------------------------------------------------------------
# Composition schemes
# ---------------------------------------------------------------------------


Scheme = Literal[
    "weighted_sum",
    "gated",
    "capped",
    "gated_capped",
    "gated_capped_differential",
]

ALL_SCHEMES: tuple[Scheme, ...] = (
    "weighted_sum",
    "gated",
    "capped",
    "gated_capped",
    "gated_capped_differential",
)


def build_spec(
    scheme: Scheme,
    cap: float = 1.0,
    delta: float = 0.1,
    gate_penalty: float = -1.0,
) -> CompositionSpec:
    """Construct a CompositionSpec for one of the five schemes.

    The five schemes are layered on top of one another so the visual
    comparison in notebook 02 isolates the effect of each structural
    property.
    """
    if scheme == "weighted_sum":
        # No gate, no cap. Reward = s_aware(x). The unstructured baseline.
        return CompositionSpec(
            gates=[],
            scorers=[Scorer(name="aware", fn=s_aware, cap=float("inf"),
                            reads_cot=True)],
            reward_bounds=(-2.0, 2.0),
            gate_penalty=gate_penalty,
        )
    if scheme == "gated":
        # Gate present, no cap on the aware scorer.
        return CompositionSpec(
            gates=[Gate(name="format", predicate=format_gate, threshold=0.5)],
            scorers=[Scorer(name="aware", fn=s_aware, cap=float("inf"),
                            reads_cot=True)],
            reward_bounds=(-2.0, 2.0),
            gate_penalty=gate_penalty,
        )
    if scheme == "capped":
        # Cap on the aware scorer, no gate.
        return CompositionSpec(
            gates=[],
            scorers=[Scorer(name="aware", fn=s_aware, cap=cap,
                            reads_cot=True)],
            reward_bounds=(-2.0, 2.0),
            gate_penalty=gate_penalty,
        )
    if scheme == "gated_capped":
        # Gate plus cap on the aware scorer.
        return CompositionSpec(
            gates=[Gate(name="format", predicate=format_gate, threshold=0.5)],
            scorers=[Scorer(name="aware", fn=s_aware, cap=cap,
                            reads_cot=True)],
            reward_bounds=(-2.0, 2.0),
            gate_penalty=gate_penalty,
        )
    if scheme == "gated_capped_differential":
        # Gate plus DifferentialCap wrapping aware/blind with budget delta.
        return CompositionSpec(
            gates=[Gate(name="format", predicate=format_gate, threshold=0.5)],
            scorers=[
                DifferentialCap(
                    name="aware_minus_blind",
                    fn_aware=s_aware,
                    fn_blind=s_blind,
                    cap=cap,
                    delta=delta,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
            gate_penalty=gate_penalty,
        )
    raise ValueError(f"unknown scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# 2D REINFORCE training loop
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training-loop configuration for the 2D toy."""

    n_steps: int = 1000
    n_samples: int = 64
    lr: float = 0.05
    sigma: float = 0.12
    seed: int = 42
    init_mu: tuple[float, float] = (0.4, 0.3)


@dataclass
class TrainHistory:
    """Trajectory data returned by `run_cot_drift`."""

    step: list[int] = field(default_factory=list)
    policy_mu: list[npt.NDArray[np.floating[Any]]] = field(default_factory=list)
    mean_reward: list[float] = field(default_factory=list)
    mean_task_reward: list[float] = field(default_factory=list)
    mean_blind: list[float] = field(default_factory=list)
    mean_aware: list[float] = field(default_factory=list)
    sampled_actions: list[npt.NDArray[np.floating[Any]]] = field(
        default_factory=list
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": list(self.step),
            "policy_mu": [np.asarray(m) for m in self.policy_mu],
            "mean_reward": list(self.mean_reward),
            "mean_task_reward": list(self.mean_task_reward),
            "mean_blind": list(self.mean_blind),
            "mean_aware": list(self.mean_aware),
            "sampled_actions": [np.asarray(a) for a in self.sampled_actions],
        }


def run_cot_drift(
    scheme: Scheme,
    config: TrainConfig | None = None,
    cap: float = 1.0,
    delta: float = 0.1,
    snapshot_every: int = 100,
) -> TrainHistory:
    """Train a 2D independent-Gaussian policy under one of the schemes.

    The policy is `pi(a|x) = N(mu, sigma^2 * I)` with `mu in R^2`,
    `sigma` fixed. REINFORCE with baseline-subtracted advantages on a
    batch of `n_samples` per step. Returns trajectory data for plotting
    in notebook 02 and ablations in notebook 04.
    """
    cfg = config or TrainConfig()
    spec = build_spec(scheme, cap=cap, delta=delta)
    compositor = Compositor(spec)

    rng = np.random.RandomState(cfg.seed)
    mu = np.array(cfg.init_mu, dtype=float)
    sigma = cfg.sigma

    history = TrainHistory()

    for step in range(cfg.n_steps):
        actions = rng.normal(mu, sigma, size=(cfg.n_samples, 2))
        actions = np.clip(actions, 0.0, 1.0)

        rewards = np.zeros(cfg.n_samples, dtype=float)
        blind_vals = np.zeros(cfg.n_samples, dtype=float)
        aware_vals = np.zeros(cfg.n_samples, dtype=float)
        task_vals = np.zeros(cfg.n_samples, dtype=float)

        for i in range(cfg.n_samples):
            result = compositor.compose(actions[i], step=step)
            rewards[i] = result.reward
            blind_vals[i] = s_blind(actions[i])
            aware_vals[i] = s_aware(actions[i])
            task_vals[i] = task_reward(actions[i])

        advantages = rewards - rewards.mean()
        # grad w.r.t. mu for N(mu, sigma^2 I) is (a - mu) / sigma^2
        grad = (advantages[:, None] * (actions - mu[None, :])).sum(axis=0)
        grad = grad / (cfg.n_samples * sigma**2)
        mu = np.clip(mu + cfg.lr * grad, 0.0, 1.0)

        history.step.append(step)
        history.policy_mu.append(mu.copy())
        history.mean_reward.append(float(rewards.mean()))
        history.mean_task_reward.append(float(task_vals.mean()))
        history.mean_blind.append(float(blind_vals.mean()))
        history.mean_aware.append(float(aware_vals.mean()))

        if step % snapshot_every == 0 or step == cfg.n_steps - 1:
            history.sampled_actions.append(actions.copy())

    return history


# ---------------------------------------------------------------------------
# Reward-hacking and monitorability metrics
# ---------------------------------------------------------------------------


def reward_hacking_rate(
    history: TrainHistory,
    hack_threshold: float = 0.5,
) -> float:
    """Fraction of recent samples where aware-blind gap exceeds threshold.

    The 2D analog of the LLM-judge-says-good-but-checker-says-wrong rate
    used in `experiments/qwen_gsm8k/`. A sample is "hacking" when the
    aware scorer rates it well but the blind scorer disagrees by more
    than `hack_threshold`.
    """
    if not history.sampled_actions:
        return 0.0
    last = history.sampled_actions[-1]
    aware = np.array([s_aware(a) for a in last])
    blind = np.array([s_blind(a) for a in last])
    return float((aware - blind > hack_threshold).mean())


def cot_drift_distance(history: TrainHistory) -> float:
    """L2 distance between final cot_content mean and its initial value.

    A monitorability proxy: a larger drift means the policy has moved
    more along the CoT axis, which corresponds to a less monitorable
    final distribution in the qwen_gsm8k metric.
    """
    if not history.policy_mu:
        return 0.0
    return float(abs(history.policy_mu[-1][1] - history.policy_mu[0][1]))


__all__ = [
    "Scheme",
    "ALL_SCHEMES",
    "TrainConfig",
    "TrainHistory",
    "task_reward",
    "s_blind",
    "s_aware",
    "format_gate",
    "build_spec",
    "run_cot_drift",
    "reward_hacking_rate",
    "cot_drift_distance",
]
