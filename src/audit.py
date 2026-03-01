"""
Static audit for reward composition configurations.

Takes a list of reward components and checks for structural issues
before training starts. Flags mistyped gates (prerequisites treated
as additive scores), variance imbalance between scorers, missing
health validators, and undeclared correlations.

Run this the same way you run a linter: before the expensive part.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from src.composition import RewardComponent, ComponentType


@dataclass
class Finding:
    check: str
    ok: bool
    severity: str  # "pass", "warn", "fail"
    detail: str


def audit(components: list[RewardComponent], n_samples: int = 1000,
          seed: int = 42) -> dict:
    """Run all checks on a composition config. Returns a report dict."""
    rng = np.random.RandomState(seed)
    findings = []

    findings.append(_check_gate_typing(components))
    findings.append(_check_variance(components, rng, n_samples))
    findings.append(_check_correlation(components, rng, n_samples))
    findings.append(_check_validators(components))
    findings.append(_check_bounds(components, rng, n_samples))

    recommendations = [
        f"[{f.severity.upper()}] {f.check} - {f.detail}"
        for f in findings if f.severity != "pass"
    ]

    return {
        "n_components": len(components),
        "n_gates": sum(1 for c in components if c.component_type == ComponentType.GATE),
        "n_scorers": sum(1 for c in components if c.component_type == ComponentType.SCORER),
        "findings": findings,
        "recommendations": recommendations,
    }


def _check_gate_typing(components: list[RewardComponent]) -> Finding:
    """Prerequisite-sounding components should be gates, not scorers."""
    prereq_keywords = {"format", "safety", "valid", "constraint", "compliance"}
    scorers = [c for c in components if c.component_type == ComponentType.SCORER]
    gates = [c for c in components if c.component_type == ComponentType.GATE]
    mistyped = [c.name for c in scorers
                if any(k in c.name.lower() for k in prereq_keywords)]

    if mistyped:
        return Finding(
            "gate typing", False, "fail",
            f"{mistyped} look like prerequisites but are typed as scorers. "
            f"This creates priority inversion risk. Make them gates."
        )
    if not gates:
        return Finding(
            "gate typing", False, "warn",
            "No gates defined. If any component is a prerequisite "
            "(format, safety), it should be a gate."
        )
    return Finding("gate typing", True, "pass",
                   f"{len(gates)} gate(s), {len(scorers)} scorer(s)")


def _check_variance(components: list[RewardComponent],
                    rng: np.random.RandomState, n: int) -> Finding:
    """High variance ratio between scorers means one will dominate."""
    scorers = [c for c in components if c.component_type == ComponentType.SCORER]
    if len(scorers) < 2:
        return Finding("variance balance", True, "pass", "Fewer than 2 scorers")

    variances = {}
    for c in scorers:
        samples = [c.fn(rng.uniform(0, 1)) for _ in range(n)]
        variances[c.name] = np.var(samples)

    hi = max(variances.values())
    lo = min(v for v in variances.values() if v > 1e-10) if any(v > 1e-10 for v in variances.values()) else hi
    ratio = hi / lo if lo > 1e-10 else float("inf")
    worst = max(variances, key=variances.get)

    if ratio > 10:
        return Finding("variance balance", False, "fail",
                       f"{ratio:.0f}x variance ratio. '{worst}' will dominate. "
                       f"Tighten its influence_cap or normalize.")
    if ratio > 3:
        return Finding("variance balance", False, "warn",
                       f"{ratio:.1f}x variance ratio. '{worst}' may have outsized influence.")
    return Finding("variance balance", True, "pass", f"{ratio:.1f}x ratio (acceptable)")


def _check_correlation(components: list[RewardComponent],
                       rng: np.random.RandomState, n: int) -> Finding:
    """Components declared independent should actually be uncorrelated."""
    pairs = []
    for c in components:
        for other_name in c.independent_of:
            other = next((x for x in components if x.name == other_name), None)
            if other:
                pairs.append((c, other))

    if not pairs:
        return Finding("independence", True, "pass", "No independence constraints declared")

    problems = []
    actions = rng.uniform(0, 1, size=n)
    for c1, c2 in pairs:
        s1 = [c1.fn(a) for a in actions]
        s2 = [c2.fn(a) for a in actions]
        r = np.corrcoef(s1, s2)[0, 1]
        if abs(r) > 0.5:
            problems.append(f"'{c1.name}' and '{c2.name}' (r={r:.2f})")

    if problems:
        return Finding("independence", False, "fail",
                       f"Correlated pairs: {', '.join(problems)}")
    return Finding("independence", True, "pass", f"{len(pairs)} pair(s) checked, all clean")


def _check_validators(components: list[RewardComponent]) -> Finding:
    """At least safety-critical components should have health validators."""
    with_val = [c.name for c in components if c.validator is not None]
    if not with_val:
        return Finding("health monitoring", False, "warn",
                       "No validators set. Signal degradation will go undetected.")
    return Finding("health monitoring", True, "pass",
                   f"Validators on: {with_val}")


def _check_bounds(components: list[RewardComponent],
                  rng: np.random.RandomState, n: int) -> Finding:
    """Check whether the raw weighted sum is well-bounded."""
    from src.composition import WeightedSumCompositor
    comp = WeightedSumCompositor(components)
    rewards = [comp.compose(rng.uniform(0, 1))["reward"] for _ in range(n)]
    lo, hi = min(rewards), max(rewards)
    span = hi - lo

    if span > 10:
        return Finding("reward range", False, "warn",
                       f"Range [{lo:.1f}, {hi:.1f}] is wide. Large swings destabilize training.")
    return Finding("reward range", True, "pass", f"Range [{lo:.2f}, {hi:.2f}]")


def print_report(report: dict):
    """Print a human-readable audit report."""
    print("=" * 55)
    print("COMPOSITION AUDIT")
    print("=" * 55)
    print(f"{report['n_components']} components "
          f"({report['n_gates']} gates, {report['n_scorers']} scorers)")
    print()
    for f in report["findings"]:
        tag = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[f.severity]
        print(f"  [{tag}] {f.check}")
        print(f"         {f.detail}")
        print()
    if report["recommendations"]:
        print("-" * 55)
        for r in report["recommendations"]:
            print(f"  {r}")
    print("=" * 55)
