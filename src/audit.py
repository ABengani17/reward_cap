"""Static audit for reward composition configurations.

Runs structural checks on a `CompositionSpec` (or a legacy list of
`RewardComponent`s) before training starts. Surfaces issues that the
theory in `docs/theory.md` predicts will cause monitorability failures.

The audit operates on three new structural rules motivated by the
constraint-projection framing:

  R1. Every CoT-reading scorer must be wrapped in a `DifferentialCap`.
      Rationale: an unbounded CoT-aware reward is exactly the optimization
      pressure Baker et al. (2025) recommend against.

  R2. The composition must contain at least one gate. Rationale: without
      gates, format/safety prerequisites become tradeable against quality
      scorers (the priority-inversion failure mode).

  R3. The sum of scorer caps is bounded by k * |gate_penalty|, k < 1 by
      default. Rationale: see docs/theory.md Section 2.2. Without this,
      the optimizer can pay the gate penalty in exchange for saturating
      every scorer.

The original rules (gate typing, variance balance, declared independence,
validator presence, total range) are preserved and continue to run on the
legacy `RewardComponent` interface.

A pytest fixture `assert_audit_passes` is exposed at the bottom of this
file so any notebook or experiment config that violates the rules will
fail CI.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.composition import (
    ComponentType,
    CompositionSpec,
    DifferentialCap,
    RewardComponent,
    Scorer,
    WeightedSumCompositor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Finding type
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single audit result.

    Severity: "pass" means the rule held; "warn" means a non-fatal
    concern; "fail" means the configuration should not be used.
    """

    check: str
    ok: bool
    severity: str
    detail: str


# ---------------------------------------------------------------------------
# Audit entry point
# ---------------------------------------------------------------------------


def audit(
    config: CompositionSpec | Sequence[RewardComponent],
    n_samples: int = 1000,
    seed: int = 42,
    cap_sum_k: float = 0.5,
) -> dict[str, Any]:
    """Run all checks on a composition config. Returns a report dict.

    Accepts either a new-API `CompositionSpec` or a legacy
    `RewardComponent` list. The new structural rules (R1-R3) run on
    both, with appropriate field interpretation. Legacy-style checks
    (variance, independence, etc.) run when a `RewardComponent` list is
    provided.

    Args:
        config: A `CompositionSpec` or a legacy `RewardComponent` list.
        n_samples: Sample budget for any variance/range probes.
        seed: Seed for the probe RNG. Stamped into the report.
        cap_sum_k: Bound constant for rule R3 (sum of caps <= k * gate
            magnitude). Default 0.5 per docs/theory.md.
    """
    rng = np.random.RandomState(seed)
    findings: list[Finding] = []

    if isinstance(config, CompositionSpec):
        findings.append(_check_diff_cap_for_cot_readers(config))
        findings.append(_check_gate_present(config))
        findings.append(_check_cap_sum(config, k=cap_sum_k))
        meta = {
            "n_components": len(config.gates) + len(config.scorers),
            "n_gates": len(config.gates),
            "n_scorers": len(config.scorers),
        }
    else:
        components = list(config)
        spec = _components_to_spec_view(components)
        findings.append(_check_diff_cap_for_cot_readers(spec))
        findings.append(_check_gate_present(spec))
        findings.append(_check_cap_sum(spec, k=cap_sum_k))
        findings.append(_check_gate_typing(components))
        findings.append(_check_variance(components, rng, n_samples))
        findings.append(_check_correlation(components, rng, n_samples))
        findings.append(_check_validators(components))
        findings.append(_check_bounds(components, rng, n_samples))
        meta = {
            "n_components": len(components),
            "n_gates": sum(
                1 for c in components
                if c.component_type is ComponentType.GATE
            ),
            "n_scorers": sum(
                1 for c in components
                if c.component_type is ComponentType.SCORER
            ),
        }

    recommendations = [
        f"[{f.severity.upper()}] {f.check} - {f.detail}"
        for f in findings
        if f.severity != "pass"
    ]
    ok = all(f.severity != "fail" for f in findings)

    return {
        **meta,
        "ok": ok,
        "seed": seed,
        "findings": findings,
        "recommendations": recommendations,
    }


def _components_to_spec_view(
    components: Sequence[RewardComponent],
) -> CompositionSpec:
    """Build a SpecView of a legacy component list for R1-R3 checking.

    We synthesize a `Scorer` with `cap = weight * influence_cap` for each
    legacy SCORER component, marking `reads_cot=False` by default (the
    legacy API has no notion of CoT). Use the new API if you want R1 to
    catch CoT-reading scorers.
    """
    from src.composition import Gate as NewGate
    from src.composition import Scorer as NewScorer

    gates = [
        NewGate(name=c.name, predicate=c.fn, threshold=c.gate_threshold)
        for c in components
        if c.component_type is ComponentType.GATE
    ]
    scorers: list[Scorer | DifferentialCap] = [
        NewScorer(
            name=c.name,
            fn=c.fn,
            cap=float(c.weight) * float(c.influence_cap),
            reads_cot=False,
        )
        for c in components
        if c.component_type is ComponentType.SCORER
    ]
    return CompositionSpec(gates=gates, scorers=scorers)


# ---------------------------------------------------------------------------
# New rules (R1-R3)
# ---------------------------------------------------------------------------


def _check_diff_cap_for_cot_readers(spec: CompositionSpec) -> Finding:
    """R1: every CoT-reading scorer must be a `DifferentialCap`."""
    unwrapped: list[str] = []
    for s in spec.scorers:
        is_diff = isinstance(s, DifferentialCap)
        reads_cot = getattr(s, "reads_cot", False)
        if reads_cot and not is_diff:
            unwrapped.append(s.name)
    if unwrapped:
        return Finding(
            "differential cap on CoT readers",
            False,
            "fail",
            f"CoT-reading scorers without DifferentialCap: {unwrapped}. "
            "Wrap each in DifferentialCap(fn_aware, fn_blind, delta) to "
            "bound the CoT-conditioned reward signal. See "
            "docs/theory.md Section 3.",
        )
    return Finding(
        "differential cap on CoT readers",
        True,
        "pass",
        "All CoT-reading scorers are wrapped in DifferentialCap.",
    )


def _check_gate_present(spec: CompositionSpec) -> Finding:
    """R2: at least one gate."""
    if not spec.gates:
        return Finding(
            "gate present",
            False,
            "fail",
            "No gates defined. Format and safety prerequisites should "
            "be gates, not additive scorers. See docs/theory.md "
            "Section 1.",
        )
    return Finding(
        "gate present",
        True,
        "pass",
        f"{len(spec.gates)} gate(s) defined: {spec.gate_names}.",
    )


def _check_cap_sum(spec: CompositionSpec, k: float = 0.5) -> Finding:
    """R3: sum of caps must be bounded relative to gate magnitude."""
    finite_caps = [s.cap for s in spec.scorers if np.isfinite(s.cap)]
    has_infinite = any(not np.isfinite(s.cap) for s in spec.scorers)

    if has_infinite:
        return Finding(
            "bounded cap sum",
            False,
            "fail",
            "At least one scorer has cap=inf. The sum of caps cannot be "
            "bounded. Set a finite cap on every scorer. See "
            "docs/theory.md Section 2.1.",
        )

    cap_sum = float(sum(finite_caps))
    gate_mag = spec.gate_magnitude
    if gate_mag <= 0:
        return Finding(
            "bounded cap sum",
            False,
            "warn",
            "gate_penalty is zero or positive; gate dominance is not "
            "enforced.",
        )

    bound = k * gate_mag
    if cap_sum > bound:
        return Finding(
            "bounded cap sum",
            False,
            "fail",
            f"sum(caps)={cap_sum:.3f} exceeds k*|gate_penalty|={bound:.3f} "
            f"(k={k}). A determined optimizer can pay the gate penalty "
            "in exchange for saturating every scorer.",
        )
    return Finding(
        "bounded cap sum",
        True,
        "pass",
        f"sum(caps)={cap_sum:.3f} within k*|gate_penalty|={bound:.3f}.",
    )


# ---------------------------------------------------------------------------
# Legacy rules
# ---------------------------------------------------------------------------


def _check_gate_typing(components: Sequence[RewardComponent]) -> Finding:
    """Prerequisite-sounding components should be gates, not scorers."""
    prereq_keywords = {
        "format",
        "safety",
        "valid",
        "constraint",
        "compliance",
    }
    scorers = [
        c for c in components if c.component_type is ComponentType.SCORER
    ]
    gates = [
        c for c in components if c.component_type is ComponentType.GATE
    ]
    mistyped = [
        c.name
        for c in scorers
        if any(k in c.name.lower() for k in prereq_keywords)
    ]

    if mistyped:
        return Finding(
            "gate typing",
            False,
            "fail",
            f"{mistyped} look like prerequisites but are typed as scorers. "
            "This creates priority inversion risk. Make them gates.",
        )
    if not gates:
        return Finding(
            "gate typing",
            False,
            "warn",
            "No gates defined. If any component is a prerequisite "
            "(format, safety), it should be a gate.",
        )
    return Finding(
        "gate typing",
        True,
        "pass",
        f"{len(gates)} gate(s), {len(scorers)} scorer(s).",
    )


def _check_variance(
    components: Sequence[RewardComponent],
    rng: np.random.RandomState,
    n: int,
) -> Finding:
    """High variance ratio between scorers means one will dominate."""
    scorers = [
        c for c in components if c.component_type is ComponentType.SCORER
    ]
    if len(scorers) < 2:
        return Finding(
            "variance balance", True, "pass", "Fewer than 2 scorers."
        )

    variances: dict[str, float] = {}
    for c in scorers:
        samples = [float(c.fn(float(rng.uniform(0, 1)))) for _ in range(n)]
        variances[c.name] = float(np.var(samples))

    hi = max(variances.values())
    positive_variances = [v for v in variances.values() if v > 1e-10]
    lo = min(positive_variances) if positive_variances else hi
    ratio = hi / lo if lo > 1e-10 else float("inf")
    worst = max(variances, key=lambda k: variances[k])

    if ratio > 10:
        return Finding(
            "variance balance",
            False,
            "fail",
            f"{ratio:.0f}x variance ratio. '{worst}' will dominate. "
            "Tighten its influence_cap or normalize.",
        )
    if ratio > 3:
        return Finding(
            "variance balance",
            False,
            "warn",
            f"{ratio:.1f}x variance ratio. '{worst}' may have outsized "
            "influence.",
        )
    return Finding(
        "variance balance", True, "pass", f"{ratio:.1f}x ratio (acceptable)."
    )


def _check_correlation(
    components: Sequence[RewardComponent],
    rng: np.random.RandomState,
    n: int,
) -> Finding:
    """Components declared independent should actually be uncorrelated."""
    pairs: list[tuple[RewardComponent, RewardComponent]] = []
    for c in components:
        for other_name in c.independent_of:
            other = next(
                (x for x in components if x.name == other_name), None
            )
            if other is not None:
                pairs.append((c, other))

    if not pairs:
        return Finding(
            "independence",
            True,
            "pass",
            "No independence constraints declared.",
        )

    problems: list[str] = []
    actions = rng.uniform(0, 1, size=n)
    for c1, c2 in pairs:
        s1 = [float(c1.fn(float(a))) for a in actions]
        s2 = [float(c2.fn(float(a))) for a in actions]
        r = float(np.corrcoef(s1, s2)[0, 1])
        if abs(r) > 0.5:
            problems.append(f"'{c1.name}' and '{c2.name}' (r={r:.2f})")

    if problems:
        return Finding(
            "independence",
            False,
            "fail",
            f"Correlated pairs: {', '.join(problems)}",
        )
    return Finding(
        "independence",
        True,
        "pass",
        f"{len(pairs)} pair(s) checked, all clean.",
    )


def _check_validators(components: Sequence[RewardComponent]) -> Finding:
    """Health validators present somewhere in the stack."""
    with_val = [c.name for c in components if c.validator is not None]
    if not with_val:
        return Finding(
            "health monitoring",
            False,
            "warn",
            "No validators set. Signal degradation will go undetected.",
        )
    return Finding(
        "health monitoring",
        True,
        "pass",
        f"Validators on: {with_val}.",
    )


def _check_bounds(
    components: Sequence[RewardComponent],
    rng: np.random.RandomState,
    n: int,
) -> Finding:
    """Raw weighted-sum range bounded enough to be stable."""
    comp = WeightedSumCompositor(components)
    rewards = [
        float(comp.compose(float(rng.uniform(0, 1)))["reward"])
        for _ in range(n)
    ]
    lo, hi = min(rewards), max(rewards)
    span = hi - lo

    if span > 10:
        return Finding(
            "reward range",
            False,
            "warn",
            f"Range [{lo:.1f}, {hi:.1f}] is wide. Large swings destabilize "
            "training.",
        )
    return Finding(
        "reward range",
        True,
        "pass",
        f"Range [{lo:.2f}, {hi:.2f}].",
    )


# ---------------------------------------------------------------------------
# Pretty-printing and pytest fixture
# ---------------------------------------------------------------------------


def format_report(report: dict[str, Any]) -> str:
    """Render an audit report as a plain-text block."""
    lines: list[str] = []
    lines.append("=" * 55)
    lines.append("COMPOSITION AUDIT")
    lines.append("=" * 55)
    lines.append(
        f"{report['n_components']} components "
        f"({report['n_gates']} gates, {report['n_scorers']} scorers)"
    )
    lines.append(f"ok={report['ok']}  seed={report['seed']}")
    lines.append("")
    for f in report["findings"]:
        tag = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}[f.severity]
        lines.append(f"  [{tag}] {f.check}")
        lines.append(f"         {f.detail}")
        lines.append("")
    if report["recommendations"]:
        lines.append("-" * 55)
        for r in report["recommendations"]:
            lines.append(f"  {r}")
    lines.append("=" * 55)
    return "\n".join(lines)


def print_report(report: dict[str, Any]) -> None:
    """Print an audit report to stdout (notebook helper, not for src use)."""
    logger.info(format_report(report))
    print(format_report(report))  # noqa: T201 - intentional in notebook helper


def assert_audit_passes(
    config: CompositionSpec | Sequence[RewardComponent],
    cap_sum_k: float = 0.5,
) -> None:
    """Pytest-friendly fixture that fails on any audit `fail` severity.

    Usage in tests or notebook smoke checks:

        from src.audit import assert_audit_passes
        assert_audit_passes(my_spec)

    Warnings are tolerated; fails raise `AssertionError`.
    """
    report = audit(config, cap_sum_k=cap_sum_k)
    fails = [f for f in report["findings"] if f.severity == "fail"]
    if fails:
        msg = "Audit failed:\n" + format_report(report)
        raise AssertionError(msg)


__all__ = [
    "Finding",
    "audit",
    "format_report",
    "print_report",
    "assert_audit_passes",
]
