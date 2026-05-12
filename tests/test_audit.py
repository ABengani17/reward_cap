"""Audit-rule tests. One positive and one negative case per rule.

Run: pytest tests/test_audit.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audit import assert_audit_passes, audit, format_report
from src.composition import (
    ComponentType,
    CompositionSpec,
    DifferentialCap,
    Gate,
    RewardComponent,
    Scorer,
)


def _good_spec(cap: float = 0.4) -> CompositionSpec:
    return CompositionSpec(
        gates=[Gate(name="format", predicate=lambda a: 1.0, threshold=0.5)],
        scorers=[
            Scorer(name="correctness", fn=lambda a: 0.5, cap=cap),
            DifferentialCap(
                name="judge",
                fn_aware=lambda a: 0.5,
                fn_blind=lambda a: 0.5,
                cap=cap,
                delta=0.1,
            ),
        ],
        gate_penalty=-2.0,
    )


# ---------------------------------------------------------------------------
# R1: differential cap on CoT readers
# ---------------------------------------------------------------------------


class TestRule1DifferentialCap:
    def test_passes_when_cot_reader_wrapped(self) -> None:
        spec = _good_spec()
        report = audit(spec)
        r1 = next(
            f for f in report["findings"]
            if f.check == "differential cap on CoT readers"
        )
        assert r1.severity == "pass"

    def test_fails_when_cot_reader_unwrapped(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 1.0)],
            scorers=[
                Scorer(
                    name="judge",
                    fn=lambda a: 0.5,
                    cap=0.4,
                    reads_cot=True,
                ),
            ],
            gate_penalty=-2.0,
        )
        report = audit(spec)
        r1 = next(
            f for f in report["findings"]
            if f.check == "differential cap on CoT readers"
        )
        assert r1.severity == "fail"
        assert "judge" in r1.detail


# ---------------------------------------------------------------------------
# R2: at least one gate
# ---------------------------------------------------------------------------


class TestRule2GatePresent:
    def test_passes_with_gate(self) -> None:
        spec = _good_spec()
        report = audit(spec)
        r2 = next(
            f for f in report["findings"] if f.check == "gate present"
        )
        assert r2.severity == "pass"

    def test_fails_without_gate(self) -> None:
        spec = CompositionSpec(
            gates=[],
            scorers=[Scorer(name="s", fn=lambda a: 0.5, cap=0.4)],
            gate_penalty=-2.0,
        )
        report = audit(spec)
        r2 = next(
            f for f in report["findings"] if f.check == "gate present"
        )
        assert r2.severity == "fail"


# ---------------------------------------------------------------------------
# R3: bounded cap sum
# ---------------------------------------------------------------------------


class TestRule3CapSum:
    def test_passes_when_caps_below_bound(self) -> None:
        # cap_sum = 0.4 + 0.4 = 0.8 <= 0.5 * |-2.0| = 1.0
        spec = _good_spec(cap=0.4)
        report = audit(spec)
        r3 = next(
            f for f in report["findings"] if f.check == "bounded cap sum"
        )
        assert r3.severity == "pass"

    def test_fails_when_cap_sum_too_large(self) -> None:
        # cap_sum = 0.9 + 0.9 = 1.8 > 0.5 * |-2.0| = 1.0
        spec = _good_spec(cap=0.9)
        report = audit(spec)
        r3 = next(
            f for f in report["findings"] if f.check == "bounded cap sum"
        )
        assert r3.severity == "fail"

    def test_fails_on_infinite_cap(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 1.0)],
            scorers=[
                Scorer(name="s", fn=lambda a: 0.5, cap=float("inf")),
            ],
            gate_penalty=-2.0,
        )
        report = audit(spec)
        r3 = next(
            f for f in report["findings"] if f.check == "bounded cap sum"
        )
        assert r3.severity == "fail"


# ---------------------------------------------------------------------------
# Legacy rules still work
# ---------------------------------------------------------------------------


class TestLegacyRules:
    def test_gate_typing_flags_mistyped_format_scorer(self) -> None:
        comps = [
            RewardComponent(
                name="format",
                fn=lambda a: 1.0,
                component_type=ComponentType.SCORER,
            ),
            RewardComponent(
                name="quality",
                fn=lambda a: 0.5,
                component_type=ComponentType.SCORER,
            ),
        ]
        report = audit(comps)
        rule = next(
            f for f in report["findings"] if f.check == "gate typing"
        )
        assert rule.severity == "fail"

    def test_variance_balance_flags_extreme_ratio(self) -> None:
        # 'low' has variance ~1/12, 'high' has variance ~25.
        # Ratio ~300x — must trip the fail branch.
        rng_low = np.random.RandomState(0)
        rng_high = np.random.RandomState(1)
        comps = [
            RewardComponent(
                name="low",
                fn=lambda a, rng=rng_low: float(rng.uniform(0, 1)),
                component_type=ComponentType.SCORER,
            ),
            RewardComponent(
                name="high",
                fn=lambda a, rng=rng_high: float(rng.normal(0, 5.0)),
                component_type=ComponentType.SCORER,
            ),
        ]
        report = audit(comps, n_samples=400, seed=0)
        rule = next(
            f for f in report["findings"] if f.check == "variance balance"
        )
        assert rule.severity in ("warn", "fail")

    def test_independence_flags_correlated_pair(self) -> None:
        comps = [
            RewardComponent(
                name="a",
                fn=lambda x: float(x),
                component_type=ComponentType.SCORER,
                independent_of=["b"],
            ),
            RewardComponent(
                name="b",
                fn=lambda x: float(x),
                component_type=ComponentType.SCORER,
            ),
        ]
        report = audit(comps, n_samples=200, seed=0)
        rule = next(
            f for f in report["findings"] if f.check == "independence"
        )
        assert rule.severity == "fail"


# ---------------------------------------------------------------------------
# Pytest fixture: assert_audit_passes
# ---------------------------------------------------------------------------


class TestAssertFixture:
    def test_good_spec_passes(self) -> None:
        assert_audit_passes(_good_spec())

    def test_bad_spec_raises(self) -> None:
        bad = CompositionSpec(
            gates=[],
            scorers=[
                Scorer(
                    name="judge",
                    fn=lambda a: 0.5,
                    cap=float("inf"),
                    reads_cot=True,
                ),
            ],
            gate_penalty=-1.0,
        )
        with pytest.raises(AssertionError):
            assert_audit_passes(bad)


# ---------------------------------------------------------------------------
# Pretty-print smoke
# ---------------------------------------------------------------------------


def test_format_report_runs() -> None:
    report = audit(_good_spec())
    text = format_report(report)
    assert "COMPOSITION AUDIT" in text
    assert "ok=True" in text
