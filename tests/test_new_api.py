"""Tests for the new typed composition API (Gate, Scorer, DifferentialCap,
Compositor, CompositionSpec) and the 2D cot_drift scenario.

Run: pytest tests/test_new_api.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.composition import (
    CompositionSpec,
    Compositor,
    DifferentialCap,
    Gate,
    Monitor,
    Scorer,
)
from src.scenarios import cot_drift

# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------


class TestGate:
    def test_passing_gate_lets_reward_through(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 1.0, threshold=0.5)],
            scorers=[Scorer(name="s", fn=lambda a: 0.4, cap=1.0)],
            reward_bounds=(-2.0, 2.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.gates_pass is True
        assert r.reward == pytest.approx(0.4)

    def test_failing_gate_short_circuits_to_penalty(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 0.0, threshold=0.5)],
            scorers=[Scorer(name="s", fn=lambda a: 0.4, cap=1.0)],
            gate_penalty=-1.0,
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.gates_pass is False
        assert r.reward == pytest.approx(-1.0)

    def test_threshold_is_inclusive(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 0.5, threshold=0.5)],
            scorers=[Scorer(name="s", fn=lambda a: 0.4, cap=1.0)],
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.gates_pass is True

    def test_multiple_gates_conjunction(self) -> None:
        spec = CompositionSpec(
            gates=[
                Gate(name="g1", predicate=lambda a: 1.0, threshold=0.5),
                Gate(name="g2", predicate=lambda a: 0.0, threshold=0.5),
            ],
            scorers=[Scorer(name="s", fn=lambda a: 0.4, cap=1.0)],
            gate_penalty=-1.0,
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.gates_pass is False
        assert r.reward == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Scorer / cap behavior
# ---------------------------------------------------------------------------


class TestScorerCap:
    def test_cap_clips_high_scores(self) -> None:
        spec = CompositionSpec(
            scorers=[Scorer(name="big", fn=lambda a: 10.0, cap=1.5)],
            reward_bounds=(-5.0, 5.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.reward == pytest.approx(1.5)
        assert r.capped["big"] == pytest.approx(1.5)
        assert r.scores["big"] == pytest.approx(10.0)

    def test_uncapped_scorer_passes_through(self) -> None:
        spec = CompositionSpec(
            scorers=[Scorer(name="s", fn=lambda a: 0.3, cap=10.0)],
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.reward == pytest.approx(0.3)

    def test_reward_bounds_clip_total(self) -> None:
        spec = CompositionSpec(
            scorers=[
                Scorer(name="a", fn=lambda a: 1.0, cap=1.0),
                Scorer(name="b", fn=lambda a: 1.0, cap=1.0),
            ],
            reward_bounds=(-1.0, 1.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        # Without bounds, 1.0 + 1.0 = 2.0. Bounded to 1.0.
        assert r.reward == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DifferentialCap behavior
# ---------------------------------------------------------------------------


class TestDifferentialCap:
    def test_delta_zero_recovers_blind(self) -> None:
        spec = CompositionSpec(
            scorers=[
                DifferentialCap(
                    name="d",
                    fn_aware=lambda a: 0.9,
                    fn_blind=lambda a: 0.3,
                    cap=2.0,
                    delta=0.0,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.reward == pytest.approx(0.3)

    def test_delta_large_recovers_aware(self) -> None:
        spec = CompositionSpec(
            scorers=[
                DifferentialCap(
                    name="d",
                    fn_aware=lambda a: 0.9,
                    fn_blind=lambda a: 0.3,
                    cap=2.0,
                    delta=10.0,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.reward == pytest.approx(0.9)

    def test_delta_clips_residual_symmetrically(self) -> None:
        # aware - blind = +0.6 should clip to +0.1 (delta=0.1)
        spec_pos = CompositionSpec(
            scorers=[
                DifferentialCap(
                    name="d",
                    fn_aware=lambda a: 0.9,
                    fn_blind=lambda a: 0.3,
                    cap=2.0,
                    delta=0.1,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
        )
        r_pos = Compositor(spec_pos).compose(np.array([0.5, 0.5]))
        assert r_pos.reward == pytest.approx(0.4)

        # aware - blind = -0.6 should clip to -0.1
        spec_neg = CompositionSpec(
            scorers=[
                DifferentialCap(
                    name="d",
                    fn_aware=lambda a: 0.3,
                    fn_blind=lambda a: 0.9,
                    cap=2.0,
                    delta=0.1,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
        )
        r_neg = Compositor(spec_neg).compose(np.array([0.5, 0.5]))
        assert r_neg.reward == pytest.approx(0.8)

    def test_cap_still_applies(self) -> None:
        # blind=0.5, aware=1.0, delta=10 lets full residual through -> 1.0
        # then cap at 0.6 clips final to 0.6
        spec = CompositionSpec(
            scorers=[
                DifferentialCap(
                    name="d",
                    fn_aware=lambda a: 1.0,
                    fn_blind=lambda a: 0.5,
                    cap=0.6,
                    delta=10.0,
                ),
            ],
            reward_bounds=(-2.0, 2.0),
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert r.reward == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Monitor behavior
# ---------------------------------------------------------------------------


class TestMonitor:
    def test_monitor_violation_surfaced(self) -> None:
        # Predicate returns False after step 0 -> violation each step.
        spec = CompositionSpec(
            scorers=[Scorer(name="s", fn=lambda a: 0.5, cap=1.0)],
            monitors=[
                Monitor(
                    name="m",
                    target="s",
                    predicate=lambda buf, step: step < 1,
                ),
            ],
        )
        c = Compositor(spec)
        r0 = c.compose(np.array([0.5, 0.5]), step=0)
        r1 = c.compose(np.array([0.5, 0.5]), step=2)
        assert r0.violations == []
        assert any(v["type"] == "monitor" for v in r1.violations)

    def test_scorer_validator_surfaced(self) -> None:
        spec = CompositionSpec(
            scorers=[
                Scorer(
                    name="s",
                    fn=lambda a: 0.5,
                    cap=1.0,
                    validator=lambda buf, step: False,
                ),
            ],
        )
        r = Compositor(spec).compose(np.array([0.5, 0.5]))
        assert any(v["type"] == "degraded" for v in r.violations)


# ---------------------------------------------------------------------------
# Spec convenience
# ---------------------------------------------------------------------------


class TestCompositionSpec:
    def test_cap_sum_and_gate_magnitude(self) -> None:
        spec = CompositionSpec(
            gates=[Gate(name="g", predicate=lambda a: 1.0)],
            scorers=[
                Scorer(name="a", fn=lambda a: 0.1, cap=0.4),
                Scorer(name="b", fn=lambda a: 0.1, cap=0.6),
            ],
            gate_penalty=-2.0,
        )
        assert spec.cap_sum == pytest.approx(1.0)
        assert spec.gate_magnitude == pytest.approx(2.0)
        assert spec.scorer_names == ["a", "b"]
        assert spec.gate_names == ["g"]


# ---------------------------------------------------------------------------
# 2D toy smoke
# ---------------------------------------------------------------------------


class TestCotDriftSmoke:
    @pytest.mark.parametrize("scheme", list(cot_drift.ALL_SCHEMES))
    def test_each_scheme_runs(self, scheme: str) -> None:
        cfg = cot_drift.TrainConfig(n_steps=30, n_samples=8, seed=0)
        h = cot_drift.run_cot_drift(scheme, cfg)  # type: ignore[arg-type]
        assert len(h.step) == 30
        assert h.policy_mu[-1].shape == (2,)
        assert np.all(h.policy_mu[-1] >= 0.0)
        assert np.all(h.policy_mu[-1] <= 1.0)
        assert np.isfinite(h.mean_reward[-1])

    def test_weighted_sum_drifts_toward_cot_ridge(self) -> None:
        # The whole point of the toy: unconstrained reward pulls the
        # policy off the initial cot_content toward the spurious CoT
        # ridge at 0.75; the differential cap suppresses that drift.
        cfg = cot_drift.TrainConfig(n_steps=600, n_samples=64, seed=0)
        h_ws = cot_drift.run_cot_drift("weighted_sum", cfg, cap=0.5)
        h_dc = cot_drift.run_cot_drift(
            "gated_capped_differential", cfg, cap=0.5, delta=0.05
        )
        init_cot = cfg.init_mu[1]
        drift_ws = abs(h_ws.policy_mu[-1][1] - init_cot)
        drift_dc = abs(h_dc.policy_mu[-1][1] - init_cot)
        # Weighted sum should drift several times more than differential.
        assert drift_ws > 3 * drift_dc

    def test_reward_hacking_rate_in_unit_interval(self) -> None:
        cfg = cot_drift.TrainConfig(n_steps=50, n_samples=16, seed=0)
        h = cot_drift.run_cot_drift("weighted_sum", cfg)
        rate = cot_drift.reward_hacking_rate(h, hack_threshold=0.3)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Legacy compat: spec_from_components
# ---------------------------------------------------------------------------


class TestLegacyConversion:
    def test_spec_from_components_round_trip(self) -> None:
        from src.composition import (
            ComponentType,
            RewardComponent,
            spec_from_components,
        )

        comps = [
            RewardComponent(
                name="g",
                fn=lambda a: 1.0,
                component_type=ComponentType.GATE,
                gate_threshold=0.5,
            ),
            RewardComponent(
                name="s",
                fn=lambda a: 0.4,
                component_type=ComponentType.SCORER,
                weight=1.0,
                influence_cap=1.5,
            ),
        ]
        spec = spec_from_components(comps)
        assert spec.gate_names == ["g"]
        assert spec.scorer_names == ["s"]
        assert spec.cap_sum == pytest.approx(1.5)
