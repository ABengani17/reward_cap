"""
Tests for composition, scenarios, and audit.
Run: pytest tests/ -v
"""

import pytest

from src.composition import (
    ComponentType,
    CompositionMonitor,
    GatedCompositor,
    RewardComponent,
    WeightedSumCompositor,
    run_bandit,
)
from src.scenarios import priority_inversion_broken, priority_inversion_fixed


class TestGatedCompositor:

    def test_failed_gate_returns_penalty(self):
        gate = RewardComponent(name="fmt", fn=lambda a: 0.0,
                               component_type=ComponentType.GATE, gate_threshold=0.5)
        scorer = RewardComponent(name="q", fn=lambda a: 1.0,
                                 component_type=ComponentType.SCORER)
        c = GatedCompositor([gate, scorer], gate_penalty=-1.0, monitor=False)
        assert c.compose(0.5)["reward"] == -1.0
        assert c.compose(0.5)["gates_pass"] is False

    def test_passed_gate_evaluates_scorers(self):
        gate = RewardComponent(name="fmt", fn=lambda a: 1.0,
                               component_type=ComponentType.GATE, gate_threshold=0.5)
        scorer = RewardComponent(name="q", fn=lambda a: 0.6,
                                 component_type=ComponentType.SCORER)
        c = GatedCompositor([gate, scorer], monitor=False)
        r = c.compose(0.5)
        assert r["gates_pass"] is True
        assert r["reward"] > 0

    def test_influence_cap_clips_contribution(self):
        big = RewardComponent(name="big", fn=lambda a: 10.0,
                              component_type=ComponentType.SCORER,
                              weight=1.0, influence_cap=1.0)
        c = GatedCompositor([big], monitor=False)
        assert c.compose(0.5)["reward"] <= 1.0

    def test_output_always_bounded(self):
        extreme = RewardComponent(name="x", fn=lambda a: 100.0,
                                  component_type=ComponentType.SCORER, weight=5.0)
        c = GatedCompositor([extreme], reward_bounds=(-1, 1), monitor=False)
        assert -1.0 <= c.compose(0.5)["reward"] <= 1.0

    def test_monotone(self):
        for v in [0.1, 0.3, 0.5, 0.7]:
            lo = GatedCompositor(
                [RewardComponent(name="r", fn=lambda a, v=v: v,
                                 component_type=ComponentType.SCORER)],
                monitor=False)
            hi = GatedCompositor(
                [RewardComponent(name="r", fn=lambda a, v=v+0.2: v,
                                 component_type=ComponentType.SCORER)],
                monitor=False)
            assert hi.compose(0.5)["reward"] >= lo.compose(0.5)["reward"]


class TestWeightedSum:

    def test_basic_sum(self):
        c1 = RewardComponent(name="a", fn=lambda a: 1.0,
                             component_type=ComponentType.SCORER, weight=2.0)
        c2 = RewardComponent(name="b", fn=lambda a: 0.5,
                             component_type=ComponentType.SCORER, weight=1.0)
        r = WeightedSumCompositor([c1, c2]).compose(0.5)
        assert abs(r["reward"] - 2.5) < 1e-6


class TestMonitor:

    def test_flags_dominance(self):
        comps = [
            RewardComponent(name="big", fn=lambda a: 5.0,
                            component_type=ComponentType.SCORER, influence_cap=1.5),
            RewardComponent(name="small", fn=lambda a: 0.1,
                            component_type=ComponentType.SCORER, influence_cap=1.5),
        ]
        m = CompositionMonitor(comps)
        v = m.record({"big": 5.0, "small": 0.1})
        assert any(x["type"] == "dominance" for x in v)

    def test_no_false_positive_when_balanced(self):
        comps = [
            RewardComponent(name="a", fn=lambda a: 0.5,
                            component_type=ComponentType.SCORER),
            RewardComponent(name="b", fn=lambda a: 0.5,
                            component_type=ComponentType.SCORER),
        ]
        m = CompositionMonitor(comps)
        assert len(m.record({"a": 0.5, "b": 0.5})) == 0


class TestPriorityInversion:

    def test_broken_converges_to_format_trap(self):
        comp = WeightedSumCompositor(priority_inversion_broken())
        h = run_bandit(comp, n_steps=2000, seed=42)
        assert h["policy_mu"][-1] < 0.5

    def test_fixed_reaches_correctness_peak(self):
        comp = GatedCompositor(priority_inversion_fixed(), monitor=False)
        h = run_bandit(comp, n_steps=2000, seed=42)
        assert h["policy_mu"][-1] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
