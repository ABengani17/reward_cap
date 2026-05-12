# rewardcap

When we combine multiple reward signals in LLM RL post-training using a weighted sum, whichever scorer has the steepest gradient ends up steering the policy, regardless of weight. Format compliance crowds out correctness ([Med-RLVR](https://arxiv.org/abs/2502.19655)). An LLM judge that can read the chain of thought ends up rewarding whatever CoT pattern correlates with high scores, even when that pattern has nothing to do with the task ([Baker et al. 2025](https://arxiv.org/abs/2503.11926)).

`rewardcap` is a small Python library that replaces the weighted sum with a gated sum of capped scorers, plus a separate per-sample budget for any scorer that reads the chain of thought.

## Install

```bash
pip install -r requirements-dev.txt
```

## Quickstart

```python
from src.composition import (
    CompositionSpec, Gate, Scorer, DifferentialCap, Compositor,
)
from src.audit import assert_audit_passes

spec = CompositionSpec(
    gates=[Gate("format", predicate=is_well_formed, threshold=0.5)],
    scorers=[
        Scorer("correctness", fn=exact_match, cap=0.5),
        DifferentialCap(
            name="judge",
            fn_aware=judge_with_cot,
            fn_blind=judge_answer_only,
            cap=0.3,
            delta=0.05,
        ),
    ],
    gate_penalty=-2.0,
)
assert_audit_passes(spec)

reward = Compositor(spec).compose(model_output).reward
```

The `format` gate short-circuits the reward to `-2.0` when the model output is malformed, so format compliance can't be traded against correctness. The `correctness` scorer is hard-clipped at 0.5 per sample. The `judge` is a CoT-reading LLM judge, so it gets a `DifferentialCap` with a CoT-blind companion (the same judge run on just the parsed answer). The wrapper computes

```
s̃(x) = s_blind(x) + clip(s_aware(x) - s_blind(x), -δ, δ)
```

clips that at `c_s`, and so the CoT-only component of the reward can move the policy by at most `δ` per sample. `δ = 0` falls back to the blind reward, `δ → ∞` recovers the unclipped aware reward, and the interesting region is small positive values.

## Does it work

The toy in `src/scenarios/cot_drift.py` is two dimensional. Actions live in `[0,1]²` with axes `action_quality` and `cot_content`. Task reward peaks at `action_quality = 0.8` and is flat in the CoT axis. A CoT-aware judge likes high quality but also assigns a sharp `+0.9` bonus near `cot_content = 0.75`, a feature that has nothing to do with task quality but is visible in the chain of thought. The bonus gradient is steeper than the quality gradient near the policy's initial position.

![Policy trajectories under the two composition schemes](results/figures/hero.png)

Under a plain weighted sum (left), the policy walks straight up the spurious CoT ridge and finishes at `(0.82, 0.77)`. With gates, caps, and the differential wrapper (right), it ignores the ridge and reaches `(0.89, 0.29)`.

![CoT drift over training and final hacking rate](results/figures/drift.png)

The hacking rate is the fraction of final samples where the CoT-aware judge scores high but the CoT-blind checker disagrees by more than `0.5`.

| scheme | quality end | CoT end | hacking rate | task reward |
| ---- | ---- | ---- | ---- | ---- |
| weighted sum | 0.82 | 0.77 | 70% | 0.91 |
| cap | 0.86 | 0.33 | 2% | 0.90 |
| cap + differential (δ=0.05) | 0.89 | 0.29 | 0% | 0.88 |

Numbers are deterministic at `SEED=0`. The full five-way ablation, including gate-only and gate+cap (both of which collapse to the rows above in this toy because the gate is satisfied everywhere by construction), is in `results/ablations.csv`.

## Audit

`audit(spec)` runs three structural checks before training. Every CoT-reading scorer must be wrapped in a `DifferentialCap`. At least one gate must be present. The sum of caps must be at most half the magnitude of the gate penalty, otherwise a determined optimizer can pay the gate penalty in exchange for saturating every scorer. The third rule is a design heuristic, and the KL argument in `docs/theory.md` is informal (second-order Fisher approximation, independent per-scorer KL, no off-policy corrections). Read it as a sizing rule for `c_s`, not a guarantee.

## Reproduce

```bash
make repro
```

Runs the 45 tests, regenerates the figures, and executes the four notebooks. The full five-scheme grid and the open-r1-style audit example live in `notebooks/02_structured_composition.ipynb` and `notebooks/03_audit.ipynb`. A GRPO scaffold for Qwen 2.5-1.5B + GSM8K is at `experiments/qwen_gsm8k/` and is unrun here.

## References

- Baker et al., 2025. [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926).
- Zhang et al., 2025. [Med-RLVR](https://arxiv.org/abs/2502.19655).
- Dalal et al., 2018. [Safe Exploration in Continuous Action Spaces](https://arxiv.org/abs/1801.08757). The safety-layer paper this borrows the projection framing from.
- Carroll et al., 2026. [Investigating the consequences of accidentally grading CoT during RL](https://alignment.openai.com/accidental-cot-grading/).

Apache 2.0.
