# rewardcap

Structured reward composition for RL training pipelines.

## The problem

Every major RL post-training pipeline composes multiple reward signals into a single scalar. Correctness verifiers, LLM judges, safety classifiers, format validators. The standard approach is a weighted sum with hand-tuned coefficients. This works until the optimizer finds the seams between components.

The Med-RLVR "Direct Answer Hacker" (Zhang et al., 2025) found that satisfying format constraints while ignoring correctness produced higher combined reward than actually answering the question. DeepSeek-R1 learned that length correlates with judge scores. Wen et al. (2024) showed models learn to produce outputs beyond the detection boundary of RLHF evaluators over training. In each case, every individual component worked as intended. The failure was in the composition.

## What this repo does

`rewardcap` replaces ad-hoc weighted sums with a composition layer that has three structural properties.

**Gating.** Prerequisites (format compliance, safety) are separated from quality signals (correctness, style). Gates are evaluated first and contribute zero gradient once satisfied. This eliminates priority inversion.

**Contribution bounds.** Each scorer's influence on the aggregate reward is capped. High-variance components cannot dominate the gradient regardless of their output values. This prevents seesaw exploitation.

**Health monitoring.** Each component can have a validator that checks whether it is still providing meaningful signal. When a safety classifier degrades because the policy has drifted beyond its training distribution, the monitor flags it.

## Repository structure

```
src/
  composition.py    core compositor and training loop
  scenarios.py      three failure scenarios with broken/fixed variants
  audit.py          static analysis for composition configs
notebooks/
  01_failures.ipynb              documented failure modes reproduced
  02_structured_composition.ipynb  gated composition preventing each
  03_audit.ipynb                  the audit tool on realistic configs
tests/
  test_composition.py            10 tests
docs/
  spec.pdf                       technical specification
  problem_statement.md           1-2 page overview
```

## Setup

```
pip install -r requirements.txt
pytest tests/ -v
cd notebooks && jupyter notebook
```

## Status

Research prototype. The simulations use a 1D bandit to make reward landscapes visible. Composition dynamics do not depend on model complexity. Full RL training experiments with open-weight models (Qwen 2.5 family) are the next step.

## License

Apache 2.0
