# rewardcap: A Constraint-Projection Layer for Monitorability-Preserving Reward Composition

**Aakarsh Karit.** Workshop draft. May 2026.

## Abstract

LLM post-training pipelines compose many heterogeneous reward signals — correctness checkers, LLM judges (sometimes with chain-of-thought access), safety classifiers, format validators — into a single scalar via ad-hoc weighted sums. When at least one scorer reads the model's chain of thought, gradient descent will discover and amplify any feature in the CoT that correlates with that scorer, producing obfuscated reward hacking (Baker et al. 2025). We propose `rewardcap`, a composition layer that treats reward aggregation as a constraint projection rather than a hyperparameter tuning problem — the reward-side analog of Dalal et al.'s (2018) safety layer for continuous-action RL. The layer is parameterized by a per-scorer cap `c_s` and, for CoT-reading scorers, a monitorability budget `δ` that bounds the per-sample policy-update component attributable to information appearing only in the CoT. We give a back-of-envelope KL bound that ties `c_s` to a total-drift budget, derive the soundness of the differential cap, and demonstrate on a 2D toy that the layer suppresses CoT drift without giving up task reward.

## 1. Introduction

The reward function in a modern RL post-training run is rarely a single object. Open-r1's GRPO recipe (Hugging Face, 2025) composes thirteen scorers: accuracy, format, tag count, reasoning steps, length, cosine-scaled length, repetition penalty, six code-execution variants. DeepSeek-R1 (DeepSeek-AI, 2025) composes a verifier with an LLM judge. Med-RLVR (Zhang et al., 2025) composes a format check with a correctness check. The composition function is invariably a weighted sum with hand-tuned coefficients.

Three documented failure modes recur. *Priority inversion*: a sharp-gradient prerequisite (format) traps the policy below the correctness peak; this is the Med-RLVR direct-answer-hacking dynamic. *Gradient dominance*: a high-variance LLM judge swamps the verifier and the policy converges to judge preferences; this is the DeepSeek-R1 length-gaming dynamic, and Wen et al. (2024) document the analog in RLHF. *CoT drift*: a CoT-reading scorer rewards spurious features that appear only in the model's chain of thought, producing the obfuscated reward hacking characterized by Baker et al. (2025) and motivating the monitorability discussion in Carroll et al. (2026) and Shlegeris (2026).

Each failure has been addressed pointwise: gating for priority inversion, normalization for gradient dominance, reduced CoT pressure for drift. We argue that all three are instances of the same problem: the composition function lacks structural constraints that make these failures impossible by construction. We propose treating the composition function as a *projection layer* — the reward-side analog of Dalal et al.'s (2018) safety layer — and we show that three structural primitives suffice to suppress the three failure modes on a controlled 2D toy.

## 2. Related work

Dalal et al. (2018) introduced a *safety layer* that projects each proposed action onto a linearized feasible set, eliminating constraint violations by construction rather than penalty. Our composition function is the reward-side analog: instead of projecting actions, we project the reward stack onto a composition that bounds each scorer's contribution.

Baker et al. (2025) trained reasoning models with explicit CoT optimization pressure and observed that obfuscated reward hacking emerges: the model continues to hack while emitting benign-looking CoT. They recommend not optimizing CoT directly. We take the next step and parameterize the CoT-conditioned reward signal by an explicit budget `δ`, recovering their recommendation in the limit `δ → 0`.

Carroll et al. (2026) report that a subset of OpenAI training runs accidentally graded CoT and find no clear evidence of monitorability degradation. Shlegeris (2026), reviewing that report, argues in footnote 5 that monitorability degradation should scale with the standard deviation of the CoT-grading effect on reward relative to total reward variance, estimating that ratio at roughly `10⁻⁴`. We give the formal derivation in Section 4.1.

Guan et al. (2025) propose intervention, process, and outcome-property archetypes for measuring monitorability. The monitorability proxy in our `qwen_gsm8k` scaffold falls in their outcome-property family.

Zhang et al. (2025), Wen et al. (2024), and the DeepSeek-R1 length-gaming reports provide the empirical record motivating Sections 3 and 5.

## 3. Method

A composition spec is a tuple `(G, S, c, m)`:

- `G` is a list of *gates* `g(x,a) ∈ {0, 1}`, predicates for prerequisite constraints.
- `S` is a list of *scorers* `s(x,a) ∈ ℝ`, real-valued quality signals.
- `c: S → ℝ₊` assigns a per-sample cap to each scorer.
- `m: S → Predicate` assigns optional health monitors.

The composed reward is

```
R(x,a) = [∀g ∈ G: g(x,a) = 1] · Σ_s min(c_s, s(x,a)).
```

The gate indicator short-circuits a sample that fails any prerequisite. Each scorer's contribution is hard-capped at `c_s` regardless of its raw value.

For a CoT-reading scorer `s_aware`, we require a companion CoT-blind scorer `s_blind` and substitute the **differential cap**

```
ŝ(x,a;δ) = min(c_s, s_blind(x,a) + clip(s_aware(x,a) - s_blind(x,a), -δ, δ)).
```

`δ` is the per-sample *monitorability budget*: the maximum magnitude the scorer is allowed to attribute to CoT-conditioned signal beyond what the CoT-blind scorer would assign. `δ = 0` collapses to the blind reward; `δ = ∞` recovers the unclipped aware reward.

We provide a `Gate`, `Scorer`, `DifferentialCap`, `Monitor`, and `CompositionSpec` dataclass API and a `Compositor` runtime, plus a static `audit` that enforces three rules: every CoT-reading scorer is wrapped in a `DifferentialCap` (R1); at least one gate is present (R2); the sum of caps is at most `k·|gate_penalty|` for `k < 1` (R3). The audit is exposed as a pytest fixture so configurations that regress fail CI.

## 4. Analysis

### 4.1 A per-scorer KL bound

Consider vanilla policy gradient with learning rate `η` on a parametric policy `π_θ`. Decompose `R = Σ_s R_s` where `R_s(x,a) = min(c_s, s(x,a))`. The contribution of scorer `s` to a single update is

```
Δθ_s = η · E_π[R_s · ∇ log π_θ].
```

For small updates the induced KL between pre- and post-update policies is well approximated by the Fisher quadratic form `KL ≈ (1/2) Δθ^⊤ F(θ) Δθ`. Substituting the gradient expression and using `‖R_s‖_∞ ≤ c_s`,

```
KL_s = O(η · c_s · Var(s))
```

per step, and over `N` steps,

```
KL_s^{(N)} = O(η · c_s · Var(s) · N).
```

This is the formal version of Shlegeris's footnote 5. To bound total policy drift attributable to the scorer stack by `ε`, set `c_s ≤ ε / (η · N · |S| · Var(s))`. High-variance scorers get smaller caps. Practitioners can read off `c_s` from quantities they already estimate.

### 4.2 Differential cap soundness

Suppose `s_blind` and `s_aware` are both unbiased estimators of the same downstream property `ρ`, i.e. `E[s_blind] = E[s_aware] = ρ`. Then replacing `s_aware` by `ŝ` shifts expected reward by at most `δ` per sample, while the component of the policy gradient attributable to information appearing *only* in the CoT — the residual `ξ = s_aware - s_blind` — is bounded by `δ · ‖∇ log π_θ‖` per sample, because the clipped integrand has magnitude at most `δ`. In particular, `δ = 0` produces the gradient of the blind reward exactly; `δ → ∞` recovers the unclipped aware gradient, and the monitorability tax of Baker et al. (2025) is paid in full.

The bound is loose: the Fisher approximation is second order, scorers are correlated in practice, off-policy corrections (PPO clipping, GRPO advantages) introduce additional terms, gate flips create bimodal reward distributions, and the unbiased-equivalence assumption between blind and aware scorers fails in real deployments where the blind scorer is a strict subset of the aware scorer's information. See [`docs/theory.md`](../docs/theory.md) §4 for the full caveat list.

## 5. Experiments

We instantiate a 2D canonical example. The action space is `[0, 1]²` with axes `action_quality` (what the user cares about) and `cot_content` (a feature visible only in the model's chain of thought). The ground-truth task reward and the CoT-blind scorer depend only on `action_quality` (peaked at 0.8). The CoT-aware scorer adds a sharp spurious bonus at `cot_content ≈ 0.75` whose gradient is steeper than the quality gradient near the policy's initial position. A 2D Gaussian policy is trained by REINFORCE for 800 steps. Initial mean is `(0.4, 0.3)`; the CoT ridge is at `0.75`; success means reaching `action_quality ≈ 0.8` without drifting up in `cot_content`.

We compare five composition schemes: `weighted_sum` (no constraints), `gated`, `capped`, `gated_capped`, and `gated_capped_differential`. Table 1 reports endpoints averaged across seed 0 (figures are byte-stable; full ablation across seeds appears in `results/ablations.csv`).

| Scheme | `action_quality` end | `cot_content` end | task reward | hacking rate |
| :-- | --: | --: | --: | --: |
| weighted_sum | 0.82 | 0.77 | 0.91 | 0.70 |
| gated | 0.82 | 0.77 | 0.91 | 0.70 |
| capped | 0.86 | 0.33 | 0.90 | 0.02 |
| gated_capped | 0.86 | 0.33 | 0.90 | 0.02 |
| gated_capped_differential | 0.89 | 0.29 | 0.88 | 0.00 |

The cap is the structural change that suppresses CoT drift; adding the differential cap on top brings the hacking rate (defined as the per-sample aware-blind gap exceeding 0.5) to zero and lets the policy reach the quality peak. Task reward drops slightly under tighter constraints, which is the expected monitorability tax — the practitioner's knob `δ` interpolates between full task reward and full monitorability.

The hero figure in the repository README (`results/figures/02_hero_seed0.png`) shows the policy trajectory under each scheme as a 2D path. Notebook 03 runs the same audit on a translation of the open-r1 GRPO reward stack (`accuracy_reward`, `format_reward`, `tag_count_reward`, `reasoning_steps_reward`, `len_reward`); the audit fails on R1 and R2 as written today.

A realistic-scale scaffold using TRL's `GRPOTrainer` with Qwen 2.5-1.5B on GSM8K is included at `experiments/qwen_gsm8k/`. It is not run in the artifact — the goal of the scaffold is to make the `--composition {weighted_sum, rewardcap}` switch a single-line change for any practitioner with a GPU.

## 6. Limitations

The 2D toy makes structural dynamics visible but does not capture off-policy corrections, KL regularization, or entropy bonuses present in real GRPO/PPO loops. The KL bound is informal and overstates how cleanly per-step KL accumulates. The differential cap requires a CoT-blind companion scorer for every CoT-reading scorer; for open-ended rewards (helpfulness, style) without natural blind companions, the formulation does not yet apply. The GRPO scaffold is unrun by us. Whether `δ > 0` outperforms `δ = 0` on task accuracy in practice is the empirical question the framework is designed to make answerable, not one we settle here.

## References

- Baker, B., Huizinga, J., et al. 2025. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. arXiv:2503.11926.
- Carroll, M., et al. 2026. Investigating the consequences of accidentally grading CoT during RL. OpenAI Alignment Blog, 2026-05-07. https://alignment.openai.com/accidental-cot-grading/
- Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., Tassa, Y. 2018. Safe Exploration in Continuous Action Spaces. arXiv:1801.08757.
- DeepSeek-AI. 2025. DeepSeek-R1. arXiv:2501.12948.
- Guan, M. Y., Wang, M., Carroll, M., et al. 2025. Monitoring Monitorability. arXiv:2512.18311.
- Hugging Face. 2025. open-r1: src/open_r1/rewards.py. https://github.com/huggingface/open-r1
- Shlegeris, B. 2026. A review of "Investigating the consequences of accidentally grading CoT during RL". Redwood Research, 2026-05-07. https://blog.redwoodresearch.org/p/openai-cot
- Wen, J., et al. 2024. Language Models Learn to Mislead Humans via RLHF. arXiv:2409.12822.
- Zhang, S., et al. 2025. Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning. arXiv:2502.19655.
