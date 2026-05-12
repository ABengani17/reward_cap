# Theory

This document gives the formal version of the framing used in the README and
paper. The framing is: `rewardcap` is a **constraint-projection layer for
monitorability-preserving reward composition** in LLM post-training. It is
the reward-side analog of the safety layer in [Dalal et al.
(2018)](https://arxiv.org/abs/1801.08757), where instead of projecting
actions onto a feasible set, we project a stack of reward signals onto a
composition that bounds each signal's contribution to policy update.

## 1. Setup

We work with a constrained Markov decision process

$$
\mathcal{M} = (\mathcal{X}, \mathcal{A}, P, R, \gamma)
$$

where $R$ is not a single reward but a *stack* of reward signals
$\{s_1, \dots, s_K\}$, each $s_k : \mathcal{X} \times \mathcal{A} \to
\mathbb{R}$. Some of these signals (LLM judges, judges with CoT access,
process-supervision rewards) condition on the model's chain of thought; we
call these **CoT-reading scorers**. Others (string-matched correctness,
program-output validators) condition only on the final answer and are
**CoT-blind scorers**.

Standard pipelines compose the stack as a weighted sum,
$R(x,a) = \sum_k w_k\, s_k(x,a)$, hand-tune the $w_k$, and feed the result
to the optimizer. This produces the failure modes documented in [Med-RLVR
(Zhang et al., 2025)](https://arxiv.org/abs/2502.19655) and the length
gaming behavior in [DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948).

We replace the weighted sum with the following composition. A composition
spec is a 4-tuple $(G, S, c, m)$:

- $G = \{g_1, \dots, g_J\}$ — *gates*, predicates $g_j(x,a) \in \{0, 1\}$
  for prerequisite constraints (format compliance, safety filters).
- $S = \{s_1, \dots, s_K\}$ — *scorers*, real-valued quality signals.
- $c : S \to \mathbb{R}_{\geq 0}$ — a per-scorer *cap*, $c_s$.
- $m : S \to (\text{predicate})$ — optional *monitors* on each scorer
  flagging degradation or distribution shift.

The composed reward is

$$
R(x,a) \;=\; \mathbb{1}\!\left[\bigwedge_{g \in G} g(x,a) = 1\right] \;\cdot\;
        \sum_{s \in S} \min\!\bigl(c_s,\; s(x,a)\bigr).
$$

The gate indicator short-circuits the whole reward — a sample that fails any
gate contributes zero gradient to scorer optimization. Each scorer's
contribution is capped at $c_s$ regardless of its raw value.

### 1.1 Differential cap

For a CoT-reading scorer $s_{\text{aware}}$ we additionally require a
companion CoT-blind scorer $s_{\text{blind}}$ measuring the same downstream
property. The differential cap is

$$
\widetilde{s}(x,a;\delta) \;=\; \min\!\Bigl(c_s,\;\;
   s_{\text{blind}}(x,a) + \operatorname{clip}\bigl(
     s_{\text{aware}}(x,a) - s_{\text{blind}}(x,a),\,
     -\delta,\, \delta\bigr)\Bigr).
$$

The parameter $\delta$ is a **monitorability budget**: it is the maximum
per-sample reward magnitude the scorer is allowed to attribute to
CoT-conditioned signal *beyond* what the CoT-blind scorer would have
assigned. Setting $\delta = 0$ collapses to the CoT-blind reward; setting
$\delta = \infty$ recovers an uncapped CoT-aware reward.

## 2. A back-of-envelope policy-shift bound

We sketch why capping at $c_s$ bounds the policy KL contribution attributable
to scorer $s$. The argument is informal — see Section 4 for where it is
loose.

Consider vanilla policy gradient with learning rate $\eta$ on a parametric
policy $\pi_\theta$:

$$
\theta \leftarrow \theta + \eta\, \mathbb{E}_{a \sim \pi_\theta}
   \!\bigl[R(x,a)\,\nabla_\theta \log \pi_\theta(a|x)\bigr].
$$

Decompose $R$ by scorer: $R = \sum_s R_s$ where $R_s(x,a) =
\min(c_s, s(x,a))$ (ignoring gates for the moment). The contribution of
scorer $s$ to a single update is

$$
\Delta \theta_s \;=\; \eta\, \mathbb{E}\!\bigl[R_s\, \nabla \log \pi_\theta\bigr].
$$

For small updates, the induced KL between the pre- and post-update policies
is well approximated by the Fisher quadratic form,

$$
\mathrm{KL}\bigl(\pi_\theta \,\|\, \pi_{\theta + \Delta\theta_s}\bigr)
   \;\approx\; \tfrac{1}{2}\, \Delta\theta_s^\top F(\theta)\, \Delta\theta_s,
$$

with $F$ the Fisher information matrix. Substituting the gradient expression
and using $\|R_s\|_\infty \le c_s$,

$$
\mathrm{KL}_s \;\le\; \tfrac{1}{2}\,\eta^2\, c_s^2 \cdot \mathbb{E}\!\bigl[
   \|\nabla \log \pi_\theta\|_{F(\theta)^{-1}}^2 \bigr]
   \cdot \operatorname{Var}_\pi(R_s)/c_s^2.
$$

To first order in the cap and in the per-sample variance of the (capped)
score, the per-step KL contribution scales as

$$
\mathrm{KL}_s \;=\; \mathcal{O}\bigl(\eta\, c_s \cdot \operatorname{Var}(s)\bigr),
$$

and across $N$ training steps, accumulating linearly under the small-step
approximation,

$$
\boxed{\;\mathrm{KL}_s^{(N)} \;=\; \mathcal{O}\!\bigl(\eta\, c_s\,
   \operatorname{Var}(s)\, N\bigr).\;}
$$

This is the formal version of the heuristic in [Shlegeris's review of the
OpenAI accidental-CoT-grading post (Redwood Research, 2026), footnote
5](https://blog.redwoodresearch.org/p/openai-cot), which estimates that if
the standard deviation of the CoT-grading effect on reward is $\sim 10^{-4}$
of the total reward standard deviation, then training a model on $N$
trajectories produces a CoT change scaling with that ratio.

### 2.1 Setting caps to bound total drift

Suppose we have a total policy-drift budget $\varepsilon$ attributable to
the entire scorer stack — that is, we want

$$
\sum_{s \in S} \mathrm{KL}_s^{(N)} \;\le\; \varepsilon.
$$

Plugging in the bound and solving for $c_s$ gives

$$
c_s \;\le\; \frac{\varepsilon}{\eta\, N\, |S|\, \operatorname{Var}(s)},
$$

i.e. each scorer's cap is inversely proportional to (a) the optimizer's
aggressiveness $\eta N$, (b) the number of scorers, and (c) the scorer's
empirical variance. High-variance scorers get smaller caps. This is
deployable in practice: estimate $\operatorname{Var}(s)$ on a held-out
prompt set, pick $\varepsilon$ from a policy drift budget (e.g. a KL bound
already used elsewhere in the training stack), and read off $c_s$.

### 2.2 Cap sum bounded by gate magnitude

A second design rule that falls out of the bound. The gate indicator
contributes a step of size $|\text{gate\_penalty}|$ to the reward whenever a
gate flips. If $\sum_s c_s$ is allowed to exceed the gate penalty, a
sufficiently determined optimizer can pay the gate penalty in exchange for
the upside of saturating every scorer. The rule

$$
\sum_{s \in S} c_s \;\le\; k\, \cdot\, |\text{gate\_penalty}|
$$

for some $k < 1$ enforces that gates dominate; we use $k = 0.5$ as a default
in the audit. This is a design rule rather than a theorem, but the failure
mode it prevents is concrete: see notebook 03 for an open-r1-style
configuration that violates it.

## 3. Differential cap soundness

**Claim.** Suppose $s_{\text{blind}}$ and $s_{\text{aware}}$ are both
unbiased estimators of the same downstream property $\rho(x,a)$, i.e.
$\mathbb{E}[s_{\text{blind}}] = \mathbb{E}[s_{\text{aware}}] = \rho$.
Replacing $s_{\text{aware}}$ by the differential cap $\widetilde s$ with
budget $\delta$ preserves the expected reward up to $\delta$ per sample
while bounding the directional component of the policy gradient that is
attributable to information appearing *only* in the CoT to be at most
$\delta\, \|\nabla \log \pi_\theta\|$ per sample.

**Argument.** Write $s_{\text{aware}} = s_{\text{blind}} + \xi$ where
$\xi = s_{\text{aware}} - s_{\text{blind}}$ is the CoT-conditioned residual.
By construction $\mathbb{E}[\xi] = 0$. After clipping, $|\widetilde s -
s_{\text{blind}}| \le \delta$, so

$$
|\mathbb{E}[\widetilde s] - \mathbb{E}[s_{\text{blind}}]| \;\le\; \delta,
$$

i.e. expected reward shifts by at most $\delta$. For the gradient,

$$
\nabla_\theta \mathbb{E}_\pi[\widetilde s] \;=\;
\nabla_\theta \mathbb{E}_\pi[s_{\text{blind}}] \;+\;
\nabla_\theta \mathbb{E}_\pi[\operatorname{clip}(\xi, -\delta, \delta)].
$$

The second term has per-sample magnitude bounded by $\delta\, \|\nabla \log
\pi_\theta\|$ exactly because the integrand is bounded by $\delta$. So the
component of the policy update that is "due to" the CoT residual — the
component a CoT obfuscation attack can exploit, per [Baker et al.
(2025)](https://arxiv.org/abs/2503.11926) — has its scale controlled by the
single hyperparameter $\delta$.

In particular, with $\delta = 0$ the policy gradient is identical to what
the CoT-blind scorer alone would produce; with $\delta \to \infty$ it
recovers an unclipped CoT-aware reward, and the "monitorability tax" Baker
et al. discuss is paid in full.

## 4. Where the bound is loose

The KL bound in Section 2 is informal and overstates how cleanly per-step KL
accumulates. The main looseness:

1. **Linearization of the policy.** The Fisher quadratic form approximates
   KL only to second order. For large $\eta$ or large $c_s$, the policy can
   make discrete jumps that the bound does not see.
2. **Independence between scorers.** We summed per-scorer KL contributions
   as if they were independent. In practice scorers are correlated (a
   correctness signal and a length signal both move with the same
   completion), and the joint update can exceed or undershoot the sum.
3. **Off-policy corrections.** PPO-style importance-sampled updates and
   trust-region clipping introduce additional terms not captured here. The
   bound is meaningful as a sizing heuristic, not a guarantee.
4. **Gate flips.** When a gate predicate changes its value across training,
   the realized reward distribution is bimodal and the variance estimate
   $\operatorname{Var}(s)$ used to set $c_s$ becomes a lower bound on the
   true variance.
5. **The differential-cap claim assumes unbiased equivalence between
   $s_{\text{blind}}$ and $s_{\text{aware}}$.** In real deployments the
   blind scorer is usually a weaker proxy (e.g. an exact-match correctness
   checker vs. an LLM judge); the bias term shows up as a constant offset
   in expected reward and a residual gradient direction the differential
   cap does not control. The right way to read the claim is: the
   differential cap bounds the *CoT-conditioned* gradient direction, not
   every direction in which the two scorers disagree.

These are the standard caveats for any first-order analysis of policy
gradient. The point of the bound is not to be tight; the point is to give
practitioners a defensible procedure for setting $c_s$ and $\delta$ from
quantities they already estimate.

## 5. Relation to prior work

[Dalal et al. (2018)](https://arxiv.org/abs/1801.08757) introduced a
*safety layer* for continuous-action RL that projects each proposed action
onto a feasible set defined by linearized constraint models. The
constraint-projection idea is the same; the locus is different. Their layer
operates on actions, ours operates on the reward signal stack. Both are
non-Lagrangian: rather than adding a penalty term and hoping the optimizer
respects the constraint in expectation, both projection layers make
violations structurally impossible.

[Baker et al. (2025)](https://arxiv.org/abs/2503.11926) showed that direct
optimization pressure on chain-of-thought tokens produces *obfuscated*
reward hacking — models continue to hack but emit benign-looking CoT
traces. They recommend not optimizing CoT directly; we take the next step
and give a procedure that bounds the CoT-conditioned reward signal by a
quantifiable budget rather than requiring it to be zero.

[Carroll et al. (2026)](https://alignment.openai.com/accidental-cot-grading/)
reports that a subset of OpenAI training runs *did* accidentally grade CoT,
and that monitorability did not detectably degrade in those runs.
[Shlegeris's review (Redwood, 2026)](https://blog.redwoodresearch.org/p/openai-cot)
argues that the absence of detected degradation is consistent with the
inadvertent CoT-grading signal being a tiny fraction of total reward
variance, and quantifies the regime (his footnote 5). Our bound in
Section 2 is the explicit derivation of that footnote.

[Guan et al. (2025)](https://arxiv.org/abs/2512.18311) — "Monitoring
Monitorability" — proposes evaluation archetypes (intervention, process,
outcome-property) for measuring CoT monitorability. The monitorability
proxy in our `experiments/qwen_gsm8k/` scaffold is in their
outcome-property family.

[Wen et al. (2024)](https://arxiv.org/abs/2409.12822) showed that RLHF
models learn to convince humans they are right even when they are wrong
(U-Sophistry). The gradient-dominance scenario in
`src/scenarios/gradient_dominance.py` is a simplified version of this
dynamic in a setting where the failure can be directly observed.

## 6. Open questions

- The KL bound is tight up to constants but the constants matter for
  picking caps in practice. An empirical fit of the constants on a real
  GRPO run would convert this from a sizing heuristic into a calibrated
  procedure.
- The differential cap requires a CoT-blind companion scorer for every
  CoT-reading scorer. For LLM-judge rewards, the natural companion is an
  exact-match or programmatic check on the final answer; for rewards
  without such a companion (e.g. open-ended helpfulness) the formulation
  does not yet apply. A natural extension is a *learned* blind scorer
  trained to predict the aware scorer's score from the answer alone.
- Whether $\delta = 0$ (CoT signal entirely ignored) underperforms
  $\delta > 0$ on task accuracy in practice is the question
  `experiments/qwen_gsm8k/` is designed to answer. The point of the
  framework is to make $\delta$ a knob practitioners can sweep, not to
  prescribe its value.
