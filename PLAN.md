<!--
PLAN.md — upgrade of rewardcap_v2 into a research-quality artifact.

Repo path: /Users/aaki17/Desktop/cais/ai safety/foresight/rewardcap_v2

STARTING STATE (read before drafting this plan):

  README.md
    Frames rewardcap as a generic "structured reward composition" layer
    with three properties: gating, contribution bounds, health monitoring.
    Cites Med-RLVR, DeepSeek-R1, Wen et al. 2024. No mention of CoT
    monitorability, Dalal et al. 2018, or any constraint-projection framing.

  src/
    __init__.py     version stub, no exports.
    composition.py  defines RewardComponent dataclass (single class for
                    gates and scorers, discriminated by ComponentType enum),
                    CompositionMonitor (dominance / correlation / validator
                    checks on a rolling window), GatedCompositor (gates
                    short-circuit to gate_penalty, scorers aggregated as a
                    bounded weighted mean with per-component influence_cap),
                    WeightedSumCompositor (baseline, no constraints), and
                    run_bandit (REINFORCE on a 1D Gaussian policy in [0,1]).
                    Pure numpy. No logging, no type-strictness, scalar
                    actions only. No DifferentialCap. ~290 lines.
    scenarios.py    Three scenario builders, each returning a broken/fixed
                    pair of RewardComponent lists for: priority inversion
                    (format vs correctness), gradient dominance (style judge
                    swamps via variance), signal degradation (drifting safety
                    classifier). 1D bandit only. ~205 lines.
    audit.py        Five checks: gate typing, variance balance, declared
                    independence, validator presence, raw weighted-sum
                    range. Findings dataclass with severity in
                    {pass, warn, fail}. Print helper. ~175 lines. Imports
                    from src.composition, used in notebook 03 but not
                    wired into pytest.

  notebooks/
    01_failures.ipynb              Three failure modes on the 1D bandit
                                   with reproducibility across 30 seeds.
    02_structured_composition.ipynb  Gated/capped variants fix each.
    03_audit.ipynb                 Audit tool demo on plausible configs.
                                   No notebook 04 today.

  tests/
    __init__.py        Empty.
    test_composition.py  10 tests covering GatedCompositor, WeightedSum,
                         Monitor, and priority-inversion convergence.
                         No tests for audit, scenarios 2/3, no 2D toy.

  docs/
    problem_statement.md  1-page essay framing reward composition as a
                          *security* problem, citing Mark Miller's object-
                          capability work. This is the current pitch.
    spec.tex, spec.pdf    Older spec (not read in detail; will retain).

  experiments/endogeneity_probe/run_experiment.py  Empty file.

  requirements.txt   numpy / matplotlib / scipy / pytest, unpinned.
  No Makefile, no requirements-dev.txt, no mypy config, no logging,
  no seeded output filenames, no figures directory.

  No paper/, no docs/theory.md, no docs/diagram.svg, no CHANGES.md.
  CLAUDE.md is also absent.
-->

# PLAN

This plan upgrades the existing prototype into a research-quality artifact
organized around a sharper thesis: `rewardcap` is a **constraint-projection
layer for monitorability-preserving reward composition in LLM post-training**,
the reward-side analog of Dalal et al. (2018)'s safety layer.

The old framing (object-capability security) stays in
`docs/problem_statement.md` as background, but the README, paper, and
theory doc are rebuilt around the constraint-projection / CoT-monitorability
thesis.

The 1D bandit prototype is replaced as the canonical example by a 2D
`(action_quality, cot_content)` toy. The 1D bandit and the three existing
scenarios are retained (they motivate the basic gating and capping rules
that the new framework subsumes), but they become a subset of the new
2D failure suite rather than the headline.

## Sequencing

Per user instruction, work in this order, running `pytest tests/ -v`
between major steps:

1. `docs/theory.md` — sharpen the thesis on paper first so the code matches
   the math, not the other way around.
2. `src/scenarios/cot_drift.py` — the 2D toy that everything downstream
   uses as the canonical example.
3. `src/composition.py` refactor — formal `(G, S, c, m)` API,
   `DifferentialCap`, declarative dataclass-based composition spec, logging,
   strict type annotations.
4. `src/audit.py` expansion — three new rules wired as a pytest fixture.
5. `tests/` expansion — at minimum: each gate type, each cap type,
   `DifferentialCap`, every audit rule, 2D toy smoke test.
6. Notebooks 01–04 — rebuild on the 2D toy, produce hero figure and
   ablation CSV.
7. `experiments/qwen_gsm8k/` — TRL+GRPO scaffold + Colab notebook.
   Not executed locally.
8. README rewrite, `docs/diagram.svg`.
9. `paper/paper.md` workshop draft.
10. `make repro` end-to-end, then `CHANGES.md`.

## Deliverable checklist

### Thesis and framing
- [ ] `docs/theory.md` with:
  - [ ] Constrained-MDP problem statement, multiple reward signals.
  - [ ] Back-of-envelope per-scorer KL bound:
        `KL_s ≤ O(η · c_s · Var(s) · N)`, with the derivation steps shown.
  - [ ] Inverse map: choose `c_s` to bound `KL_s ≤ ε / |S|`. Formal
        version of Shlegeris's footnote 5 (cite `[TODO: verify citation]`
        if I cannot pull a primary source for the exact footnote text).
  - [ ] `DifferentialCap` soundness statement: if `s_blind` is a valid
        reward signal in expectation, capping `δ` preserves expected reward
        up to `δ` per sample while bounding policy shift in CoT-conditioned
        directions.
  - [ ] Loose-bound caveats: non-linear policies, large η, off-policy
        corrections, etc.
- [ ] Citations: Dalal 2018, Baker 2025, Carroll 2026 (OpenAI Alignment
      Blog 2026-05-07), Shlegeris fn5, Guan 2025, Med-RLVR (Zhang 2025),
      DeepSeek-R1, Wen 2024. Any I cannot pin down to a real reference
      via web search get a `[TODO: verify citation]` marker. **The
      OpenAI canary string is excluded from every file.**

### Code: composition layer
- [ ] `src/composition.py` refactored as the formal tuple
      `(G, S, c, m)` with composition
      `R(x) = [∀g ∈ G: g(x)=1] · Σ_s min(c_s, s(x))`.
- [ ] `DifferentialCap(s_aware, s_blind, δ)` returning
      `min(c_s, s_blind(x) + clip(s_aware(x) - s_blind(x), -δ, δ))`.
- [ ] Declarative dataclass spec (`CompositionSpec` or similar) so users
      can write `CompositionSpec(gates=[...], scorers=[...])` and get
      a typed object the auditor and compositor both consume.
- [ ] Distinct dataclasses for `Gate`, `Scorer`, `Cap`,
      `DifferentialCap`, `Monitor` rather than a single `RewardComponent`
      with an enum discriminator. The current `RewardComponent` is kept
      as a thin compatibility shim **only** if removing it would break a
      test that's still meaningful; otherwise removed cleanly.
- [ ] All public functions type-annotated. `mypy --strict` passes.
- [ ] `logging` replaces any `print` in `src/`.

### Code: 2D scenario
- [ ] `src/scenarios/cot_drift.py` exposing a 2D action space
      `(action_quality, cot_content) ∈ [0,1]²`, an `s_aware` that depends
      on both, an `s_blind` that depends only on `action_quality`, and a
      ground-truth task reward.
- [ ] A 2D policy-gradient training loop (REINFORCE on an
      independent-Gaussian policy, or a small MLP — bandit is fine,
      depth doesn't matter for these dynamics).
- [ ] Refactor of `src/scenarios.py` → `src/scenarios/` package with
      `priority_inversion.py`, `gradient_dominance.py`,
      `signal_degradation.py`, `cot_drift.py`, all importing from the
      refactored composition module.

### Code: audit
- [ ] `src/audit.py` adds:
  - [ ] Rule: every CoT-reading scorer (any `s_aware`) must be wrapped
        in a `DifferentialCap`.
  - [ ] Rule: at least one gate is present.
  - [ ] Rule: `Σ_s c_s ≤ k · gate_magnitude` for some `k`. The bound
        constant is justified in `docs/theory.md`.
- [ ] Old rules (gate typing, variance, independence, validators, range)
      preserved.
- [ ] Pytest fixture `assert_audit_passes(config)` so any notebook or
      experiment config that violates the rules fails CI.

### Tests
- [ ] Tests for each gate type (threshold, soft, conjunction if added).
- [ ] Tests for each cap type (hard `Cap`, `DifferentialCap`).
- [ ] Tests for every audit rule (positive and negative case each).
- [ ] Smoke test for the 2D toy (training runs, finishes, produces a
      finite policy mean).
- [ ] Existing 10 tests preserved and passing.
- [ ] Target: ≥ 25 tests, all green under `pytest -v`.

### Notebooks (executed locally)
- [ ] `notebooks/01_failures.ipynb` — three failure modes on the 2D toy,
      including CoT drift as the new one. One paragraph per failure.
- [ ] `notebooks/02_structured_composition.ipynb` — five-scheme comparison
      (vanilla weighted sum / gated only / capped only / gated+capped /
      gated+capped+differential cap). Hero figure suitable for the README.
- [ ] `notebooks/03_audit.ipynb` — three realistic configs, one cribbed
      from the open-r1 reward stack. Show which rules each violates.
- [ ] `notebooks/04_ablations.ipynb` — ablation table over
      {gate, cap, differential cap, monitor} × {monitorability,
      hacking rate, task reward}. Output as a Markdown table and
      `results/ablations.csv`.

### Realistic experiment scaffold (not executed)
- [ ] `experiments/qwen_gsm8k/` directory.
- [ ] `run.py` using TRL `GRPOTrainer` with Qwen 2.5-1.5B on GSM8K.
- [ ] Reward mix: correctness, length, LLM-as-judge (with optional CoT
      access flag).
- [ ] `--composition {weighted_sum, rewardcap}`.
- [ ] Metrics: task accuracy, reward-hacking rate
      (`judge_high ∧ correctness_wrong`), monitorability proxy
      (held-out classifier accuracy on CoT-only inputs).
- [ ] `run.ipynb` Colab-compatible.
- [ ] `README.md` inside the experiment directory describing how to
      launch on a GPU (the actual training is the user's job).

### Tooling and hygiene
- [ ] Pinned `requirements.txt`.
- [ ] `requirements-dev.txt` with pytest, mypy, ruff, jupyter,
      nbconvert.
- [ ] `Makefile` targets: `setup`, `test`, `lint`, `repro`, `notebooks`.
      `make repro` executes notebooks via
      `jupyter nbconvert --to notebook --execute` and dumps figures to
      `results/figures/`.
- [ ] Seed plumbing: every random call uses a seed from a config
      object; seed is stamped into output filenames
      (`figures/02_cot_drift_seed42.png` and so on).
- [ ] `mypy --strict` clean for `src/`.

### Presentation
- [ ] README rewritten in the 9-section structure (one-sentence pitch /
      hero figure / problem in three paragraphs / framework / quickstart /
      reproducing / what's novel / limitations / citations).
- [ ] `docs/diagram.svg` — architecture figure produced via matplotlib
      (or graphviz if installed). One diagram showing raw scorer stack
      passing through the rewardcap layer before reaching the optimizer.
- [ ] `paper/paper.md` — 4-page workshop write-up:
      Introduction / Related Work / Method / Experiments / Limitations
      / References.

### Final
- [ ] `make repro` runs end-to-end with no notebook errors.
- [ ] `CHANGES.md` summarizing what was done, assumptions made, what
      was *not* implemented (and why), what to review most carefully.

## Risks and assumptions to flag now

1. **Citations.** I cannot independently verify several of the listed
   references (Carroll 2026, Shlegeris's footnote 5, Guan 2025 exact
   title, Med-RLVR exact author list, Wen 2024 venue). I will run web
   searches and use the result if found. Anything I cannot verify gets
   `[TODO: verify citation]` — I will not invent metadata.
2. **`mypy --strict`.** This is genuinely strict (no implicit `Any`,
   no untyped decorators, etc.). The existing `numpy` usage and the
   `Callable` signatures will need narrowing (`Callable[[float], float]`
   etc.). I will add stubs only if necessary.
3. **TRL/GRPO/Qwen scaffold.** I can write a runnable script but I
   cannot test it without a GPU. I'll lint it and dry-run argument
   parsing; the user runs the actual training.
4. **Open-r1 reward stack** for notebook 03. I will fetch the public
   reward-function configuration from the open-r1 repo via web search;
   if the exact file is unavailable I will use a plausible reconstruction
   and note it as such.
5. **Hero figure** in the README is referenced from `results/figures/`,
   so it only appears after `make repro` runs. I'll commit a placeholder
   path; first `make repro` populates the actual PNG.
6. **Canary handling.** The OpenAI canary string from the prompt is not
   written to any file in this repo. Citations to Carroll 2026 reference
   the blog post by title and date only.

Ready for review. Once you confirm, I'll start at deliverable 1
(`docs/theory.md`) and run tests between each major step.
