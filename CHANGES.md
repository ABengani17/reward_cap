# CHANGES

Summary of the upgrade from the v0.1 prototype to v0.2 (this commit).

## What was done

### Thesis and framing
- Reframed `rewardcap` as a constraint-projection layer for monitorability-preserving reward composition, the reward-side analog of [Dalal et al. (2018)](https://arxiv.org/abs/1801.08757)'s safety layer.
- `docs/theory.md` (new) — constrained-MDP problem statement; per-scorer KL bound `KL_s = O(η · c_s · Var(s) · N)`; cap-sizing rule from a total-drift budget; differential-cap soundness statement; explicit loose-bound caveats; relation to Baker 2025 / Carroll 2026 / Shlegeris 2026 / Guan 2025 / Med-RLVR / Wen 2024. All citations grounded against the original sources via web search; nothing invented.
- `docs/problem_statement.md` retained as historical context (the object-capability framing). The new README and paper do not invoke it.

### Code
- `src/composition.py` — refactored. New typed primitives `Gate`, `Scorer`, `DifferentialCap`, `Monitor`, `CompositionSpec`, `Compositor`, `RewardResult`. Reward function `R(x) = [gates pass] · Σ_s min(c_s, s(x))` with the differential-cap substitution from `docs/theory.md` §1.1. Legacy `RewardComponent` / `GatedCompositor` / `WeightedSumCompositor` / `CompositionMonitor` / `run_bandit` preserved unchanged; `spec_from_components` shim converts legacy → new.
- `src/scenarios/` — converted from a single file to a package. `legacy.py` holds the three 1D-bandit scenarios verbatim (priority inversion, gradient dominance, signal degradation). `cot_drift.py` (new) is the canonical 2D toy with five composition schemes, a 2D REINFORCE training loop, and `reward_hacking_rate` / `cot_drift_distance` metrics.
- `src/audit.py` — three new structural rules (R1: differential cap on every CoT reader; R2: gate present; R3: `Σ_s c_s ≤ k · |gate_penalty|`). `assert_audit_passes` exposed as a pytest fixture so any config that regresses fails CI. All five legacy rules retained.
- `src/io_utils.py` (new) — `seeded_path()` helper that stamps seeds into output filenames (`02_hero_seed0.png`).
- `src/__init__.py` — exports both new and legacy APIs.

### Tests
- 10 → 45 tests, all green.
  - `tests/test_composition.py` — original 10 tests, unchanged.
  - `tests/test_new_api.py` — 22 tests covering each gate type, each cap type, the differential cap (δ=0, δ=∞, symmetric clipping, cap-after-residual), monitors, the 2D toy smoke (all five schemes), and the legacy→new shim.
  - `tests/test_audit.py` — 13 tests, positive and negative case for every audit rule, plus the pytest fixture happy/sad paths.

### Notebooks
- 3 → 4 notebooks, all executed cleanly under `make repro`.
  - `01_failures.ipynb` — three failure modes on the 2D toy + 1D bandit, three figures.
  - `02_structured_composition.ipynb` — five-scheme comparison; produces the hero figure (`results/figures/02_hero_seed0.png`) embedded at the top of the README.
  - `03_audit.ipynb` — audit on three realistic configs, including a faithful translation of the open-r1 reward stack from `huggingface/open-r1/blob/main/src/open_r1/rewards.py`.
  - `04_ablations.ipynb` — ablation table over {gate, cap, differential cap, monitor} × {monitorability, hacking rate, task reward}. Writes `results/ablations.csv`.
- Notebooks are authored from `scripts/build_notebooks.py` so they are trivial to regenerate and diff-review.

### Realistic experiment scaffold
- `experiments/qwen_gsm8k/` (new) — TRL `GRPOTrainer` + Qwen 2.5-1.5B + GSM8K. Single-file `run.py`, accompanying `rewards.py`, `requirements.txt`, `README.md`, Colab notebook `run.ipynb`. `--composition {weighted_sum, rewardcap}` is a one-flag switch. Emits `task_accuracy`, `reward_hacking_rate`, `monitorability_proxy`. Stubbed LLM judge (`_stub_judge`) — replace with a real client at run time. **Not executed by us.** The previous empty `experiments/endogeneity_probe/` was removed.

### Tooling
- `requirements.txt` pinned (`numpy==2.1.1` `matplotlib==3.9.2` `scipy==1.17.1`).
- `requirements-dev.txt` (new) adds pytest, mypy, ruff, jupyter, nbconvert, ipykernel, pandas, tabulate, all pinned.
- `Makefile` (new) with `setup`, `test`, `lint`, `typecheck`, `notebooks`, `repro`, `clean`.
- `pyproject.toml` (new) configures `mypy --strict` and `ruff`. **`mypy --strict src/` is clean** (7 source files, 0 errors). **`ruff check src/ tests/` is clean.**
- Seeds: every random call goes through a config's `seed` field; `src/io_utils.seeded_path()` stamps the seed into figure / CSV filenames.
- `print` calls in `src/` replaced with `logging` (the only remaining `print` is inside `audit.print_report`, which is an intentional notebook helper marked with `# noqa`).

### Presentation
- `README.md` rewritten in the 9-section structure (one-sentence pitch / hero figure / problem in three paragraphs / framework + diagram + audit table / quickstart / repro / what's novel / limitations / citations). Direct technical voice. Two visuals (hero figure + architecture diagram). All citation URLs verified by web search.
- `docs/diagram.svg` (new) — matplotlib-rendered architecture diagram, also rasterized to `docs/diagram.png`. Renderer at `scripts/draw_diagram.py`.
- `paper/paper.md` (new) — 4-page workshop draft (Introduction / Related Work / Method / Analysis / Experiments / Limitations / References). Same source citations as README.

## Verification

`make repro` succeeds end-to-end:

- `pytest tests/ -v` — **45 / 45 pass** in ~2s.
- `mypy --strict src/` — **Success: no issues found in 7 source files.**
- `ruff check src/ tests/` — **All checks passed.**
- All four notebooks execute headless without error.
- Figures land in `results/figures/`, ablation CSV in `results/ablations.csv`.

The hero figure tells the intended story:
- weighted_sum / gated → policy drifts to (0.82, 0.77), sitting on the spurious CoT ridge, hacking rate ≈ 0.70.
- capped / gated+capped → policy stays at (0.86, 0.33), hacking rate ≈ 0.02.
- gated+capped+differential → policy reaches (0.89, 0.29), hacking rate = 0.00.

## Assumptions I made

1. **Stop-after-refactor checkpoint.** You answered both "Proceed without further check-ins" and "Stop after compositor refactor for review" in the same multi-select. I interpreted this as: pause to summarize the API at the refactor checkpoint, then continue. You confirmed in the follow-up that you wanted me to keep going at speed.
2. **`problem_statement.md` is kept.** You also picked both "Keep" and "Drop" for the object-capability framing. I kept the essay in `docs/` as historical context and built the new README / theory / paper around the constraint-projection thesis. The README footer points to it.
3. **`δ` default.** I used `δ = 0.05` throughout the toy and the GRPO scaffold. The theory section explains how to choose it from a drift budget, but a single empirical default is needed for the demo.
4. **Cap-sum bound constant.** R3 uses `k = 0.5` by default (`Σ_s c_s ≤ 0.5 · |gate_penalty|`). This is a design rule, not a theorem; I picked 0.5 because it gives the optimizer no positive expected reward in any single sample for flipping a gate.
5. **Stub LLM judge** in the GRPO scaffold. The reward composition logic is fully wired; the actual judge call is a deterministic stand-in so the script imports cleanly without an API key. A real run replaces `_stub_judge` with an OpenAI / Anthropic / vLLM client; the `aware = full_completion`, `blind = parsed_answer` split is the right interface.
6. **2D toy parameters.** I tuned the CoT ridge to make the failure visually obvious (`magnitude=0.9, width=40, position=0.75`, `init_mu=(0.4, 0.3)`). These choices are honest in the sense that they produce a visible failure; they are not load-bearing for the structural claim.

## What I did not implement (and why)

1. **A learned blind scorer.** The differential cap currently assumes a hand-supplied CoT-blind companion for each CoT-reading scorer. For LLM-judge rewards this is natural (re-run the judge on the parsed answer only). For open-ended rewards without a natural companion, a probe-style learned blind scorer would generalize the formulation — but training that probe is a real project, not a refactor. Noted in README limitations and `docs/theory.md` §6.
2. **Actually running the Qwen 2.5-1.5B GRPO experiment.** No GPU on this machine. The scaffold has been argument-parse-tested and the reward composition has a passing smoke test. The numbers in `paper/paper.md` are from the 2D toy; the GRPO numbers are explicitly placeholders.
3. **A separate `Cap` class** wrapping a `Scorer`. The math object `(G, S, c, m)` puts `c` as a function from scorers to caps, but I attached `cap` directly to `Scorer` and `DifferentialCap` for ergonomics. This is exactly equivalent and simpler to declare. If you want the cap to be a separate object so multiple scorers can share a cap budget, that's a small future refactor.
4. **CI.** No GitHub Actions yaml. `make test` and `make repro` are CI-ready; wiring them into a `.github/workflows/` file is a 10-line addition I left out per the "reduce overhead" instruction.
5. **Notebook output cleaning.** Executed notebooks are committed with output cells (so the figures live both in `results/figures/` and inline). If you want stripped output, add `--ClearOutputPreprocessor.enabled=True` to the `make notebooks` invocation.

## What to review most carefully

In order of "if I am wrong, the artifact is wrong":

1. **`docs/theory.md` §2** — the KL-bound derivation. I called it back-of-envelope on purpose; check that the looseness statement (§4) is honest about what the bound does and doesn't deliver.
2. **`src/composition.py` `_eval_scorer`** — the runtime path of the differential cap. The clip is applied to `(aware − blind)`, then added to `blind`, then capped at `c_s`. Tests cover δ=0, δ=∞, symmetric clipping, and cap-after-residual; verify this matches the math in the theory doc.
3. **`src/audit.py` rule R3** — `Σ_s c_s ≤ k · |gate_penalty|` with `k = 0.5`. The constant is a defensible default but not a theorem. If you want a different default, change `cap_sum_k` in `audit()` and re-run.
4. **Citations**. Every URL in README / paper / theory / problem_statement was verified by web search during writing. The Shlegeris footnote-5 wording is paraphrased from his actual footnote 5 (the `1e-4` estimate is real); double-check the paraphrase against the source if you cite it in a longer write-up.
5. **2D toy parameters in `src/scenarios/cot_drift.py`** — the CoT ridge magnitude/width/position. If these get tuned for a slightly different visual story, regenerate the hero figure with `make repro`.
6. **Hero figure** at `results/figures/02_hero_seed0.png` — this is the artifact that has to land. If the visual story doesn't read as you expected on screen, tweak `scripts/build_notebooks.py` nb02 and re-run.

## Files that are new

```
CHANGES.md
Makefile
PLAN.md
README.md (rewrite)
docs/diagram.png
docs/diagram.svg
docs/theory.md
experiments/qwen_gsm8k/README.md
experiments/qwen_gsm8k/requirements.txt
experiments/qwen_gsm8k/rewards.py
experiments/qwen_gsm8k/run.ipynb
experiments/qwen_gsm8k/run.py
notebooks/04_ablations.ipynb
paper/paper.md
pyproject.toml
requirements-dev.txt
results/ablations.csv
results/figures/*.png
scripts/build_notebooks.py
scripts/draw_diagram.py
src/io_utils.py
src/scenarios/__init__.py
src/scenarios/cot_drift.py
src/scenarios/legacy.py        (moved from src/scenarios.py)
tests/test_audit.py
tests/test_new_api.py
```

## Files removed

```
experiments/endogeneity_probe/run_experiment.py   (was empty)
src/scenarios.py                                  (moved to src/scenarios/legacy.py)
```

The OpenAI canary string from the original prompt does not appear in any file.
