"""Generate notebooks/*.ipynb programmatically.

Notebooks are checked into the repo as JSON so that diff-friendly review
works, but they are easier to author from Python. Each `make repro`
re-executes them and persists outputs; if you change a notebook, update
the cell list in this script and re-run `python scripts/build_notebooks.py`,
then `make notebooks` to re-execute.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"


def md(*lines: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }


def code(*lines: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(lines),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Notebook 01: failure modes
# ---------------------------------------------------------------------------


nb01 = notebook([
    md(
        "# Three Failure Modes of Reward Composition",
        "",
        "We reproduce three documented failure modes on the canonical 2D toy",
        "from `src.scenarios.cot_drift`. The action space is `[0,1]^2` with",
        "axes `action_quality` and `cot_content`. A ground-truth task reward",
        "depends only on `action_quality`; a CoT-reading scorer assigns a",
        "spurious bonus to a specific `cot_content` pattern.",
        "",
        "Each failure mode is one paragraph and one figure.",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))",
        "",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "from src.scenarios import cot_drift",
        "from src.io_utils import seeded_path",
        "",
        "plt.rcParams.update({",
        "    'figure.dpi': 130, 'font.size': 10,",
        "    'axes.spines.top': False, 'axes.spines.right': False,",
        "})",
        "SEED = 0",
    ),
    md(
        "## 1. CoT drift",
        "",
        "Under a vanilla weighted sum the policy is pulled off the",
        "`action_quality` axis toward the CoT-content ridge (`cot_content`",
        "near 0.6). The ground-truth task reward stays flat in the CoT axis,",
        "but the policy distribution drifts anyway, because that direction",
        "is where the LLM-judge gradient is steepest. This is the failure",
        "mode the differential cap is designed to suppress.",
    ),
    code(
        "cfg = cot_drift.TrainConfig(n_steps=600, n_samples=64, seed=SEED)",
        "h_ws = cot_drift.run_cot_drift('weighted_sum', cfg)",
        "h_dc = cot_drift.run_cot_drift('gated_capped_differential', cfg, delta=0.05)",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))",
        "for ax, h, title in [(axes[0], h_ws, 'weighted sum'),",
        "                     (axes[1], h_dc, 'gated + capped + differential')]:",
        "    mu = np.array(h.policy_mu)",
        "    ax.plot(mu[:, 0], mu[:, 1], lw=1.4, color='#d32f2f')",
        "    ax.scatter([mu[0, 0]], [mu[0, 1]], color='black', s=30, zorder=5, label='start')",
        "    ax.scatter([mu[-1, 0]], [mu[-1, 1]], color='#1976d2', s=30, zorder=5, label='end')",
        "    ax.axhline(0.5, color='gray', ls=':', alpha=0.4)",
        "    ax.axhline(0.75, color='#ef6c00', ls=':', alpha=0.6, label='spurious CoT ridge')",
        "    ax.set_xlim(0, 1); ax.set_ylim(0, 1)",
        "    ax.set_xlabel('action_quality'); ax.set_ylabel('cot_content')",
        "    ax.set_title(title)",
        "    ax.legend(fontsize=8, loc='lower right')",
        "",
        "fig.suptitle('Failure mode 1: CoT drift', y=1.02)",
        "fig.tight_layout()",
        "fig.savefig(seeded_path('01_cot_drift', SEED), bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "## 2. Priority inversion (legacy 1D bandit)",
        "",
        "The Med-RLVR direct-answer-hacking finding (Zhang et al., 2025).",
        "Format compliance is treated as an additive scorer and creates a",
        "local maximum that traps the policy. The fix is to make it a gate.",
    ),
    code(
        "from src.composition import GatedCompositor, WeightedSumCompositor, run_bandit",
        "from src.scenarios import priority_inversion_broken, priority_inversion_fixed",
        "",
        "h_broken = run_bandit(WeightedSumCompositor(priority_inversion_broken()),",
        "                      n_steps=2000, seed=SEED)",
        "h_fixed  = run_bandit(GatedCompositor(priority_inversion_fixed(), monitor=False),",
        "                      n_steps=2000, seed=SEED)",
        "",
        "fig, ax = plt.subplots(figsize=(9, 3.5))",
        "ax.plot(h_broken['policy_mu'], color='#d32f2f', lw=1.6, label='weighted sum')",
        "ax.plot(h_fixed['policy_mu'],  color='#388e3c', lw=1.6, label='gated')",
        "ax.axhline(0.3, color='#ef6c00', ls=':', alpha=0.5, label='format trap')",
        "ax.axhline(0.8, color='#1976d2', ls=':', alpha=0.5, label='correctness peak')",
        "ax.set_xlabel('step'); ax.set_ylabel('policy mean')",
        "ax.set_title('Failure mode 2: priority inversion')",
        "ax.legend(fontsize=9, loc='center right')",
        "fig.tight_layout()",
        "fig.savefig(seeded_path('01_priority_inversion', SEED), bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "## 3. Seesaw exploitation (gradient dominance)",
        "",
        "A high-variance LLM judge swamps the gradient even at low nominal",
        "weight. Bounding each scorer's contribution restores balance.",
    ),
    code(
        "from src.scenarios import gradient_dominance_broken, gradient_dominance_fixed",
        "rng = np.random.RandomState(SEED)",
        "h_broken = run_bandit(WeightedSumCompositor(gradient_dominance_broken(rng)),",
        "                      n_steps=2000, seed=SEED)",
        "rng = np.random.RandomState(SEED)",
        "h_fixed  = run_bandit(GatedCompositor(gradient_dominance_fixed(rng), monitor=False),",
        "                      n_steps=2000, seed=SEED)",
        "",
        "fig, ax = plt.subplots(figsize=(9, 3.5))",
        "ax.plot(h_broken['policy_mu'], color='#d32f2f', lw=1.6, label='weighted sum')",
        "ax.plot(h_fixed['policy_mu'],  color='#388e3c', lw=1.6, label='gated + capped')",
        "ax.axhline(0.5, color='#9c27b0', ls=':', alpha=0.5, label='style attractor')",
        "ax.axhline(0.7, color='#1976d2', ls=':', alpha=0.5, label='correctness peak')",
        "ax.set_xlabel('step'); ax.set_ylabel('policy mean')",
        "ax.set_title('Failure mode 3: seesaw exploitation')",
        "ax.legend(fontsize=9, loc='center right')",
        "fig.tight_layout()",
        "fig.savefig(seeded_path('01_seesaw', SEED), bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "Three failure modes, three concrete fixes. The next notebook puts",
        "them side by side as a single comparison.",
    ),
])


# ---------------------------------------------------------------------------
# Notebook 02: five-scheme comparison + hero figure
# ---------------------------------------------------------------------------


nb02 = notebook([
    md(
        "# Five Composition Schemes Compared",
        "",
        "We run the 2D `cot_drift` toy under five composition schemes",
        "layered on top of one another so each plot isolates the effect of",
        "one structural property: a gate, a cap, both, and finally the",
        "differential cap that bounds the CoT-conditioned reward signal.",
        "",
        "This is the hero figure for the README. It is the visualisation",
        "the framework is judged against.",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))",
        "",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "from src.scenarios import cot_drift",
        "from src.io_utils import seeded_path",
        "",
        "plt.rcParams.update({",
        "    'figure.dpi': 130, 'font.size': 10,",
        "    'axes.spines.top': False, 'axes.spines.right': False,",
        "})",
        "SEED = 0",
        "CFG = cot_drift.TrainConfig(n_steps=800, n_samples=64, seed=SEED)",
    ),
    code(
        "histories = {",
        "    s: cot_drift.run_cot_drift(s, CFG, cap=0.5, delta=0.05)",
        "    for s in cot_drift.ALL_SCHEMES",
        "}",
        "for name, h in histories.items():",
        "    mu = h.policy_mu[-1]",
        "    print(f'  {name:<32}  policy_mu_end=({mu[0]:.3f}, {mu[1]:.3f})'",
        "          f'  hack_rate={cot_drift.reward_hacking_rate(h):.2f}'",
        "          f'  cot_drift={cot_drift.cot_drift_distance(h):.3f}')",
    ),
    md(
        "## The hero figure",
        "",
        "Each panel shows the trajectory of the policy mean across training",
        "in the 2D space. Red is the weighted-sum baseline; layered",
        "structural properties move the endpoint closer to the",
        "task-reward peak (action_quality=0.8) without drifting along the",
        "spurious cot_content axis.",
    ),
    code(
        "labels = {",
        "    'weighted_sum':                 'a. weighted sum (baseline)',",
        "    'gated':                        'b. gated',",
        "    'capped':                       'c. capped',",
        "    'gated_capped':                 'd. gated + capped',",
        "    'gated_capped_differential':    'e. gated + capped + differential',",
        "}",
        "colors = {",
        "    'weighted_sum': '#d32f2f',",
        "    'gated': '#f57c00',",
        "    'capped': '#fbc02d',",
        "    'gated_capped': '#7cb342',",
        "    'gated_capped_differential': '#1976d2',",
        "}",
        "",
        "fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.6), sharex=True, sharey=True)",
        "for ax, scheme in zip(axes, cot_drift.ALL_SCHEMES):",
        "    h = histories[scheme]",
        "    mu = np.array(h.policy_mu)",
        "    ax.plot(mu[:, 0], mu[:, 1], lw=1.4, color=colors[scheme])",
        "    ax.scatter([mu[0, 0]], [mu[0, 1]], color='black', s=22, zorder=5)",
        "    ax.scatter([mu[-1, 0]], [mu[-1, 1]], color=colors[scheme], s=42, zorder=6,",
        "               edgecolor='white', linewidth=0.8)",
        "    ax.axhline(0.75, color='#ef6c00', ls=':', alpha=0.45, label='CoT ridge')",
        "    ax.axvline(0.8, color='#1976d2', ls=':', alpha=0.45)",
        "    ax.set_xlim(0, 1); ax.set_ylim(0, 1)",
        "    ax.set_title(labels[scheme], fontsize=9.5)",
        "    ax.set_xlabel('action_quality')",
        "axes[0].set_ylabel('cot_content')",
        "fig.suptitle('Policy trajectory in 2D under each composition scheme',",
        "             y=1.04, fontsize=12)",
        "fig.tight_layout()",
        "fig.savefig(seeded_path('02_hero', SEED), bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "## Summary metrics",
        "",
        "We track three things per scheme: how close the policy ends to the",
        "task-reward peak, how much it drifted along the CoT axis, and the",
        "fraction of final samples where the aware/blind scorer gap exceeds",
        "a threshold (the toy analog of the LLM-judge-vs-correctness-check",
        "reward-hacking rate).",
    ),
    code(
        "import pandas as pd",
        "",
        "rows = []",
        "for scheme, h in histories.items():",
        "    mu = h.policy_mu[-1]",
        "    rows.append({",
        "        'scheme': labels[scheme],",
        "        'task_reward_proxy': round(float(h.mean_task_reward[-1]), 3),",
        "        'distance_to_quality_peak': round(abs(float(mu[0]) - 0.8), 3),",
        "        'cot_drift_distance': round(cot_drift.cot_drift_distance(h), 3),",
        "        'reward_hacking_rate': round(cot_drift.reward_hacking_rate(h), 3),",
        "    })",
        "df = pd.DataFrame(rows).set_index('scheme')",
        "df",
    ),
    md(
        "The endpoint distance to the quality peak drops monotonically as",
        "structural properties are layered on. The CoT-drift distance is",
        "highest for the weighted-sum and capped-only schemes and lowest",
        "for the differential-cap scheme. The differential cap is doing",
        "what it is supposed to do: bounding policy update in the",
        "CoT-conditioned direction without giving up on task reward.",
    ),
])


# ---------------------------------------------------------------------------
# Notebook 03: audit on three realistic configs
# ---------------------------------------------------------------------------


nb03 = notebook([
    md(
        "# Auditing Realistic Composition Configs",
        "",
        "We run `src.audit.audit` on three plausible-looking composition",
        "configs and read off which structural rules each violates:",
        "",
        "1. A naive open-r1-style stack with no gating and no differential",
        "   cap on the LLM judge.",
        "2. A gated but unbounded stack that lets the CoT-aware judge",
        "   contribute arbitrarily.",
        "3. A fully constrained stack that should pass.",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))",
        "",
        "from src.audit import audit, format_report",
        "from src.composition import (",
        "    CompositionSpec, DifferentialCap, Gate, Scorer,",
        ")",
    ),
    md(
        "## Config 1: open-r1-style naive stack",
        "",
        "Inspired by the open-r1 `rewards.py` reward functions documented",
        "at `huggingface/open-r1/blob/main/src/open_r1/rewards.py`:",
        "`accuracy_reward`, `format_reward`, `tag_count_reward`,",
        "`reasoning_steps_reward`, `len_reward`. We translate the stack",
        "into the new API as faithfully as is possible without running",
        "actual completions; in particular we treat `format_reward` and",
        "`tag_count_reward` as additive scorers (which is how they are",
        "wired in the GRPO config), and treat `reasoning_steps_reward` as a",
        "CoT-reading scorer (it inspects intermediate reasoning text).",
        "",
        "The audit will fail on R1 (CoT reader unwrapped), R2 (no gates),",
        "and possibly R3 (cap sum vs gate penalty).",
    ),
    code(
        "# Stand-in scorer functions; the actual open-r1 implementations",
        "# operate on completion strings. The dynamics we audit are",
        "# structural, so the toy stand-ins suffice for rule checking.",
        "def accuracy(a): return 0.5",
        "def fmt(a): return 0.4",
        "def tag_count(a): return 0.3",
        "def reasoning_steps(a): return 0.6  # this one reads CoT",
        "def length_pen(a): return -0.1",
        "",
        "naive_stack = CompositionSpec(",
        "    gates=[],",
        "    scorers=[",
        "        Scorer(name='accuracy', fn=accuracy, cap=1.0),",
        "        Scorer(name='format', fn=fmt, cap=1.0),",
        "        Scorer(name='tag_count', fn=tag_count, cap=1.0),",
        "        Scorer(name='reasoning_steps', fn=reasoning_steps,",
        "               cap=1.0, reads_cot=True),",
        "        Scorer(name='length_pen', fn=length_pen, cap=1.0),",
        "    ],",
        "    gate_penalty=-1.0,",
        ")",
        "report = audit(naive_stack)",
        "print(format_report(report))",
    ),
    md(
        "All three new structural rules fail: no `DifferentialCap` on the",
        "CoT-reading `reasoning_steps`, no gates, and the cap sum exceeds",
        "the gate magnitude bound. This is exactly the configuration that",
        "the open-r1 community has reported produces format-reward collapse",
        "and reward-plateau pathologies (see open-r1 issues #363 and #256).",
    ),
    md(
        "## Config 2: gated but unbounded",
        "",
        "A common partial fix: promote `format` to a gate. Helps with",
        "priority inversion. Does not help with CoT drift if the LLM judge",
        "is still unbounded.",
    ),
    code(
        "gated_unbounded = CompositionSpec(",
        "    gates=[Gate(name='format_gate', predicate=fmt, threshold=0.5)],",
        "    scorers=[",
        "        Scorer(name='accuracy', fn=accuracy, cap=1.0),",
        "        Scorer(name='reasoning_steps', fn=reasoning_steps,",
        "               cap=1.0, reads_cot=True),",
        "        Scorer(name='length_pen', fn=length_pen, cap=1.0),",
        "    ],",
        "    gate_penalty=-1.0,",
        ")",
        "report = audit(gated_unbounded)",
        "print(format_report(report))",
    ),
    md(
        "R2 passes; R1 still fails because the CoT reader is not wrapped;",
        "R3 may or may not pass depending on the cap budget.",
    ),
    md(
        "## Config 3: fully constrained",
        "",
        "Each CoT reader is wrapped in a `DifferentialCap` with a small",
        "`delta`. The cap sum sits well below the gate magnitude.",
    ),
    code(
        "constrained = CompositionSpec(",
        "    gates=[Gate(name='format_gate', predicate=fmt, threshold=0.5)],",
        "    scorers=[",
        "        Scorer(name='accuracy', fn=accuracy, cap=0.5),",
        "        DifferentialCap(",
        "            name='reasoning_steps',",
        "            fn_aware=reasoning_steps,",
        "            fn_blind=accuracy,",
        "            cap=0.3,",
        "            delta=0.05,",
        "        ),",
        "        Scorer(name='length_pen', fn=length_pen, cap=0.2),",
        "    ],",
        "    gate_penalty=-2.0,",
        ")",
        "report = audit(constrained)",
        "print(format_report(report))",
        "assert report['ok']",
    ),
    md(
        "All three new rules pass. The audit is wired as a pytest fixture",
        "(`src.audit.assert_audit_passes`), so any notebook or experiment",
        "config that regresses will fail CI.",
    ),
])


# ---------------------------------------------------------------------------
# Notebook 04: ablations
# ---------------------------------------------------------------------------


nb04 = notebook([
    md(
        "# Ablation Table",
        "",
        "We isolate the contribution of each structural property (gate,",
        "cap, differential cap, monitor) to three metrics on the 2D toy:",
        "monitorability preservation (low CoT-drift distance),",
        "reward-hacking rate (low aware/blind gap), and task reward (high",
        "ground-truth task score). Output is dumped to",
        "`results/ablations.csv`.",
    ),
    code(
        "import sys, pathlib",
        "sys.path.insert(0, str(pathlib.Path.cwd().parent))",
        "",
        "import numpy as np",
        "import pandas as pd",
        "",
        "from src.scenarios import cot_drift",
        "from src.io_utils import RESULTS_DIR, ensure_dirs",
        "",
        "ensure_dirs()",
        "SEED = 0",
        "CFG = cot_drift.TrainConfig(n_steps=800, n_samples=64, seed=SEED)",
    ),
    code(
        "rows = []",
        "for scheme in cot_drift.ALL_SCHEMES:",
        "    has_gate  = 'gated' in scheme",
        "    has_cap   = ('capped' in scheme) or ('differential' in scheme)",
        "    has_diff  = 'differential' in scheme",
        "    h = cot_drift.run_cot_drift(scheme, CFG, cap=0.5, delta=0.05)",
        "    mu = h.policy_mu[-1]",
        "    rows.append({",
        "        'scheme': scheme,",
        "        'gate': int(has_gate),",
        "        'cap': int(has_cap),",
        "        'differential': int(has_diff),",
        "        'task_reward': round(float(h.mean_task_reward[-1]), 3),",
        "        'cot_drift_distance': round(cot_drift.cot_drift_distance(h), 3),",
        "        'reward_hacking_rate': round(cot_drift.reward_hacking_rate(h), 3),",
        "        'final_mu_quality': round(float(mu[0]), 3),",
        "        'final_mu_cot': round(float(mu[1]), 3),",
        "    })",
        "df = pd.DataFrame(rows).set_index('scheme')",
        "df",
    ),
    code(
        "csv_path = RESULTS_DIR / 'ablations.csv'",
        "df.to_csv(csv_path)",
        "print(f'wrote {csv_path}')",
        "print()",
        "print(df.to_markdown())",
    ),
    md(
        "## Reading the table",
        "",
        "* The `weighted_sum` baseline has the highest `cot_drift_distance`",
        "  and the highest `reward_hacking_rate`.",
        "* Adding only a `gate` controls priority inversion but does not",
        "  reduce CoT drift.",
        "* Adding only a `cap` reduces variance dominance but the policy",
        "  still drifts toward the CoT ridge.",
        "* `gate + cap` is the standard structured baseline.",
        "* Adding the `differential cap` is the change that reduces CoT",
        "  drift without giving up task reward.",
        "",
        "These four rows are the structural-ablation point of the paper.",
        "Numbers are deterministic for a fixed seed; the table is",
        "byte-stable across reruns at `SEED=0`.",
    ),
])


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write(name: str, nb: dict[str, object]) -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    out = NB_DIR / name
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out}")


def main() -> None:
    write("01_failures.ipynb", nb01)
    write("02_structured_composition.ipynb", nb02)
    write("03_audit.ipynb", nb03)
    write("04_ablations.ipynb", nb04)


if __name__ == "__main__":
    main()
