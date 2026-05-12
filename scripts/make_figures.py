"""Generate the figures embedded in the README.

Two plain figures, no decorative contours or colorbars:

  results/figures/hero.png     two 2D trajectory panels side by side
  results/figures/drift.png    CoT-content training curve + hacking-rate bar
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.scenarios import cot_drift  # noqa: E402

FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
CFG = cot_drift.TrainConfig(n_steps=800, n_samples=64, seed=SEED)

SCHEMES = ("weighted_sum", "capped", "gated_capped_differential")
LABELS = {
    "weighted_sum": "weighted sum",
    "capped": "cap",
    "gated_capped_differential": "cap + differential",
}
COLORS = {
    "weighted_sum": "#b42318",
    "capped": "#ca8504",
    "gated_capped_differential": "#175cd3",
}

TASK_PEAK = 0.8
COT_RIDGE = 0.75


def run_all() -> dict[str, cot_drift.TrainHistory]:
    return {s: cot_drift.run_cot_drift(s, CFG, cap=0.5, delta=0.05)
            for s in SCHEMES}


def hero(histories: dict[str, cot_drift.TrainHistory]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    panels = [
        (axes[0], "weighted_sum", "weighted sum"),
        (axes[1], "gated_capped_differential", "gate + cap + differential"),
    ]
    for ax, scheme, title in panels:
        mu = np.array(histories[scheme].policy_mu)
        ax.plot(mu[:, 0], mu[:, 1], color=COLORS[scheme], lw=1.8)
        ax.scatter([mu[0, 0]], [mu[0, 1]], color="#101828", s=48, zorder=5,
                   label="start")
        ax.scatter([mu[-1, 0]], [mu[-1, 1]], color=COLORS[scheme], s=70,
                   edgecolor="white", linewidth=1.2, zorder=6, label="end")
        ax.axvline(TASK_PEAK, color="#175cd3", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(COT_RIDGE, color="#dc6803", ls="--", lw=0.9, alpha=0.7)
        ax.text(TASK_PEAK + 0.012, 0.96, "task peak", color="#175cd3",
                fontsize=9, alpha=0.9, va="top")
        ax.text(0.02, COT_RIDGE + 0.02, "CoT ridge", color="#dc6803",
                fontsize=9, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("action quality")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    axes[0].set_ylabel("CoT content")
    fig.tight_layout()
    out = FIG_DIR / "hero.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def drift(histories: dict[str, cot_drift.TrainHistory]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))

    for s in SCHEMES:
        mu = np.array(histories[s].policy_mu)
        ax1.plot(mu[:, 1], color=COLORS[s], lw=1.5, label=LABELS[s])
    ax1.axhline(COT_RIDGE, color="#dc6803", ls="--", lw=0.9, alpha=0.6,
                label="CoT ridge")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("policy mean (CoT content)")
    ax1.set_title("CoT drift")
    ax1.set_ylim(0.2, 0.85)
    ax1.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    rates = [cot_drift.reward_hacking_rate(histories[s]) for s in SCHEMES]
    xs = np.arange(len(SCHEMES))
    bars = ax2.bar(xs, rates, color=[COLORS[s] for s in SCHEMES], width=0.55)
    for b, r in zip(bars, rates):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{r:.0%}", ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([LABELS[s] for s in SCHEMES], fontsize=9.5)
    ax2.set_ylabel("hacking rate")
    ax2.set_title("Final-policy hacking rate")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "drift.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 9.5,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#98a2b3",
        "axes.labelcolor": "#344054",
        "axes.titlesize": 11,
        "xtick.color": "#667085",
        "ytick.color": "#667085",
        "grid.color": "#eaecf0",
        "grid.linewidth": 0.7,
    })
    histories = run_all()
    hero(histories)
    drift(histories)


if __name__ == "__main__":
    main()
