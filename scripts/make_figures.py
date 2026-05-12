"""Generate the figures embedded in the README.

Two figures:
  results/figures/hero.png     -- reward landscape + two trajectories
  results/figures/drift.png    -- CoT-axis drift over training, per scheme

The five-scheme grid in the notebooks remains for reproducing the full
ablation. These two figures are the visuals the README actually uses.
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


def landscape() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.linspace(0.0, 1.0, 200)
    Q, C = np.meshgrid(g, g)
    Z = np.exp(-6.0 * (Q - 0.8) ** 2) + 0.9 * np.exp(-40.0 * (C - 0.75) ** 2)
    return Q, C, Z


def hero() -> None:
    Q, C, Z = landscape()
    h_ws = cot_drift.run_cot_drift("weighted_sum", CFG, cap=0.5, delta=0.05)
    h_dc = cot_drift.run_cot_drift(
        "gated_capped_differential", CFG, cap=0.5, delta=0.05
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)
    cs = None
    for ax, h, title in [
        (axes[0], h_ws, "weighted sum"),
        (axes[1], h_dc, "gate + cap + differential cap"),
    ]:
        cs = ax.contourf(Q, C, Z, levels=16, cmap="Greys", alpha=0.6)
        ax.contour(Q, C, Z, levels=10, colors="#667085", linewidths=0.5,
                   alpha=0.55)
        mu = np.array(h.policy_mu)
        ax.plot(mu[:, 0], mu[:, 1], color="#b42318", lw=2.4, zorder=4)
        ax.scatter([mu[0, 0]], [mu[0, 1]], color="#101828", s=70, zorder=5,
                   label="start")
        ax.scatter([mu[-1, 0]], [mu[-1, 1]], color="#175cd3", s=110, zorder=5,
                   edgecolor="white", linewidth=1.4, label="end")
        ax.axvline(0.8, color="#175cd3", ls=":", alpha=0.6, lw=1.0)
        ax.axhline(0.75, color="#dc6803", ls=":", alpha=0.6, lw=1.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("action quality", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    axes[0].set_ylabel("CoT content", fontsize=11)

    axes[0].annotate("spurious CoT ridge\n(judge-only signal)",
                     xy=(0.55, 0.75), xytext=(0.04, 0.95),
                     fontsize=10, color="#7a2e0e",
                     arrowprops=dict(arrowstyle="->", color="#7a2e0e", lw=1.0))
    axes[1].annotate("task peak\n(quality = 0.8)",
                     xy=(0.8, 0.45), xytext=(0.4, 0.08),
                     fontsize=10, color="#175cd3",
                     arrowprops=dict(arrowstyle="->", color="#175cd3", lw=1.0))

    fig.subplots_adjust(left=0.06, right=0.91, wspace=0.08, top=0.9)
    cax = fig.add_axes([0.925, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(cs, cax=cax)
    cbar.set_label("CoT-aware reward landscape", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Policy mean during training, plotted on the CoT-aware reward landscape",
        fontsize=13, y=0.97,
    )
    out = FIG_DIR / "hero.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def drift() -> None:
    schemes = ["weighted_sum", "capped", "gated_capped_differential"]
    labels = {
        "weighted_sum": "weighted sum",
        "capped": "cap",
        "gated_capped_differential": "cap + differential",
    }
    colors = {
        "weighted_sum": "#b42318",
        "capped": "#ca8504",
        "gated_capped_differential": "#175cd3",
    }
    histories = {
        s: cot_drift.run_cot_drift(s, CFG, cap=0.5, delta=0.05) for s in schemes
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for s in schemes:
        mu = np.array(histories[s].policy_mu)
        ax1.plot(mu[:, 1], color=colors[s], lw=1.6, label=labels[s])
    ax1.axhline(0.75, color="#dc6803", ls=":", alpha=0.7, lw=1.0,
                label="CoT ridge")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("CoT content (policy mean)")
    ax1.set_title("CoT drift over training")
    ax1.set_ylim(0.2, 0.85)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(True, alpha=0.35)

    # final-state bar chart for hacking rate
    rates = [cot_drift.reward_hacking_rate(histories[s]) for s in schemes]
    xs = np.arange(len(schemes))
    bars = ax2.bar(xs, rates, color=[colors[s] for s in schemes], width=0.55)
    for b, r in zip(bars, rates):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"{r:.0%}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([labels[s] for s in schemes], fontsize=9)
    ax2.set_ylabel("reward-hacking rate")
    ax2.set_title("Final-policy hacking rate")
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "drift.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#98a2b3",
        "axes.labelcolor": "#344054",
        "xtick.color": "#667085",
        "ytick.color": "#667085",
        "grid.color": "#eaecf0",
        "grid.linewidth": 0.8,
    })
    hero()
    drift()


if __name__ == "__main__":
    main()
