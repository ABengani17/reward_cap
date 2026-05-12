"""Render the architecture diagram to docs/diagram.svg.

Shows the data flow: raw scorer stack -> rewardcap layer (gates ->
differential caps -> caps -> sum) -> bounded reward to optimizer. One
diagram, matplotlib-only, no graphviz dependency. Re-run after changes
via `make repro` or directly:

    python scripts/draw_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Color palette
    raw_color = "#cfd8dc"
    gate_color = "#ef9a9a"
    cap_color = "#90caf9"
    diff_color = "#a5d6a7"
    sum_color = "#ffe082"
    bg_color = "#fafafa"
    text_color = "#212121"

    # Background panel for the rewardcap layer
    layer = mpatches.FancyBboxPatch(
        (2.6, 0.8),
        6.0,
        4.3,
        boxstyle="round,pad=0.04",
        linewidth=1.2,
        edgecolor="#90a4ae",
        facecolor=bg_color,
    )
    ax.add_patch(layer)
    ax.text(
        5.6, 5.25, "rewardcap layer",
        ha="center", va="bottom", fontsize=12, weight="bold",
        color="#37474f",
    )

    # Left: raw scorers
    raw_labels = [
        "format check",
        "exact-match correctness",
        "LLM judge (CoT-aware)",
        "LLM judge (CoT-blind companion)",
        "length penalty",
    ]
    for i, label in enumerate(raw_labels):
        y = 4.5 - i * 0.85
        box = mpatches.FancyBboxPatch(
            (0.1, y - 0.3),
            2.2,
            0.6,
            boxstyle="round,pad=0.02",
            linewidth=0.8,
            edgecolor="#90a4ae",
            facecolor=raw_color,
        )
        ax.add_patch(box)
        ax.text(
            1.2, y, label,
            ha="center", va="center", fontsize=9, color=text_color,
        )
        # Arrow into the layer
        ax.annotate(
            "",
            xy=(2.7, y),
            xytext=(2.3, y),
            arrowprops=dict(arrowstyle="->", color="#607d8b", lw=1.0),
        )

    # Inside the layer: gate, differential cap, cap, sum
    def labelled_box(
        x: float, y: float, w: float, h: float,
        face: str, top: str, bottom: str = "",
    ) -> None:
        b = mpatches.FancyBboxPatch(
            (x, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=0.8,
            edgecolor="#546e7a",
            facecolor=face,
        )
        ax.add_patch(b)
        ax.text(
            x + w / 2, y + 0.07, top,
            ha="center", va="center", fontsize=10, weight="bold",
            color=text_color,
        )
        if bottom:
            ax.text(
                x + w / 2, y - 0.18, bottom,
                ha="center", va="center", fontsize=8.5, color="#37474f",
            )

    labelled_box(
        2.9, 4.0, 1.7, 0.9,
        gate_color, "Gate G", r"[ $\bigwedge_g g(x)=1$ ]",
    )
    labelled_box(
        2.9, 2.7, 1.7, 0.9,
        diff_color, "DifferentialCap",
        r"$s_{\rm blind}+{\rm clip}(s_{\rm aware}-s_{\rm blind},\pm\delta)$",
    )
    labelled_box(
        2.9, 1.4, 1.7, 0.9,
        cap_color, "Cap", r"$\min(c_s, s(x))$",
    )

    labelled_box(
        5.4, 2.7, 1.6, 0.9,
        sum_color, r"$\sum$", r"sum of capped scorers",
    )

    labelled_box(
        7.4, 2.7, 1.0, 0.9,
        gate_color, r"$\cdot$", "gate indicator",
    )

    # Wiring inside the layer
    ax.annotate(
        "", xy=(5.4, 2.7), xytext=(4.6, 4.0),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0),
    )
    ax.annotate(
        "", xy=(5.4, 2.7), xytext=(4.6, 2.7),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0),
    )
    ax.annotate(
        "", xy=(5.4, 2.7), xytext=(4.6, 1.4),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0),
    )
    ax.annotate(
        "", xy=(7.4, 2.7), xytext=(7.0, 2.7),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0),
    )

    # Right: bounded reward to optimizer
    bound_box = mpatches.FancyBboxPatch(
        (9.0, 2.4),
        1.8,
        0.6,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#37474f",
        facecolor="#fff9c4",
    )
    ax.add_patch(bound_box)
    ax.text(
        9.9, 2.7, "bounded R(x)",
        ha="center", va="center", fontsize=10, weight="bold",
        color=text_color,
    )
    ax.annotate(
        "", xy=(9.0, 2.7), xytext=(8.4, 2.7),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.2),
    )

    opt_box = mpatches.FancyBboxPatch(
        (9.0, 1.2),
        1.8,
        0.7,
        boxstyle="round,pad=0.02",
        linewidth=1.0,
        edgecolor="#37474f",
        facecolor="#eceff1",
    )
    ax.add_patch(opt_box)
    ax.text(
        9.9, 1.55, "policy gradient\n(PPO / GRPO)",
        ha="center", va="center", fontsize=9, color=text_color,
    )
    ax.annotate(
        "", xy=(9.9, 1.95), xytext=(9.9, 2.4),
        arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.0),
    )

    # Caption
    ax.text(
        5.5, 0.35,
        "Constraint-projection composition: scorers pass through the layer "
        "before reaching the optimizer.",
        ha="center", va="center", fontsize=9.5, style="italic",
        color="#37474f",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "diagram.svg"
    draw(out)
    print(f"wrote {out}")
