"""Filesystem helpers shared by notebooks and experiments.

Seed plumbing requirement: every random call uses a seed passed via a
config, and the seed is stamped into output filenames so a `make repro`
that re-runs at the same seed is byte-stable. Use `seeded_path` to
build figure/CSV output paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create results/ and results/figures/ if missing."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def seeded_path(
    name: str,
    seed: int,
    ext: str = "png",
    subdir: str | None = "figures",
) -> Path:
    """Build a results path with the seed stamped into the filename.

    Example:
        seeded_path("02_cot_drift", seed=42)  # => results/figures/02_cot_drift_seed42.png
    """
    ensure_dirs()
    base = RESULTS_DIR if subdir is None else RESULTS_DIR / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}_seed{seed}.{ext}"


__all__ = [
    "REPO_ROOT",
    "RESULTS_DIR",
    "FIGURES_DIR",
    "ensure_dirs",
    "seeded_path",
]
