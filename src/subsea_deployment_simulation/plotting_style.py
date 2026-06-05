"""Common plotting style helpers for SCI_paper scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_times_new_roman_style() -> None:
    """Apply a publication-style Matplotlib theme centered on Times New Roman.

    If Times New Roman is unavailable on the host machine, Matplotlib will
    automatically fall back to the next available serif font.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )
