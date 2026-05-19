"""Plotting defaults and a savefig wrapper that always emits PNG+PDF at >=600 dpi."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_style(base_size: int = 14) -> None:
    """Apply a clean, presentation-friendly default style.

    Mirrors the R preference for theme_classic + large base_size, minimal
    padding, print-and-screen-safe colours.
    """
    mpl.rcParams.update({
        "font.size": base_size,
        "axes.titlesize": base_size + 2,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,  # editable text in vector PDFs
        "ps.fonttype": 42,
    })


def savefig(fig: plt.Figure, path: str | Path, *, dpi: int = 600) -> tuple[Path, Path]:
    """Save the figure as both PNG and PDF (high-res). Returns the two paths."""
    path = Path(path)
    stem = path.with_suffix("")
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf
