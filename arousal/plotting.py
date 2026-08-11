"""Shared figure style and small plotting helpers."""

from __future__ import annotations

import numpy as np

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#27ae60"
CRIMSON = "#c0392b"
GRAY = "#888888"
DKRED = "#7B1F1F"
DKBLUE = "#1A3A5C"
ORANGE = "#DD8452"
SEAGREEN = "#55A868"
PURPLE = "#8172B3"

FS_AX = 11      # axis labels
FS_TK = 10      # tick labels
FS_AN = 10      # in-axes annotations
FS_PAN = 12     # panel letters


def tidy(ax, grid_axis="y"):
    """Drop the top/right spines and put a dashed grid behind the data."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TK)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PAN,
            fontweight="bold", va="top")


def fmt_p(p, thresh=0.001):
    return f"p < {thresh}" if p < thresh else f"p = {p:.3f}"


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (len(p) - rank))
        adjusted[idx] = min(running, 1.0)
    return adjusted
