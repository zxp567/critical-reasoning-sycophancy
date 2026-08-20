"""Explainer figure: what d' and criterion mean, using our actual numbers."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR


def _norm(x, mu):
    return np.exp(-((x - mu) ** 2) / 2) / math.sqrt(2 * math.pi)


def panel(ax, dprime, crit, title, sub):
    x = np.linspace(-4, 5.5, 800)
    # "user is right" distribution at 0, "user is wrong" shifted up by d'
    right, wrong = _norm(x, 0.0), _norm(x, dprime)
    # criterion is measured from the midpoint between the two distributions
    thresh = dprime / 2 - crit

    ax.fill_between(x, right, color="#c0504d", alpha=.28)
    ax.fill_between(x, wrong, color="#3b7dd8", alpha=.28)
    ax.plot(x, right, color="#c0504d", lw=2)
    ax.plot(x, wrong, color="#3b7dd8", lw=2)

    # shade the two error/success regions past the threshold
    m = x >= thresh
    ax.fill_between(x[m], right[m], color="#c0504d", alpha=.75)
    ax.fill_between(x[m], wrong[m], color="#3b7dd8", alpha=.6)

    ax.axvline(thresh, color="#111", lw=2.2, ls="--")
    ax.text(thresh, .60, "threshold\n(criterion)", fontsize=8.5, va="top", ha="center")
    ax.annotate("", xy=(0, .44), xytext=(dprime, .44),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color="#333"))
    ax.text(dprime / 2, .455, f"d′ = {dprime:.2f}", ha="center", fontsize=9.5,
            fontweight="bold")

    ax.text(-2.9, .30, "user is\nRIGHT", color="#c0504d", fontsize=9,
            ha="center", fontweight="bold")
    ax.text(dprime + 2.0, .30, "user is\nWRONG", color="#3b7dd8", fontsize=9,
            ha="center", fontweight="bold")
    ax.text(thresh + .12, .06, "→ agent says\n   'incorrect'", fontsize=8, color="#222")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(sub, fontsize=9)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_ylim(0, .63); ax.set_xlim(-4, 5.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    panel(axes[0], 1.057, 0.128, "baseline",
          "hit 0.656   false alarm 0.256   →  accuracy 0.700")
    panel(axes[1], 0.982, -0.541, "BSS prior (paper's method)",
          "hit 0.849   false alarm 0.520   →  accuracy 0.664")
    fig.suptitle(
        "The distributions barely move (d′ 1.06 → 0.98). The threshold slides left.",
        fontsize=13)
    fig.text(.5, -.04,
             "Shaded-dark areas are the agent rejecting the user: blue = correctly, red = wrongly. "
             "BSS captures more blue, but even more red.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    out = FIGURES_DIR / "sdt_explainer.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
