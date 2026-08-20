"""The two figures that carry the result."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs, signal_detection, two_proportion_z, wilson
from config import FIGURES_DIR

ORDER = ["baseline", "warning_only", "bss", "critical", "critical_cot", "critical_bss"]
PRETTY = {
    "baseline": "baseline",
    "warning_only": "warning\nonly",
    "bss": "BSS prior\n(paper)",
    "critical": "critical",
    "critical_cot": "critical\n+ CoT",
    "critical_bss": "critical\n+ BSS",
}
COLORS = {
    "baseline": "#8c8c8c", "warning_only": "#c2a25a", "bss": "#c0504d",
    "critical": "#3b7dd8", "critical_cot": "#4fa3a5", "critical_bss": "#7a5ea8",
}


def _acc(split: str, cond: str) -> tuple[int, int]:
    logs = load_logs(pathlib.Path(f"../logs/{split}/{cond}.jsonl"))
    return sum(r["majority_correct"] for r in logs), len(logs)


def fig_headline(out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (split, title) in zip(
        axes,
        [("main", "Paper protocol\n(user is always wrong)"),
         ("balanced", "Balanced control\n(user is right half the time)")],
    ):
        base = _acc(split, "baseline")
        vals, los, his, stars = [], [], [], []
        for c in ORDER:
            k, n = _acc(split, c)
            lo, hi = wilson(k, n)
            vals.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
            p = two_proportion_z(k, n, *base)
            stars.append("*" if (p < 0.05 and c != "baseline") else "")
        bars = ax.bar(range(len(ORDER)), vals, color=[COLORS[c] for c in ORDER],
                      yerr=[los, his], capsize=4, error_kw=dict(lw=1, alpha=.7))
        ax.axhline(base[0] / base[1], ls="--", lw=1.2, color="#333", alpha=.8)
        for b, v, s in zip(bars, vals, stars):
            ax.text(b.get_x() + b.get_width() / 2, v + max(his) + .022,
                    f"{v:.3f}{s}", ha="center", fontsize=9,
                    fontweight="bold" if s else "normal")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([PRETTY[c] for c in ORDER], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    axes[0].set_ylabel("majority-vote accuracy")
    fig.suptitle(
        "Every intervention wins when the user is always wrong. None survives the control.",
        fontsize=13, y=1.0,
    )
    fig.text(.5, -.03, "* = p < 0.05 vs baseline;  dashed line = baseline;  error bars Wilson 95% CI",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_signal_detection(out: pathlib.Path) -> None:
    sd = {c: signal_detection(load_logs(pathlib.Path(f"../logs/balanced/{c}.jsonl")))
          for c in ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    x = np.arange(len(ORDER))
    ax.bar(x - .2, [sd[c]["hit_rate"] for c in ORDER], .4,
           label="correctly rejects a wrong user (hit)", color="#3b7dd8")
    ax.bar(x + .2, [sd[c]["false_alarm_rate"] for c in ORDER], .4,
           label="wrongly rejects a right user (false alarm)", color="#c0504d")
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[c] for c in ORDER], fontsize=8)
    ax.set_ylabel("rate"); ax.legend(fontsize=8.5)
    ax.set_title("Interventions raise BOTH kinds of rejection", fontsize=11)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)

    ax = axes[1]
    base = sd["baseline"]
    for c in ORDER:
        ax.scatter(sd[c]["criterion"], sd[c]["d_prime"], s=150, color=COLORS[c],
                   zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(PRETTY[c].replace("\n", " "),
                    (sd[c]["criterion"], sd[c]["d_prime"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5)
    ax.axhline(base["d_prime"], ls="--", lw=1.2, color="#333", alpha=.7)
    ax.set_xlabel("criterion  ←  more willing to reject the user")
    ax.set_ylabel("d′  (ability to tell right from wrong)")
    ax.set_title("All movement is sideways, not up", fontsize=11)
    ax.set_ylim(0.6, 1.4)
    ax.grid(alpha=.25); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_headline(FIGURES_DIR / "headline.png")
    fig_signal_detection(FIGURES_DIR / "signal_detection.png")
    print(f"wrote headline.png and signal_detection.png to {FIGURES_DIR}")
