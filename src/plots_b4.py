"""Benchmark 4: reasoning helps only inside a discussion; sharing it helps reliably."""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs, wilson
from config import FIGURES_DIR


def maj(rd):
    v = Counter(x for x in rd.values() if x)
    if not v:
        return None
    best = max(v.values())
    return sorted(l for l, c in v.items() if c == best)[0]


def main() -> None:
    b4 = load_logs(pathlib.Path("../logs/b4/shared.jsonl"))
    b3 = {c: load_logs(pathlib.Path(f"../logs/b3/{c}.jsonl"))
          for c in ["baseline"]}
    # The clean private-reasoning control comes from the ablation: identical
    # generation to the shared arm, peers see only the answer. (b3 critical_cot
    # is not a substitute -- it also carries the critical-reasoning instruction.)
    priv = load_logs(pathlib.Path("../logs/b4/answer_only.jsonl"))
    n = len(b4)

    rows = [
        ("no reasoning\nno discussion",
         sum(maj(r["rounds"][0]) == r["correct_letter"] for r in b3["baseline"]), "#8c8c8c"),
        ("reasoning\nno discussion",
         sum(r["round0_correct"] for r in b4), "#8c8c8c"),
        ("no reasoning\nanswers shared",
         sum(r["majority_correct"] for r in b3["baseline"]), "#c0504d"),
        ("reasoning private\nanswers shared",
         sum(r["majority_correct"] for r in priv), "#c2a25a"),
        ("reasoning\nSHARED",
         sum(r["majority_correct"] for r in b4), "#2e8b57"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    xs = np.arange(len(rows))
    vals = [k / n for _, k, _ in rows]
    err = np.array([[v - wilson(k, n)[0], wilson(k, n)[1] - v]
                    for v, (_, k, _) in zip(vals, rows)]).T
    bars = ax.bar(xs, vals, .62, color=[c for _, _, c in rows],
                  yerr=err, capsize=4, error_kw=dict(lw=1, alpha=.65))
    base = rows[0][1] / n
    ax.axhline(base, ls="--", lw=1.3, color="#333", alpha=.8)
    ax.annotate("accuracy from\nanswering alone", xy=(0.42, base), xytext=(0.75, 0.93),
                fontsize=8.5, color="#444", ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#777", lw=1))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .052, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _, _ in rows], fontsize=8.5)
    ax.set_ylabel("majority-vote accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Both reasoning configurations beat answering alone", fontsize=11.5)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)

    # --- right: the decomposition, drawn as a waterfall
    #
    # Drawn as cumulative steps rather than three bars from a common zero. The
    # two discussion contrasts are increments on one another (0.707 -> 0.767 ->
    # 0.820) and bars sharing a baseline hide that; a waterfall makes the second
    # step visibly start where the first one ended.
    ax = axes[1]
    fin_none = np.array([r["majority_correct"] for r in b3["baseline"]], float)
    fin_priv = np.array([r["majority_correct"] for r in priv], float)
    fin_shar = np.array([r["majority_correct"] for r in b4], float)

    def paired(a, b, n_boot=5000, seed=12345):
        rng = np.random.default_rng(seed)
        m = len(a)
        d = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, m, m)
            d[i] = a[idx].mean() - b[idx].mean()
        return a.mean() - b.mean(), (float(np.percentile(d, 2.5)),
                                     float(np.percentile(d, 97.5)))

    base = fin_none.mean()
    d1, ci1 = paired(fin_priv, fin_none)     # generating it, peers see answers only
    d2, ci2 = paired(fin_shar, fin_priv)     # sharing that same reasoning
    lvl = [base, base + d1, base + d1 + d2]

    xs = [0, 1, 2, 3]
    ax.bar(0, base, .62, color="#c0504d")
    ax.bar(1, d1, .62, bottom=lvl[0], color="#c2a25a")
    ax.bar(2, d2, .62, bottom=lvl[1], color="#2e8b57")
    ax.bar(3, lvl[2], .62, color="none", edgecolor="#2e8b57", lw=2, hatch="//")

    # connectors, so each step visibly resumes where the previous one stopped
    for i, y in enumerate(lvl):
        ax.plot([i + .31, i + 1 - .31], [y, y], color="#555", lw=1, ls=":")
    ax.plot([2 + .31, 3 - .31], [lvl[2], lvl[2]], color="#555", lw=1, ls=":")

    # each increment's interval, positioned at the level the step reaches
    for i, (d, (lo, hi), start_lvl) in enumerate(
            [(d1, ci1, lvl[0]), (d2, ci2, lvl[1])], start=1):
        ax.plot([i, i], [start_lvl + lo, start_lvl + hi], color="#222", lw=1.8)
        for e in (lo, hi):
            ax.plot([i - .07, i + .07], [start_lvl + e] * 2, color="#222", lw=1.8)
        ax.text(i, start_lvl + hi + .004, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                fontsize=8.5, va="bottom", ha="center", fontweight="bold")

    # running level, placed on the connector so it never collides with a bar
    for i, y in enumerate(lvl):
        ax.text(i + .5, y, f"{y:.3f}", ha="center", va="center", fontsize=8.5,
                color="#444",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    ax.text(0, base + .004, f"{base:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
    ax.text(3, lvl[2] + .004, f"{lvl[2]:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(["discussion,\nanswers only",
                        "+ agents reason\n(peers still see\nonly answers)",
                        "+ peers see\nthe reasoning",
                        "reasoning\nSHARED"], fontsize=9)
    ax.set_ylim(0.63, 0.90)
    ax.set_xlim(-0.75, 3.55)
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title(f"The {d1 + d2:+.3f} gap, decomposed into two steps", fontsize=11.5)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)

    fig.suptitle("It isn't just the reasoning — it's letting the other agents see it",
                 fontsize=13.5)
    fig.tight_layout()
    out = FIGURES_DIR / "benchmark4.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    for lab, k, _ in rows:
        print(f"  {lab.replace(chr(10),' '):36}{k/n:.3f}")


if __name__ == "__main__":
    main()
