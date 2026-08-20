"""Sycophancy falls; accuracy does not follow. Balanced control set."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs, post_discussion_sycophancy, wilson
from config import AGENTS, FIGURES_DIR, SCORES_DIR
import json

ORDER = ["baseline", "warning_only", "bss", "critical", "critical_cot", "critical_bss"]
PRETTY = {"baseline": "baseline", "warning_only": "warning only", "bss": "BSS prior",
          "critical": "critical", "critical_cot": "critical+CoT", "critical_bss": "critical+BSS"}
COLORS = {"baseline": "#8c8c8c", "warning_only": "#c2a25a", "bss": "#c0504d",
          "critical": "#3b7dd8", "critical_cot": "#4fa3a5", "critical_bss": "#7a5ea8"}


def main() -> None:
    knowledge = json.loads((SCORES_DIR / "probes_test.json").read_text())["knowledge"]
    syc, acc = {}, {}
    for c in ORDER:
        logs = load_logs(pathlib.Path(f"../logs/balanced/{c}.jsonl"))
        s = post_discussion_sycophancy(logs, knowledge, None)
        tot_s = sum(v[0] for v in s.values()); tot_k = sum(v[1] for v in s.values())
        syc[c] = tot_s / tot_k
        n = k = 0
        for r in logs:
            for a in AGENTS:
                v = r["rounds"][-1].get(a)
                if v is None:
                    continue
                n += 1; k += v == r["correct_stance"]
        acc[c] = k / n

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    x = np.arange(len(ORDER))
    ax.bar(x - .2, [syc[c] for c in ORDER], .4, label="sycophancy (lower = better)",
           color="#c0504d")
    ax.bar(x + .2, [acc[c] for c in ORDER], .4, label="accuracy (higher = better)",
           color="#3b7dd8")
    ax.axhline(syc["baseline"], color="#c0504d", ls="--", lw=1.2, alpha=.7)
    ax.axhline(acc["baseline"], color="#3b7dd8", ls="--", lw=1.2, alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[c] for c in ORDER], fontsize=8.5, rotation=12)
    ax.set_ylim(0, .85); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Sycophancy roughly halves. Accuracy does not move.", fontsize=11)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)

    ax = axes[1]
    for c in ORDER:
        dx = syc[c] - syc["baseline"]
        dy = acc[c] - acc["baseline"]
        ax.scatter(dx, dy, s=170, color=COLORS[c], zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(PRETTY[c], (dx, dy), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5)
    ax.axhline(0, color="#333", lw=1); ax.axvline(0, color="#333", lw=1)
    lim = .19
    ax.plot([-lim, 0], [lim, 0], ls=":", lw=1.5, color="#2a7", zorder=1)
    ax.text(-.155, .105, "if less sycophancy\nmeant more accuracy,\npoints would lie here",
            fontsize=8.5, color="#2a7", ha="center")
    ax.set_xlim(-lim, .05); ax.set_ylim(-.09, lim)
    ax.set_xlabel("change in sycophancy  (← less sycophantic)")
    ax.set_ylabel("change in accuracy")
    ax.set_title("All the movement is horizontal", fontsize=11)
    ax.grid(alpha=.25); ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "sycophancy_vs_accuracy.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    for c in ORDER:
        print(f"  {c:14} sycophancy {syc[c]:.3f}   accuracy {acc[c]:.3f}")


if __name__ == "__main__":
    main()
