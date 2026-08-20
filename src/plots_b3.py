"""Benchmark 3 and the three-benchmark synthesis."""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs, round_trajectory, wilson
from config import AGENTS, FIGURES_DIR

COLORS = {"baseline": "#8c8c8c", "bss": "#c0504d", "critical": "#3b7dd8",
          "critical_cot": "#4fa3a5"}
PRETTY = {"baseline": "baseline", "bss": "BSS prior", "critical": "critical",
          "critical_cot": "critical + CoT"}


def b3_traj(cond: str) -> list[float]:
    """Majority-letter accuracy per round (4-way MCQ, no user)."""
    logs = load_logs(pathlib.Path(f"../logs/b3/{cond}.jsonl"))
    out = []
    for ri in range(len(logs[0]["rounds"])):
        ok = 0
        for r in logs:
            votes = Counter(v for v in r["rounds"][ri].values() if v)
            if not votes:
                continue
            best = max(votes.values())
            maj = sorted(l for l, c in votes.items() if c == best)[0]
            ok += maj == r["correct_letter"]
        out.append(ok / len(logs))
    return out


def main() -> None:
    conds = ["baseline", "bss", "critical", "critical_cot"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: benchmark 3 trajectories
    ax = axes[0]
    for c in conds:
        t = b3_traj(c)
        ax.plot(range(len(t)), t, marker="o", ms=5, lw=2.2, color=COLORS[c],
                label=PRETTY[c], ls="--" if c == "baseline" else "-")
    ax.axhline(0.25, color="#b00", lw=1, ls=":", alpha=.7)
    ax.text(0.06, 0.263, "chance (4-way)", fontsize=8, color="#b00")
    ax.set_xticks(range(5))
    ax.set_xlabel("discussion round")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title("Benchmark 3 — no user at all, agents answer A/B/C/D", fontsize=11)
    ax.set_ylim(0.22, 0.82)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)

    # --- right: what discussion does to accuracy, per benchmark
    ax = axes[1]
    labels = ["Benchmark 1\nuser always wrong", "Benchmark 2\nuser right 50%",
              "Benchmark 3\nno user"]
    deltas = {}
    for cond in ["baseline", "critical"]:
        d = []
        for split in ["main", "balanced"]:
            t = round_trajectory(load_logs(pathlib.Path(f"../logs/{split}/{cond}.jsonl")))["majority"]
            d.append(t[-1] - t[0])
        t3 = b3_traj(cond)
        d.append(t3[-1] - t3[0])
        deltas[cond] = d

    x = np.arange(3)
    ax.bar(x - .19, deltas["baseline"], .38, color=COLORS["baseline"], label="baseline")
    ax.bar(x + .19, deltas["critical"], .38, color=COLORS["critical"], label="critical")
    ax.axhline(0, color="#333", lw=1.2)
    for i, cond in enumerate(["baseline", "critical"]):
        for j, v in enumerate(deltas[cond]):
            off = -.19 if i == 0 else .19
            ax.text(j + off, v + (0.006 if v >= 0 else -0.017), f"{v:+.3f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy change from discussion\n(round 4 − round 0)")
    ax.set_title("What five rounds of discussion buy you", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "benchmark3.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")
    for c in conds:
        t = b3_traj(c)
        print(f"  {c:14}", " → ".join(f"{v:.3f}" for v in t))


if __name__ == "__main__":
    main()
