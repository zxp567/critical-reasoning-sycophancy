"""Benchmark 4 trajectories: majority, per-agent, and who gains from the channel."""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs
from config import FIGURES_DIR

# weakest to strongest by neutral MCQ accuracy
ORDER = ["llama3b", "llama8b", "gemma4b", "qwen7b", "llama70b", "qwen72b"]
COLORS = {"llama3b": "#d1495b", "llama8b": "#e08b3c", "gemma4b": "#c9a227",
          "qwen7b": "#4c9f70", "llama70b": "#3b7dd8", "qwen72b": "#6a4c93"}


def maj(rd):
    v = Counter(x for x in rd.values() if x)
    if not v:
        return None
    best = max(v.values())
    return sorted(l for l, c in v.items() if c == best)[0]


def traj_majority(logs):
    n = len(logs)
    return [sum(maj(r["rounds"][i]) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def traj_agent(logs, a):
    n = len(logs)
    return [sum(r["rounds"][i].get(a) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def main() -> None:
    b4 = load_logs(pathlib.Path("../logs/b4/shared.jsonl"))
    b3 = load_logs(pathlib.Path("../logs/b3/baseline.jsonl"))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    # (a) majority vote, both channels
    ax = axes[0, 0]
    for logs, lab, col, ls in [(b3, "answers only (B3)", "#c0504d", "--"),
                               (b4, "reasoning shared (B4)", "#2e8b57", "-")]:
        t = traj_majority(logs)
        ax.plot(range(len(t)), t, marker="o", ms=6, lw=2.6, color=col, label=lab, ls=ls)
    ax.axhline(traj_majority(b3)[0], color="#333", lw=1.1, ls=":", alpha=.8)
    ax.text(2.0, traj_majority(b3)[0] - .028, "answering alone", fontsize=8.5,
            color="#444", ha="center")
    ax.set_xticks(range(5))
    ax.set_xlabel("discussion round")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title("(a) Majority vote", fontsize=11.5)
    ax.set_ylim(0.62, 0.87)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=.25); ax.set_axisbelow(True)

    # (b) net change per agent, both channels
    ax = axes[0, 1]
    x = np.arange(len(ORDER))
    d3 = [traj_agent(b3, a)[-1] - traj_agent(b3, a)[0] for a in ORDER]
    d4 = [traj_agent(b4, a)[-1] - traj_agent(b4, a)[0] for a in ORDER]
    ax.bar(x - .2, d3, .4, color="#c0504d", label="answers only (B3)")
    ax.bar(x + .2, d4, .4, color="#2e8b57", label="reasoning shared (B4)")
    ax.axhline(0, color="#333", lw=1.2)
    for xi, v in zip(x - .2, d3):
        ax.text(xi, v + (.009 if v >= 0 else -.026), f"{v:+.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + .2, d4):
        ax.text(xi, v + (.009 if v >= 0 else -.026), f"{v:+.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=20, fontsize=8.5)
    ax.set_ylabel("accuracy change, round 0 $\\rightarrow$ 4")
    ax.set_title("(b) Who gains — and who gets dragged down", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(-0.16, 0.42)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
    ax.annotate("only the strongest\ntwo agents", xy=(4.5, -0.075), xytext=(3.6, -0.145),
                fontsize=7.5, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=.9))

    # (c) and (d): per-agent trajectories, shared axis
    for ax, logs, title in [(axes[1, 0], b3, "(c) Each agent — answers only (B3)"),
                            (axes[1, 1], b4, "(d) Each agent — reasoning shared (B4)")]:
        for a in ORDER:
            t = traj_agent(logs, a)
            ax.plot(range(len(t)), t, marker="o", ms=4.5, lw=2, color=COLORS[a], label=a)
        t = traj_majority(logs)
        ax.plot(range(len(t)), t, lw=3, color="#111", ls="--", label="majority", zorder=5)
        ax.set_xticks(range(5))
        ax.set_xlabel("discussion round")
        ax.set_title(title, fontsize=11.5)
        ax.set_ylim(0.38, 0.88)
        ax.grid(alpha=.25); ax.set_axisbelow(True)
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].legend(fontsize=7.5, ncol=2, loc="lower right")

    fig.suptitle("Sharing reasoning lifts weak agents without dragging the strong ones down",
                 fontsize=13.5)
    fig.tight_layout()
    out = FIGURES_DIR / "benchmark4_trajectory.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}\n")
    print(f"{'agent':10}{'B3 change':>12}{'B4 change':>12}{'difference':>13}")
    for a, x3, x4 in zip(ORDER, d3, d4):
        print(f"{a:10}{x3:+12.3f}{x4:+12.3f}{x4-x3:+13.3f}")


if __name__ == "__main__":
    main()
