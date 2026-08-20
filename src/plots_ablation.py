"""Per-agent trajectories across the channel ablation and the strong-roster replication."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs
from config import FIGURES_DIR, ROSTERS

MAIN = ["llama3b", "llama8b", "gemma4b", "qwen7b", "llama70b", "qwen72b"]
STRONG = list(ROSTERS["strong"])
COLORS = {"llama3b": "#d1495b", "llama8b": "#e08b3c", "gemma4b": "#c9a227",
          "qwen7b": "#4c9f70", "llama70b": "#3b7dd8", "qwen72b": "#6a4c93",
          "gemma12b": "#d1495b", "qwen3-8b": "#e08b3c", "qwen3-30b": "#c9a227",
          "qwen3next80b": "#4c9f70"}


def traj(logs, a):
    n = len(logs)
    return [sum(r["rounds"][i].get(a) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def maj_traj(logs):
    from collections import Counter
    n = len(logs)
    out = []
    for i in range(len(logs[0]["rounds"])):
        ok = 0
        for r in logs:
            v = Counter(x for x in r["rounds"][i].values() if x)
            if not v:
                continue
            b = max(v.values())
            ok += sorted(l for l, c in v.items() if c == b)[0] == r["correct_letter"]
        out.append(ok / n)
    return out


def panel(ax, logs, agents, title, note=None):
    for a in agents:
        t = traj(logs, a)
        ax.plot(range(len(t)), t, marker="o", ms=4, lw=1.9, color=COLORS.get(a, "#888"),
                label=a)
    m = maj_traj(logs)
    ax.plot(range(len(m)), m, lw=3, color="#111", ls="--", label="majority", zorder=5)
    ax.set_xticks(range(5))
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0.38, 0.90)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)
    if note:
        ax.text(.03, .97, note, transform=ax.transAxes, fontsize=7, color="#a33",
                va="top")


def main() -> None:
    L = {c: load_logs(pathlib.Path(f"../logs/b4/{c}.jsonl"))
         for c in ["answer_only", "confidence", "rationale", "shared"]}
    S = load_logs(pathlib.Path("../logs/b4_strong/shared.jsonl"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    panel(axes[0, 0], L["answer_only"], MAIN, "(a) peers see: answer only")
    panel(axes[0, 1], L["confidence"], MAIN, "(b) peers see: answer + confidence",
          note="round 0 differs — separate generation")
    panel(axes[0, 2], L["rationale"], MAIN, "(c) peers see: answer + one sentence")
    panel(axes[1, 0], L["shared"], MAIN, "(d) peers see: answer + full reasoning")
    panel(axes[1, 1], S, STRONG, "(e) full reasoning — STRONG roster")

    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("accuracy")
    for ax in axes[1][:2]:
        ax.set_xlabel("discussion round")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    axes[1, 1].legend(fontsize=7, ncol=2, loc="lower right")

    # (f) per-agent gain against channel richness
    ax = axes[1, 2]
    chans = ["answer_only", "rationale", "shared"]
    xs = np.arange(len(chans))
    for a in MAIN:
        g = [traj(L[c], a)[-1] - traj(L[c], a)[0] for c in chans]
        ax.plot(xs, g, marker="o", ms=6, lw=2, color=COLORS[a], label=a)
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["answer\nonly", "+ one\nsentence", "+ full\nreasoning"],
                       fontsize=8.5)
    ax.set_ylabel("accuracy change, round 0 $\\rightarrow$ 4")
    ax.set_title("(f) Gain vs. how much the channel carries", fontsize=10.5)
    ax.grid(alpha=.22)
    ax.set_axisbelow(True)
    ax.annotate("qwen72b — the strongest agent —\nonly stops losing once peers\nsend full reasoning",
                xy=(0, -0.047), xytext=(0.42, -0.15), fontsize=7.5, color="#6a4c93",
                arrowprops=dict(arrowstyle="->", color="#6a4c93", lw=1))
    ax.set_ylim(-0.22, 0.38)
    ax.set_xlabel("what the peer channel carries")  # not a round axis

    fig.suptitle("Per-agent trajectories across the channel ablation "
                 "(rounds 0-4, majority in black)", fontsize=13.5)
    fig.tight_layout()
    out = FIGURES_DIR / "ablation_trajectory.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}\n")

    print(f"{'agent':10}" + "".join(f"{c:>16}" for c in chans))
    for a in MAIN:
        print(f"{a:10}" + "".join(
            f"{traj(L[c],a)[-1]-traj(L[c],a)[0]:+16.3f}" for c in chans))


if __name__ == "__main__":
    main()
