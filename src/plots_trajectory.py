"""Round-by-round trajectories: how accuracy evolves during the discussion."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import load_logs, round_trajectory
from config import AGENTS, FIGURES_DIR, N_ROUNDS

ORDER = ["baseline", "warning_only", "bss", "critical", "critical_cot", "critical_bss"]
PRETTY = {"baseline": "baseline", "warning_only": "warning only", "bss": "BSS prior",
          "critical": "critical", "critical_cot": "critical + CoT",
          "critical_bss": "critical + BSS"}
COND_COLORS = {"baseline": "#8c8c8c", "warning_only": "#c2a25a", "bss": "#c0504d",
               "critical": "#3b7dd8", "critical_cot": "#4fa3a5", "critical_bss": "#7a5ea8"}
# weakest to strongest, so the legend reads as a capability gradient
MODEL_ORDER = ["gemma4b", "llama3b", "llama8b", "qwen7b", "llama70b", "qwen72b"]
MODEL_COLORS = {"gemma4b": "#d1495b", "llama3b": "#e08b3c", "llama8b": "#c9a227",
                "qwen7b": "#4c9f70", "llama70b": "#3b7dd8", "qwen72b": "#6a4c93"}

SPLITS = [("main", "Benchmark 1 — user always wrong"),
          ("balanced", "Benchmark 2 — user right half the time")]


def _traj(split: str, cond: str) -> dict:
    return round_trajectory(load_logs(pathlib.Path(f"../logs/{split}/{cond}.jsonl")))


def fig_majority(out: pathlib.Path) -> None:
    """Majority accuracy per round, every condition, both benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (split, title) in zip(axes, SPLITS):
        for c in ORDER:
            t = _traj(split, c)["majority"]
            ax.plot(range(len(t)), t, marker="o", ms=5, lw=2.2,
                    color=COND_COLORS[c], label=PRETTY[c],
                    ls="--" if c == "baseline" else "-",
                    zorder=3 if c == "baseline" else 2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("discussion round")
        ax.set_xticks(range(N_ROUNDS))
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0.55, 0.92)
    axes[0].set_ylabel("majority-vote accuracy")
    axes[0].legend(fontsize=8.5, ncol=2, loc="lower left")
    axes[0].annotate("round 0 = answered independently,\nbefore agents see each other",
                     xy=(0, _traj("main", "baseline")["majority"][0]),
                     xytext=(0.55, 0.86), fontsize=8.5, color="#444",
                     arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    fig.suptitle("Where the accuracy goes: round by round", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_model(split: str, out: pathlib.Path, title: str) -> None:
    """One panel per condition; one line per agent."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharey=True, sharex=True)
    for ax, c in zip(axes.flat, ORDER):
        tr = _traj(split, c)
        for m in MODEL_ORDER:
            if m not in tr:
                continue
            ax.plot(range(len(tr[m])), tr[m], marker="o", ms=3.5, lw=1.6,
                    color=MODEL_COLORS[m], label=m, alpha=.9)
        ax.plot(range(len(tr["majority"])), tr["majority"], lw=2.6, color="#111",
                ls="--", label="majority", zorder=4)
        ax.set_title(PRETTY[c], fontsize=10.5)
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        ax.set_xticks(range(N_ROUNDS))
    for ax in axes[1]:
        ax.set_xlabel("discussion round")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    fig_majority(FIGURES_DIR / "trajectory_majority.png")
    fig_per_model("main", FIGURES_DIR / "trajectory_per_model_b1.png",
                  "Benchmark 1 (user always wrong): every agent, every round")
    fig_per_model("balanced", FIGURES_DIR / "trajectory_per_model_b2.png",
                  "Benchmark 2 (user right half the time): every agent, every round")
    print("wrote trajectory_majority.png, trajectory_per_model_b1.png, "
          "trajectory_per_model_b2.png")
