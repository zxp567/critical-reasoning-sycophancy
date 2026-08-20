"""Per-agent trajectories for the GSM8K channel ablation (lite roster).

The MMLU counterpart of this figure (plots_channel.py) pins panel (a) to a fixed
0.68-0.88 window so the two rosters can be compared by eye. That cannot be done here:
the no-reasoning arm sits near 0.33 and the reasoning arms near 0.95, a 62-point
spread that no shared window with the MMLU figures could show. Panels are therefore
scaled to this experiment, and the axis limits are stated in the caption.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import load_logs
from benchmark_gsm import majority
from config import FIGURES_DIR

SCEN = [
    ("no reasoning", "#8c8c8c", "--", "no_reasoning"),
    ("reasoning, private", "#c0504d", "-", "answer_only"),
    ("reasoning, shared", "#2e8b57", "-", "shared"),
]
AGENTS_LITE = ["llama3b", "qwen7b", "nemo12b", "novamicro", "llama70b", "qwen72b"]
AGENT_COLORS = ["#d1495b", "#e08b3c", "#c9a227", "#4c9f70", "#3b7dd8", "#6a4c93"]


def maj_traj(logs):
    n = len(logs)
    return [sum(majority(r["rounds"][i]) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def agent_traj(logs, a):
    n = len(logs)
    return [sum(r["rounds"][i].get(a, {}).get("answer") == r["correct_letter"]
                for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def figure(log_dir: pathlib.Path, out: pathlib.Path):
    logs = [load_logs(log_dir / f"{c}.jsonl") for _, _, _, c in SCEN]
    colors = dict(zip(AGENTS_LITE, AGENT_COLORS))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    ax = axes[0]
    for L, (lab, col, ls, _) in zip(logs, SCEN):
        t = maj_traj(L)
        ax.plot(range(len(t)), t, marker="o", ms=5.5, lw=2.6, color=col, label=lab,
                ls=ls)
    ax.axhline(maj_traj(logs[0])[0], color="#333", lw=1, ls=":", alpha=.75)
    ax.set_xticks(range(5))
    ax.set_xlabel("discussion round")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title("(a) Majority vote", fontsize=11.5)
    ax.set_ylim(0.20, 1.02)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)

    ylo = min(min(agent_traj(L, a)) for L in logs for a in AGENTS_LITE) - .05
    yhi = max(max(agent_traj(L, a)) for L in logs for a in AGENTS_LITE) + .05
    for ax, L, (lab, _, _, _), tag in zip(axes[1:], logs, SCEN, "bcd"):
        for a in AGENTS_LITE:
            t = agent_traj(L, a)
            ax.plot(range(len(t)), t, marker="o", ms=4, lw=1.8, color=colors[a],
                    label=a)
        m = maj_traj(L)
        ax.plot(range(len(m)), m, lw=2.8, color="#111", ls="--", label="majority",
                zorder=5)
        ax.set_xticks(range(5))
        ax.set_xlabel("discussion round")
        ax.set_title(f"({tag}) per agent — {lab}", fontsize=11.5)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
    axes[1].set_ylabel("accuracy")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("GSM8K, independent roster — what the peer channel carries",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    rows = [(a, [(agent_traj(L, a)[-1], agent_traj(L, a)[-1] - agent_traj(L, a)[0])
                 for L in logs]) for a in AGENTS_LITE]
    maj_row = [(maj_traj(L)[-1], maj_traj(L)[-1] - maj_traj(L)[0]) for L in logs]
    r0_row = [maj_traj(L)[0] for L in logs]
    return rows, maj_row, r0_row


def emit(rows, maj_row, r0_row):
    names = [s[0] for s in SCEN]
    print(f"\n{'agent':12}" + "".join(f"{n:>24}" for n in names))
    for a, cells in rows:
        print(f"{a:12}" + "".join(f"{f:18.3f} ({d:+.3f})" for f, d in cells))
    print(f"{'majority':12}" + "".join(f"{f:18.3f} ({d:+.3f})" for f, d in maj_row))
    print(f"{'(round 0)':12}" + "".join(f"{v:24.3f}" for v in r0_row))

    tex = ["\\begin{tabular}{lrrr}", "\\toprule",
           "Agent & No reasoning & Reasoning, private & Reasoning, shared \\\\",
           "\\midrule"]
    for a, cells in rows:
        tex.append(f"\\texttt{{{a}}} & " +
                   " & ".join(f"{f:.3f} \\scriptsize$({d:+.3f})$" for f, d in cells) +
                   " \\\\")
    tex.append("\\midrule")
    tex.append("\\textbf{Majority} & " +
               " & ".join(f"$\\mathbf{{{f:.3f}}}$ \\scriptsize$({d:+.3f})$"
                          for f, d in maj_row) + " \\\\")
    tex.append("Round 0 (majority) & " +
               " & ".join(f"{v:.3f}" for v in r0_row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    pathlib.Path("../paper/table_gsm_channel.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / "channel_gsm.png"
    rows, maj_row, r0_row = figure(pathlib.Path("../logs/gsm_channel"), out)
    emit(rows, maj_row, r0_row)
    print(f"\nwrote {out} and ../paper/table_gsm_channel.tex")
