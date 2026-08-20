"""Channel ablation, one figure and one table per roster.

Three scenarios, in increasing order of what travels between agents:

  1. no reasoning        -- agents answer directly; peers see the answer
  2. reasoning, private  -- agents reason first; peers still see only the answer
  3. reasoning, shared   -- agents reason first; peers see the reasoning too

Scenario 1 vs 2 isolates the value of reasoning at all. Scenario 2 vs 3 isolates
the value of transmitting it, holding generation fixed.
"""

from __future__ import annotations

import pathlib
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_logs, wilson
from config import FIGURES_DIR, ROSTERS

SCEN = [
    ("no reasoning", "#8c8c8c", "--"),
    ("reasoning, private", "#c0504d", "-"),
    ("reasoning, shared", "#2e8b57", "-"),
]
AGENT_COLORS = ["#d1495b", "#e08b3c", "#c9a227", "#4c9f70", "#3b7dd8", "#6a4c93"]

ROSTER_ORDER = {
    "default": ["llama3b", "llama8b", "gemma4b", "qwen7b", "llama70b", "qwen72b"],
    "strong": ["gemma12b", "qwen3-8b", "qwen3-30b", "llama70b", "qwen3next80b",
               "qwen72b"],
}


def maj(rd):
    v = Counter(x for x in rd.values() if x)
    if not v:
        return None
    b = max(v.values())
    return sorted(l for l, c in v.items() if c == b)[0]


def maj_traj(logs):
    n = len(logs)
    return [sum(maj(r["rounds"][i]) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def agent_traj(logs, a):
    n = len(logs)
    return [sum(r["rounds"][i].get(a) == r["correct_letter"] for r in logs) / n
            for i in range(len(logs[0]["rounds"]))]


def load(roster):
    """Return the three scenarios for a roster, in order."""
    if roster == "default":
        paths = ["../logs/b3/baseline.jsonl", "../logs/b4/answer_only.jsonl",
                 "../logs/b4/shared.jsonl"]
    else:
        paths = ["../logs/b3_strong/baseline.jsonl",
                 "../logs/b4_strong/answer_only.jsonl",
                 "../logs/b4_strong/shared.jsonl"]
    return [load_logs(pathlib.Path(p)) for p in paths]


def figure(roster: str, title: str, out: pathlib.Path) -> list:
    logs = load(roster)
    agents = ROSTER_ORDER[roster]
    colors = dict(zip(agents, AGENT_COLORS))

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    # (a) majority vote, all three scenarios
    ax = axes[0]
    for L, (lab, col, ls) in zip(logs, SCEN):
        t = maj_traj(L)
        ax.plot(range(len(t)), t, marker="o", ms=5.5, lw=2.6, color=col, label=lab,
                ls=ls)
    ax.axhline(maj_traj(logs[0])[0], color="#333", lw=1, ls=":", alpha=.75)
    ax.set_xticks(range(5))
    ax.set_xlabel("discussion round")
    ax.set_ylabel("majority-vote accuracy")
    ax.set_title("(a) Majority vote", fontsize=11.5)
    # Fixed across rosters: auto-scaling makes the strong roster's much smaller
    # differences look as large as the main roster's.
    ax.set_ylim(0.68, 0.88)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)

    # (b)-(d) per-agent trajectories, one panel per scenario, shared y-axis
    ylo = min(min(agent_traj(L, a)) for L in logs for a in agents) - .04
    yhi = max(max(agent_traj(L, a)) for L in logs for a in agents) + .04
    for ax, L, (lab, _, _), tag in zip(axes[1:], logs, SCEN, "bcd"):
        for a in agents:
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

    fig.suptitle(title, fontsize=13.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Report final accuracy as the primary number with the change in parentheses.
    # Round 0 is NOT identical across scenarios (the no-reasoning arm answers with a
    # different prompt), so the changes alone are not comparable between columns.
    rows = []
    for a in agents:
        rows.append((a, [(agent_traj(L, a)[-1], agent_traj(L, a)[-1] - agent_traj(L, a)[0])
                         for L in logs]))
    maj_row = [(maj_traj(L)[-1], maj_traj(L)[-1] - maj_traj(L)[0]) for L in logs]
    r0_row = [maj_traj(L)[0] for L in logs]
    return rows, maj_row, r0_row


def emit_table(roster, rows, maj_row, r0_row):
    names = [s[0] for s in SCEN]
    print(f"\n{'agent':14}" + "".join(f"{n:>24}" for n in names))
    for a, cells in rows:
        print(f"{a:14}" + "".join(f"{f:18.3f} ({d:+.3f})" for f, d in cells))
    print(f"{'majority':14}" + "".join(f"{f:18.3f} ({d:+.3f})" for f, d in maj_row))
    print(f"{'(round 0)':14}" + "".join(f"{v:24.3f}" for v in r0_row))

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
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    pathlib.Path(f"../paper/table_{roster}.tex").write_text("\n".join(tex))


if __name__ == "__main__":
    import os

    FIGURES_DIR.mkdir(exist_ok=True)
    for roster, title, fname in [
        ("default", "Main roster — what the peer channel carries",
         "channel_main.png"),
        ("strong", "Strong roster — what the peer channel carries",
         "channel_strong.png"),
    ]:
        try:
            rows, maj_row, r0_row = figure(roster, title, FIGURES_DIR / fname)
        except FileNotFoundError as e:
            print(f"[skip {roster}] missing {e.filename}")
            continue
        print(f"\n=== {title} ===")
        emit_table(roster, rows, maj_row, r0_row)
        print(f"wrote {FIGURES_DIR/fname} and ../paper/table_{roster}.tex")
