"""Figures: accuracy by condition, sycophancy, round trajectories, influence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import wilson
from config import AGENTS, FIGURES_DIR, LOGS_DIR

ORDER = ["baseline", "warning_only", "bss", "critical", "critical_cot", "critical_bss"]
COLORS = {
    "baseline": "#8c8c8c",
    "warning_only": "#c2a25a",
    "bss": "#c0504d",
    "critical": "#3b7dd8",
    "critical_cot": "#4fa3a5",
    "critical_bss": "#7a5ea8",
}
PRETTY = {
    "baseline": "baseline",
    "warning_only": "warning only",
    "bss": "BSS prior\n(paper)",
    "critical": "critical",
    "critical_cot": "critical\n+ CoT",
    "critical_bss": "critical\n+ BSS",
}


def _conds(res: dict) -> list[str]:
    return [c for c in ORDER if c in res]


def fig_accuracy(res: dict, out: Path) -> None:
    conds = _conds(res)
    cols = AGENTS + ["majority"]
    x = np.arange(len(cols))
    w = 0.8 / len(conds)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, c in enumerate(conds):
        acc = res[c]["accuracy"]
        vals = [acc[m][0] / acc[m][1] for m in cols]
        errs = np.array(
            [
                [
                    acc[m][0] / acc[m][1] - wilson(*acc[m])[0],
                    wilson(*acc[m])[1] - acc[m][0] / acc[m][1],
                ]
                for m in cols
            ]
        ).T
        ax.bar(
            x + i * w - 0.4 + w / 2, vals, w, label=PRETTY[c].replace("\n", " "),
            color=COLORS[c], yerr=errs, capsize=2,
            error_kw=dict(lw=0.8, alpha=0.6),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=15)
    ax.set_ylabel("final-round accuracy")
    ax.set_title("Final accuracy by agent and condition (Wilson 95% CI)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_majority(res: dict, out: Path) -> None:
    conds = _conds(res)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    vals, los, his = [], [], []
    for c in conds:
        k, n = res[c]["accuracy"]["majority"]
        lo, hi = wilson(k, n)
        vals.append(k / n)
        los.append(k / n - lo)
        his.append(hi - k / n)
    bars = ax.bar(
        range(len(conds)), vals, color=[COLORS[c] for c in conds],
        yerr=[los, his], capsize=4, error_kw=dict(lw=1.0, alpha=0.7),
    )
    base = res["baseline"]["accuracy"]["majority"]
    ax.axhline(base[0] / base[1], ls="--", lw=1, color="#444", alpha=0.7)
    for b, v, c in zip(bars, vals, conds):
        p = res[c].get("_p_vs_baseline")
        star = "*" if (p is not None and p < 0.05) else ""
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.035, f"{v:.3f}{star}",
            ha="center", fontsize=9,
        )
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([PRETTY[c] for c in conds], fontsize=9)
    ax.set_ylabel("majority-vote accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Discussion outcome accuracy (* = p < 0.05 vs baseline)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_sycophancy(res: dict, out: Path) -> None:
    conds = _conds(res)
    x = np.arange(len(AGENTS))
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, c in enumerate(conds):
        syc = res[c]["sycophancy"]
        vals = [(syc[a][0] / syc[a][1] if syc[a][1] else np.nan) for a in AGENTS]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w,
               label=PRETTY[c].replace("\n", " "), color=COLORS[c])
    ax.set_xticks(x)
    ax.set_xticklabels(AGENTS, rotation=15)
    ax.set_ylabel("post-discussion sycophancy")
    ax.set_title("Endorsement of the user's wrong stance at the final round (lower is better)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_trajectory(res: dict, out: Path) -> None:
    conds = _conds(res)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for c in conds:
        t = res[c]["trajectory"]["majority"]
        ax.plot(range(len(t)), t, marker="o", label=PRETTY[c].replace("\n", " "),
                color=COLORS[c], lw=2)
    ax.set_xlabel("discussion round")
    ax.set_ylabel("majority accuracy")
    ax.set_title("Round-by-round majority accuracy")
    ax.set_xticks(range(len(res[conds[0]]["trajectory"]["majority"])))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_influence(res: dict, out: Path) -> None:
    conds = _conds(res)
    n = len(conds)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), squeeze=False)
    for ax, c in zip(axes[0], conds):
        m = pd.DataFrame(res[c]["influence"]).reindex(index=AGENTS, columns=AGENTS)
        im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=40)
        ax.set_xticks(range(len(AGENTS)))
        ax.set_xticklabels(AGENTS, rotation=90, fontsize=7)
        ax.set_yticks(range(len(AGENTS)))
        ax.set_yticklabels(AGENTS if c == conds[0] else [], fontsize=7)
        ax.set_title(PRETTY[c].replace("\n", " "), fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="% of target's flips")
    fig.suptitle("Pairwise influence (source row -> target column)", fontsize=11)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(log_dir: Path) -> None:
    res = json.loads((log_dir / "results.json").read_text())
    acc = pd.read_csv(log_dir / "accuracy.csv").set_index("condition")
    for c in res:
        if c in acc.index:
            res[c]["_p_vs_baseline"] = acc.loc[c, "p_vs_baseline"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_majority(res, FIGURES_DIR / "majority_accuracy.png")
    fig_accuracy(res, FIGURES_DIR / "accuracy_by_agent.png")
    fig_sycophancy(res, FIGURES_DIR / "sycophancy.png")
    fig_trajectory(res, FIGURES_DIR / "trajectory.png")
    fig_influence(res, FIGURES_DIR / "influence.png")
    print(f"wrote 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--log_dir", default=str(LOGS_DIR / "main"))
    main(Path(ap.parse_args().log_dir))
