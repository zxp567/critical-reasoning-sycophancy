"""The four channel settings on GSM8K, with paired bootstrap contrasts.

  1. answering alone, no reasoning     round 0 of `no_reasoning`
  2. answering alone, with reasoning   round 0 of `answer_only` (identical to `shared`)
  3. discussion, reasoning private     final round of `answer_only`
  4. discussion, reasoning shared      final round of `shared`

Settings 2-4 generate exactly the same thing; only what peers can see differs, so
3 vs 4 isolates transmission from generation. 1 vs 2 isolates reasoning itself.

A fifth row, discussion with no reasoning at all, is reported for continuity with the
MMLU tables, where it is the answer-only discussion arm.
"""

from __future__ import annotations

import pathlib

import numpy as np

from analyze import load_logs, wilson
from benchmark_gsm import majority

N_BOOT = 5000
SEED = 12345


def vec(logs: list[dict], rnd: int) -> np.ndarray:
    logs = sorted(logs, key=lambda r: r["qid"])
    return np.array([majority(r["rounds"][rnd]) == r["correct_letter"] for r in logs],
                    dtype=float)


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Difference a - b with a bootstrap CI, resampling questions (not arms)."""
    rng = np.random.default_rng(SEED)
    n = len(a)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d[i] = a[idx].mean() - b[idx].mean()
    return a.mean() - b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main(log_dir: pathlib.Path) -> None:
    L = {c: load_logs(log_dir / f"{c}.jsonl")
         for c in ("no_reasoning", "answer_only", "shared")}

    arms = {
        "answering alone, no reasoning":   vec(L["no_reasoning"], 0),
        "answering alone, w/ reasoning":   vec(L["answer_only"], 0),
        "discussion, answers only":        vec(L["no_reasoning"], -1),
        "discussion, reasoning private":   vec(L["answer_only"], -1),
        "discussion, reasoning shared":    vec(L["shared"], -1),
    }
    n = len(next(iter(arms.values())))

    print(f"GSM8K channel ablation, lite roster (n = {n})\n")
    print(f"{'setting':34}{'reasoning':>10}{'shared':>8}{'accuracy':>10}{'95% Wilson':>18}")
    meta = {
        "answering alone, no reasoning": ("no", "--"),
        "answering alone, w/ reasoning": ("yes", "--"),
        "discussion, answers only":      ("no", "no"),
        "discussion, reasoning private": ("yes", "no"),
        "discussion, reasoning shared":  ("yes", "yes"),
    }
    for name, v in arms.items():
        lo, hi = wilson(int(v.sum()), n)
        r, s = meta[name]
        print(f"{name:34}{r:>10}{s:>8}{v.mean():10.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")

    print("\ncontrasts (paired bootstrap over questions, 5000 resamples)")
    pairs = [
        ("reasoning itself (alone, w/ vs no)",
         "answering alone, w/ reasoning", "answering alone, no reasoning"),
        ("sharing it (shared vs private)",
         "discussion, reasoning shared", "discussion, reasoning private"),
        ("shared vs answer-only discussion",
         "discussion, reasoning shared", "discussion, answers only"),
        ("shared vs answering alone (w/ reasoning)",
         "discussion, reasoning shared", "answering alone, w/ reasoning"),
        ("private vs answering alone (w/ reasoning)",
         "discussion, reasoning private", "answering alone, w/ reasoning"),
    ]
    for label, a, b in pairs:
        d, lo, hi = paired_ci(arms[a], arms[b])
        star = " *" if lo > 0 or hi < 0 else ""
        print(f"  {label:42}{d:+.3f} [{lo:+.3f},{hi:+.3f}]{star}")
    print("\n  * interval excludes zero")

    print("\nper-round majority accuracy")
    for cond, lab in [("no_reasoning", "no reasoning"),
                      ("answer_only", "reasoning, private"),
                      ("shared", "reasoning, shared")]:
        traj = [vec(L[cond], i).mean() for i in range(len(L[cond][0]["rounds"]))]
        print(f"  {lab:22}" + "  ".join(f"{t:.3f}" for t in traj))

    print("\nunparsed answers (agent-decisions)")
    for cond in ("no_reasoning", "answer_only", "shared"):
        tot = sum(r["n_unparsed"] for r in L[cond])
        cells = len(L[cond]) * 6 * len(L[cond][0]["rounds"])
        print(f"  {cond:16}{tot:5d} / {cells:5d}  ({tot/cells:.1%})")


if __name__ == "__main__":
    import sys

    d = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("../logs/gsm_channel")
    main(d)
