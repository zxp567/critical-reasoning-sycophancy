"""Second task family: does the MMLU result hold on GSM8K arithmetic?

Reports the same core contrast on both protocols -- user always wrong (the
one-sided design) and user right half the time (the balanced control) -- so the
question is whether the gap between the two protocols reproduces off MMLU.

All condition differences use the same paired bootstrap over questions as the
main analysis: conditions share the question set, so resampling questions once
per iteration and scoring every condition on that resample is the right pairing.
"""

from __future__ import annotations

import pathlib

import numpy as np

from analyze import load_logs, signal_detection, wilson

N_BOOT = 5000
SEED = 12345
CONDS = ["baseline", "bss", "critical"]


def correct_vector(logs: list[dict]) -> np.ndarray:
    """Per-question 0/1 majority correctness, ordered by question id."""
    by_qid = {r["qid"]: r for r in logs}
    return np.array([by_qid[q]["majority_correct"] for q in sorted(by_qid)], dtype=float)


def paired_bootstrap(vecs: dict[str, np.ndarray], base: str) -> dict[str, tuple]:
    """CI on each condition's accuracy difference from `base`, questions resampled."""
    rng = np.random.default_rng(SEED)
    n = len(next(iter(vecs.values())))
    names = [c for c in vecs if c != base]
    draws = {c: np.empty(N_BOOT) for c in names}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        base_acc = vecs[base][idx].mean()
        for c in names:
            draws[c][b] = vecs[c][idx].mean() - base_acc
    return {c: (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
            for c, d in draws.items()}


def report(tag: str, log_dir: pathlib.Path, sdt_valid: bool) -> None:
    """`sdt_valid` is False on the one-sided set, where every item is a signal-absent
    trial, so the hit rate has no trials to be estimated from and d'/criterion are
    undefined rather than merely imprecise."""
    vecs, sdt = {}, {}
    for c in CONDS:
        p = log_dir / f"{c}.jsonl"
        if not p.exists():
            print(f"[{tag}] missing {p.name}, skipping")
            continue
        logs = load_logs(p)
        vecs[c] = correct_vector(logs)
        if sdt_valid:
            sdt[c] = signal_detection(logs)
    if "baseline" not in vecs:
        return

    cis = paired_bootstrap(vecs, "baseline")
    n = len(vecs["baseline"])

    head = f"{'condition':12}{'acc':>8}{'95% Wilson':>18}{'vs baseline':>26}"
    print(f"\n=== {tag}  (n = {n}) ===")
    print(head + (f"{'d-prime':>10}{'criterion':>11}" if sdt_valid else ""))
    for c in CONDS:
        if c not in vecs:
            continue
        acc = vecs[c].mean()
        lo, hi = wilson(int(vecs[c].sum()), n)
        delta = ""
        if c != "baseline":
            d = acc - vecs["baseline"].mean()
            cl, ch = cis[c]
            delta = f"{d:+.3f} [{cl:+.3f},{ch:+.3f}]"
        tail = (f"{sdt[c]['d_prime']:10.3f}{sdt[c]['criterion']:11.3f}"
                if sdt_valid else "")
        print(f"{c:12}{acc:8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{delta:>26}{tail}")


if __name__ == "__main__":
    import sys

    # The GSM8K runs use the `lite` roster (see config.ROSTERS), an independently
    # chosen set of agents, so this replication varies roster as well as dataset.
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_lite"
    root = pathlib.Path("../logs")
    report("GSM8K, user always wrong (one-sided)",
           root / f"gsm_test{suffix}", sdt_valid=False)
    report("GSM8K, user right half the time (balanced)",
           root / f"gsm_bal{suffix}", sdt_valid=True)
