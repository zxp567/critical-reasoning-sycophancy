"""How reliable is the BSS ranking itself?

The method presents agents with a confident four-tier ordering of their peers.
That ordering is estimated from a finite calibration sample, so before asking
whether it helps we should ask whether it is measuring anything stable.

Two checks:
  1. Pairwise two-proportion z-tests between every pair of agents' BSS rates.
  2. A nonparametric bootstrap over the calibration items, asking how often the
     estimated ranking reproduces itself.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter

from analyze import two_proportion_z
from config import AGENTS, SCORES_DIR, score_path


def main(n_boot: int = 5000, seed: int = 0) -> dict:
    bss = json.loads(score_path("bss").read_text())
    detail, raw = bss["detail"], bss["raw"]

    print("Pairwise BSS differences (two-proportion z-test)")
    n_sig = 0
    pairs = list(itertools.combinations(AGENTS, 2))
    for a1, a2 in pairs:
        k1, n1 = detail[a1]["n_sycophantic"], detail[a1]["n_in_K"]
        k2, n2 = detail[a2]["n_sycophantic"], detail[a2]["n_in_K"]
        p = two_proportion_z(k1, n1, k2, n2)
        n_sig += p < 0.05
        print(f"  {a1:9} {k1/n1:.3f}  vs  {a2:9} {k2/n2:.3f}   p={p:.3f}")
    print(f"\n  {n_sig}/{len(pairs)} pairs differ significantly at p<0.05\n")

    rng = random.Random(seed)
    truth = sorted(AGENTS, key=lambda a: raw[a])
    top, bottom, exact = Counter(), Counter(), 0
    for _ in range(n_boot):
        est = {}
        for a in AGENTS:
            n, k = detail[a]["n_in_K"], detail[a]["n_sycophantic"]
            est[a] = sum(rng.random() < k / n for _ in range(n)) / n
        order = sorted(AGENTS, key=lambda a: est[a])
        top[order[0]] += 1
        bottom[order[-1]] += 1
        exact += order == truth

    print(f"Bootstrap over calibration items ({n_boot} resamples)")
    print("  P(ranked least sycophantic):",
          {k: round(v / n_boot, 3) for k, v in top.most_common()})
    print("  P(ranked most sycophantic): ",
          {k: round(v / n_boot, 3) for k, v in bottom.most_common()})
    print(f"  P(full ranking reproduced exactly) = {exact / n_boot:.3f}")

    out = {
        "n_significant_pairs": n_sig,
        "n_pairs": len(pairs),
        "p_exact_ranking": exact / n_boot,
        "p_least": {k: v / n_boot for k, v in top.items()},
        "p_most": {k: v / n_boot for k, v in bottom.items()},
    }
    (score_path("bss_reliability")).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
