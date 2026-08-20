"""Build the calibration and test question sets from MMLU.

Each item pairs a question with an option that the simulated user asserts.

Two test sets are produced:
  * test.csv           - the user's option is ALWAYS incorrect (paper protocol).
  * test_balanced.csv  - the user is correct on half the items (control).

The balanced set exists because under the paper's protocol the correct stance is
always 'incorrect', so a degenerate constant responder scores perfectly and any
disagreement-inducing prompt is rewarded regardless of whether it reasons.

Calibration and test are disjoint, matching Section 3.1.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from config import BALANCED_FRACTION_USER_CORRECT, DATA_DIR, SEED, SUBJECTS

LETTERS = ["A", "B", "C", "D"]


# The first BASE_CAL_PER_SUBJECT calibration items and the test block are drawn
# exactly as in the original 30/30 build, so enlarging the calibration set never
# moves them and their cached model responses stay valid.
BASE_CAL_PER_SUBJECT = 30


def _load_split(subject: str, split: str) -> list[dict]:
    from datasets import load_dataset

    try:
        ds = load_dataset("cais/mmlu", subject, split=split)
    except (ValueError, KeyError):
        return []
    out = []
    for row in ds:
        choices = list(row["choices"])
        if len(choices) != 4:
            continue
        out.append(
            {
                "subject": subject,
                "question": row["question"].strip(),
                "choices": choices,
                "answer_idx": int(row["answer"]),
            }
        )
    return out


def _load_subject(subject: str) -> list[dict]:
    return _load_split(subject, "test")


def _load_overflow(subject: str) -> list[dict]:
    """Validation + dev items, used only to extend the calibration set."""
    return _load_split(subject, "validation") + _load_split(subject, "dev")


def _make_row(it: dict, user_is_correct: bool, rng: random.Random) -> dict:
    """Attach the user's asserted option and the resulting ground-truth stance."""
    correct_idx = it["answer_idx"]
    if user_is_correct:
        user_idx = correct_idx
    else:
        user_idx = rng.choice([i for i in range(4) if i != correct_idx])
    return {
        "subject": it["subject"],
        "question": it["question"],
        "choices": json.dumps(it["choices"], ensure_ascii=False),
        "answer_idx": correct_idx,
        "correct_letter": LETTERS[correct_idx],
        "correct_answer": it["choices"][correct_idx],
        "user_letter": LETTERS[user_idx],
        "user_answer": it["choices"][user_idx],
        "user_is_correct": user_is_correct,
        # The stance an ideal agent should take on the user's assertion.
        "correct_stance": "correct" if user_is_correct else "incorrect",
    }


def build(n_cal_per_subject: int = 30, n_test_per_subject: int = 30) -> dict[str, Path]:
    # Two independent RNG streams: `rng` reproduces the cal/test draw sequence
    # exactly regardless of whether the balanced set is built, so its cached
    # model responses stay valid. `rng_bal` is used only for the control set.
    rng = random.Random(SEED)
    rng_extra = random.Random(SEED + 2)
    rng_bal = random.Random(SEED + 1)
    cal_rows, test_rows, bal_rows = [], [], []
    per_subject_test: dict[str, list[dict]] = {}
    per_subject_spare: dict[str, list[dict]] = {}

    base_cal = min(n_cal_per_subject, BASE_CAL_PER_SUBJECT)

    # -- Pass 1: identical draw sequence to the original 30/30 build ----------
    for subject in SUBJECTS:
        items = _load_subject(subject)
        rng.shuffle(items)
        need = base_cal + n_test_per_subject
        if len(items) < need:
            raise ValueError(
                f"{subject}: only {len(items)} questions available, need {need}"
            )
        test_items = items[base_cal:need]
        per_subject_test[subject] = test_items
        # Everything past the test block, plus validation/dev, can extend
        # calibration without disturbing anything already drawn.
        per_subject_spare[subject] = items[need:]

        # Calibration follows the paper exactly: the user is always wrong.
        for it in items[:base_cal]:
            cal_rows.append(_make_row(it, False, rng))

        # Paper-protocol test set.
        for it in test_items:
            test_rows.append(_make_row(it, False, rng))

    # -- Pass 2: additional calibration items on their own RNG stream ---------
    n_extra = n_cal_per_subject - base_cal
    if n_extra > 0:
        for subject in SUBJECTS:
            pool = list(per_subject_spare[subject]) + _load_overflow(subject)
            rng_extra.shuffle(pool)
            if len(pool) < n_extra:
                raise ValueError(
                    f"{subject}: calibration pool exhausted - {len(pool)} spare "
                    f"items but {n_extra} requested. Max n_cal_per_subject here "
                    f"is {base_cal + len(pool)}."
                )
            for it in pool[:n_extra]:
                cal_rows.append(_make_row(it, False, rng_extra))

    # Balanced control, built in a second pass on its own RNG stream. Same
    # questions in the same order as test.csv, so knowledge probes are shared.
    for subject in SUBJECTS:
        test_items = per_subject_test[subject]
        n_correct = round(len(test_items) * BALANCED_FRACTION_USER_CORRECT)
        flags = [True] * n_correct + [False] * (len(test_items) - n_correct)
        rng_bal.shuffle(flags)
        for it, flag in zip(test_items, flags):
            bal_rows.append(_make_row(it, flag, rng_bal))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (
        ("cal", cal_rows),
        ("test", test_rows),
        ("test_balanced", bal_rows),
    ):
        p = DATA_DIR / f"{name}.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths[name] = p
    return paths


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "choices" in df.columns:
        df["choices"] = df["choices"].apply(json.loads)
    else:
        # free-form datasets (e.g. GSM8K) have no option list
        df["choices"] = [[] for _ in range(len(df))]
    return df


def choices_block(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


if __name__ == "__main__":
    import sys

    n_cal = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for name, p in build(n_cal, n_test).items():
        df = pd.read_csv(p)
        frac = df["user_is_correct"].mean()
        print(f"wrote {p.name:18} {len(df):4d} rows  (user correct on {frac:.0%})")
