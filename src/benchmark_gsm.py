"""The channel ablation of benchmarks 3--4, rerun on GSM8K arithmetic.

Four settings, isolating what reasoning is worth from what *sharing* it is worth:

  1. answering with no reasoning                  -- round 0 of `no_reasoning`
  2. answering with reasoning                     -- round 0 of `answer_only`/`shared`
  3. discussion, reasoning generated but private  -- final round of `answer_only`
  4. discussion, reasoning shared                 -- final round of `shared`

1 vs 2 isolates the value of reasoning at all; 3 vs 4 isolates the value of
transmitting it, holding generation fixed. Settings 2, 3 and 4 all generate exactly
the same thing, so any difference between them is about what peers can see.

There is no user here, as in benchmarks 3--4: no agree/disagree axis exists, so no
response threshold can be slid. The only way to score better is to be right more often.

The MCQ versions of these benchmarks do not transfer directly. GSM8K answers are
free-form integers rather than one of four supplied options, which changes two things:
the model must produce a number rather than select a letter, and the majority vote is
taken over an unbounded answer space, where agents can disagree six ways instead of
four. That makes ties both more likely and less meaningful, so the tie-break is
deliberately blind to correctness -- see `majority`.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

import data as data_mod
import prompts
from backend import Backend
from config import AGENTS, LOGS_DIR, MODELS, N_ROUNDS, TEMPERATURE

MAX_TOKENS_REASONING = 300  # 3 sentences + a verdict line, with headroom
MAX_TOKENS_DIRECT = 16      # a bare number, no room to reason first

SYSTEM_GSM = "You are a careful assistant. Answer the arithmetic word problem."

# -- prompts ------------------------------------------------------------------
# The no-reasoning arm must not be able to reason in the open, so it is given no
# scratchpad and a token budget that fits only the answer.
DIRECT_ROUND0 = """\
Question: {question}

Write only the final numeric answer, as a single number on a line by itself. \
Do not show any working."""

DIRECT_FOLLOWUP = (
    "\nWrite only the final numeric answer, as a single number on a line by itself. "
    "Do not show any working."
)

REASON_ROUND0 = """\
Question: {question}

First, in at most three sentences, work out the answer.

Then, on a final line by itself, write only the final numeric answer and nothing else."""

REASON_FOLLOWUP = (
    "\nFirst, in at most three sentences, work out the answer. Take the reasoning above "
    "into account where it is sound, and say so if you think it is mistaken.\n\n"
    "Then, on a final line by itself, write only the final numeric answer and nothing "
    "else."
)

PEER_HEADER = (
    "\n\n[Information about other agents]\n\n"
    "You are now going to find out how other LLM agents answered this question{and_why}.\n\n"
    "Your goal is to determine the correct answer.\n\n"
    "You can change your answer based on this info if you would like or keep it "
    "the same.\n\n"
    "{peer_block}\n"
)

CONDITIONS_GSM = {
    # reason: does the agent generate reasoning? channel: what peers get to see.
    "no_reasoning": dict(reason=False, channel="answer"),
    "answer_only":  dict(reason=True,  channel="answer"),
    "shared":       dict(reason=True,  channel="full"),
}

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> str | None:
    """The answer is the last number on the last line that contains one.

    Returned as a canonical string ('42') rather than a float, so that equality is
    exact and majority counting never merges two genuinely different answers.
    """
    if not text:
        return None
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        found = _NUM.findall(line.replace("$", ""))
        if not found:
            continue
        raw = found[-1].replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        return str(int(val)) if val == int(val) else str(val)
    return None


def parse(text: str) -> tuple[str | None, str]:
    """Return (answer, reasoning); reasoning is everything above the verdict line."""
    if not text:
        return None, ""
    lines = [l for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        got = parse_number(lines[i])
        if got is not None:
            return got, "\n".join(lines[:i]).strip()
    return None, text.strip()


def render_peer(agent: str, rec: dict, channel: str) -> str:
    ans = rec["answer"] or "no answer"
    if channel == "answer":
        return f"{agent}: {ans}"
    body = rec["reasoning"] or "(no reasoning given)"
    return f"--- {agent} answered {ans} ---\n{body}"


def peer_block(prev: dict, me: str, channel: str) -> str:
    out = [render_peer(o, prev[o], channel) for o in AGENTS if o != me]
    return ("\n" if channel == "answer" else "\n\n").join(out)


def majority(rd: dict) -> str | None:
    """Modal answer; ties broken by the numerically smallest value.

    The tie-break has to be independent of the correct answer, or it would leak
    ground truth into the score. Smallest-value is arbitrary with respect to
    correctness, which is exactly the property required, and it is deterministic
    so the reported number does not depend on dict ordering.
    """
    votes = Counter(v["answer"] for v in rd.values() if v["answer"])
    if not votes:
        return None
    best = max(votes.values())
    tied = [a for a, c in votes.items() if c == best]
    return min(tied, key=lambda a: (float(a), a))


async def _ask(be: Backend, agent: str, user: str, reason: bool):
    text = await be.chat(
        MODELS[agent], SYSTEM_GSM, user,
        max_tokens=MAX_TOKENS_REASONING if reason else MAX_TOKENS_DIRECT,
        temperature=TEMPERATURE,
    )
    return parse(text)


async def run_discussion(be: Backend, row, cfg: dict) -> dict:
    reason, channel = cfg["reason"], cfg["channel"]
    base = (REASON_ROUND0 if reason else DIRECT_ROUND0).format(question=row.question)
    followup = REASON_FOLLOWUP if reason else DIRECT_FOLLOWUP
    and_why = ", and the reasoning they gave" if channel == "full" else ""

    r0 = await asyncio.gather(*(_ask(be, a, base, reason) for a in AGENTS))
    state = {a: {"answer": ans, "reasoning": rz} for a, (ans, rz) in zip(AGENTS, r0)}
    rounds = [{a: dict(v) for a, v in state.items()}]
    n_unparsed = sum(v["answer"] is None for v in state.values())

    for _ in range(1, N_ROUNDS):
        prev = {a: dict(v) for a, v in state.items()}
        users = {
            a: base
            + PEER_HEADER.format(and_why=and_why,
                                 peer_block=peer_block(prev, a, channel))
            + followup
            for a in AGENTS
        }
        got = await asyncio.gather(*(_ask(be, a, users[a], reason) for a in AGENTS))
        for agent, (ans, rz) in zip(AGENTS, got):
            if ans is None:
                n_unparsed += 1
                state[agent] = dict(prev[agent])  # unparseable -> hold previous answer
            else:
                state[agent] = {"answer": ans, "reasoning": rz}
        rounds.append({a: dict(v) for a, v in state.items()})

    correct = str(row.correct_answer)
    maj = majority(rounds[-1])
    return {
        "qid": int(row.Index),
        "subject": row.subject,
        # `correct_letter` is the column name the shared analysis helpers expect;
        # on this dataset it holds the correct number.
        "correct_letter": correct,
        "rounds": rounds,
        "majority": maj,
        "majority_correct": maj == correct,
        "n_unparsed": n_unparsed,
    }


async def run_condition(be: Backend, df, condition: str, out_dir: Path) -> None:
    cfg = CONDITIONS_GSM[condition]
    logs = await tqdm_asyncio.gather(
        *(run_discussion(be, row, cfg) for row in df.itertuples()),
        desc=f"{condition:13}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{condition}.jsonl").open("w") as f:
        for rec in logs:
            f.write(json.dumps(rec) + "\n")

    n = len(logs)
    r0 = sum(majority(r["rounds"][0]) == r["correct_letter"] for r in logs) / n
    fin = sum(r["majority_correct"] for r in logs) / n
    unp = sum(r["n_unparsed"] for r in logs)
    print(f"  {condition:13} round0 = {r0:.3f}   final = {fin:.3f}   (unparsed: {unp})")


async def main(conditions: list[str], n_questions: int | None, out_dir: Path) -> None:
    df = data_mod.load(data_mod.DATA_DIR / "gsm_test.csv")
    if n_questions:
        df = df.head(n_questions)
    async with Backend() as be:
        for cond in conditions:
            await run_condition(be, df, cond, out_dir)
        print(f"\nbackend stats: {be.stats()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-c", "--conditions", nargs="+", default=list(CONDITIONS_GSM))
    p.add_argument("-n", "--n_questions", type=int, default=None)
    p.add_argument("-o", "--out_dir", default=str(LOGS_DIR / "gsm_channel"))
    a = p.parse_args()
    asyncio.run(main(a.conditions, a.n_questions, Path(a.out_dir)))
