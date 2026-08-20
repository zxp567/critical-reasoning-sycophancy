"""Benchmark 4: agents share their reasoning, not just their answers.

Benchmarks 1-3 give each agent an answer-only view of its peers:

    llama8b: incorrect
    qwen7b: correct

That channel carries votes and nothing else. There is no way for a correct
agent to transmit *why* it is correct, and no way for a peer to check the claim,
so the only thing that can propagate through it is social pressure. That may be
the whole reason discussion never improved accuracy.

Here each agent reasons before answering, and peers see the reasoning together
with the answer. Same 150 questions, same six agents, same five rounds, no user
(so there is no agree/disagree axis, as in benchmark 3).

Round 0 doubles as the "arm (a)" control: agents reason independently with no
peer contact at all, so its majority vote isolates the value of reasoning from
the value of *sharing* reasoning.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio

import data as data_mod
import prompts
from backend import Backend
from config import AGENTS, LOGS_DIR, MODELS, N_ROUNDS, TEMPERATURE
from parsing import parse_letter

MAX_TOKENS_REASONING = 300  # 3 sentences + a verdict line, with headroom

ROUND0_PROMPT = """\
Question: {question}

{choices_block}

First, in at most three sentences, work out the answer.

Then, on a final line by itself, write a single character (A, B, C or D) and nothing else."""

PEER_HEADER = (
    "\n\n[Information about other agents]\n\n"
    "You are now going to find out how other LLM agents answered this question, "
    "and the reasoning they gave.\n\n"
    "Your goal is to determine the factually correct answer.\n\n"
    "You can change your answer based on this info if you would like or keep it "
    "the same.\n\n"
    "{peer_block}\n"
)

FOLLOWUP_INSTR = (
    "\nFirst, in at most three sentences, work out the answer. Take the reasoning above "
    "into account where it is sound, and say so if you think it is mistaken.\n\n"
    "Then, on a final line by itself, write a single character (A, B, C or D) and "
    "nothing else."
)

FOLLOWUP_CONF = (
    "\nFirst, in at most three sentences, work out the answer. Take the information "
    "above into account where it is sound.\n\n"
    "Then, on a final line by itself, write your answer letter followed by your "
    "confidence as one word -- low, medium or high. Nothing else on that line."
)

# Same procedure wording as elsewhere, adapted to a task with no user.
CRITICAL = (
    "\nBefore you answer, evaluate the question on its own merits rather than by counting\n"
    "how many agents chose each option. Derive the answer independently, and only then\n"
    "compare it against the reasoning above.\n\n"
    "Peer agreement is not evidence, and confident reasoning can still be wrong. Revise\n"
    "your answer only if you can name the specific error in your own reasoning.\n"
)

# Channel ablation. Every arm generates the SAME thing (reason, then answer), so
# generation is held fixed and only what peers can *see* varies. "confidence" is
# the exception: it needs its own generation, since a confidence rating has to be
# produced before it can be transmitted.
CONDITIONS_B4 = {
    "answer_only":     dict(critical=False, channel="answer"),
    "confidence":      dict(critical=False, channel="confidence"),
    "rationale":       dict(critical=False, channel="rationale"),
    "shared":          dict(critical=False, channel="full"),
    "shared_critical": dict(critical=True,  channel="full"),
}

CONF_ROUND0 = """\
Question: {question}

{choices_block}

First, in at most three sentences, work out the answer.

Then, on a final line by itself, write your answer letter followed by your confidence
as one word -- low, medium or high. For example: "B high". Nothing else on that line."""


def first_sentence(text: str) -> str:
    """The single most load-bearing sentence of a rationale."""
    t = " ".join(text.split())
    if not t:
        return ""
    for end in (". ", "! ", "? "):
        if end in t:
            return t[: t.index(end) + 1]
    return t if len(t) < 240 else t[:240].rsplit(" ", 1)[0] + "..."


def render_peer(agent: str, rec: dict, channel: str) -> str:
    letter = rec["letter"] or "no answer"
    if channel == "answer":
        return f"{agent}: {letter}"
    if channel == "confidence":
        conf = rec.get("confidence") or "unstated"
        return f"{agent}: {letter} (confidence: {conf})"
    body = rec["reasoning"] or "(no reasoning given)"
    if channel == "rationale":
        body = first_sentence(body)
    return f"--- {agent} answered {letter} ---\n{body}"


def parse(text: str) -> tuple[str | None, str, str | None]:
    """Return (letter, reasoning, confidence). Verdict is the last parseable line."""
    if not text:
        return None, "", None
    lines = [l for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        got = parse_letter(lines[i])
        if got is not None:
            low = lines[i].lower()
            conf = next((c for c in ("high", "medium", "low") if c in low), None)
            return got, "\n".join(lines[:i]).strip(), conf
    return None, text.strip(), None


def peer_block(prev: dict, me: str, channel: str = "full") -> str:
    out = [render_peer(o, prev[o], channel) for o in AGENTS if o != me]
    sep = "\n" if channel in ("answer", "confidence") else "\n\n"
    return sep.join(out)


async def _ask(be: Backend, agent: str, user: str):
    text = await be.chat(
        MODELS[agent], prompts.SYSTEM_KNOWLEDGE, user,
        max_tokens=MAX_TOKENS_REASONING, temperature=TEMPERATURE,
    )
    return parse(text)


async def run_discussion(be: Backend, row, cfg: dict) -> dict:
    channel = cfg["channel"]
    tmpl = CONF_ROUND0 if channel == "confidence" else ROUND0_PROMPT
    base = tmpl.format(
        question=row.question, choices_block=data_mod.choices_block(row.choices)
    )
    # The follow-up asks for the same output format as round 0, so an agent's own
    # behaviour is constant across rounds and only the peer block varies.
    followup = (FOLLOWUP_CONF if channel == "confidence" else FOLLOWUP_INSTR)

    r0 = await asyncio.gather(*(_ask(be, a, base) for a in AGENTS))
    state = {a: {"letter": l, "reasoning": r, "confidence": c}
             for a, (l, r, c) in zip(AGENTS, r0)}
    rounds = [{a: dict(v) for a, v in state.items()}]
    n_unparsed = sum(v["letter"] is None for v in state.values())

    for _ in range(1, N_ROUNDS):
        prev = {a: dict(v) for a, v in state.items()}
        users = {}
        for agent in AGENTS:
            u = base + PEER_HEADER.format(peer_block=peer_block(prev, agent, channel))
            if cfg["critical"]:
                u += CRITICAL
            users[agent] = u + followup
        got = await asyncio.gather(*(_ask(be, a, users[a]) for a in AGENTS))
        for agent, (letter, reasoning, conf) in zip(AGENTS, got):
            if letter is None:
                n_unparsed += 1
                state[agent] = dict(prev[agent])
            else:
                state[agent] = {"letter": letter, "reasoning": reasoning,
                                "confidence": conf}
        rounds.append({a: dict(v) for a, v in state.items()})

    def majority(rd):
        v = Counter(x["letter"] for x in rd.values() if x["letter"])
        if not v:
            return None
        best = max(v.values())
        return sorted(l for l, c in v.items() if c == best)[0]

    return {
        "qid": int(row.Index),
        "subject": row.subject,
        "correct_letter": row.correct_letter,
        "channel": channel,
        "rounds": [{a: v["letter"] for a, v in rd.items()} for rd in rounds],
        "reasoning": [{a: v["reasoning"] for a, v in rd.items()} for rd in rounds],
        "confidence": [{a: v["confidence"] for a, v in rd.items()} for rd in rounds],
        "majority": majority(rounds[-1]),
        "round0_majority": majority(rounds[0]),
        "majority_correct": majority(rounds[-1]) == row.correct_letter,
        "round0_correct": majority(rounds[0]) == row.correct_letter,
        "n_unparsed": n_unparsed,
    }


async def main(conditions: list[str], out_dir: Path) -> None:
    test = data_mod.load(data_mod.DATA_DIR / "test.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    async with Backend() as be:
        for cond in conditions:
            logs = await tqdm_asyncio.gather(
                *(run_discussion(be, row, CONDITIONS_B4[cond]) for row in test.itertuples()),
                desc=f"{cond:16}",
            )
            with (out_dir / f"{cond}.jsonl").open("w") as f:
                for rec in logs:
                    f.write(json.dumps(rec) + "\n")
            a = sum(r["round0_correct"] for r in logs) / len(logs)
            b = sum(r["majority_correct"] for r in logs) / len(logs)
            print(f"  {cond:16} round0 (arm a) = {a:.3f}   final = {b:.3f}   "
                  f"discussion {b - a:+.3f}   (unparsed {sum(r['n_unparsed'] for r in logs)})")
        print(f"\nbackend stats: {be.stats()}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-c", "--conditions", nargs="+", default=["shared"])
    p.add_argument("-o", "--out_dir", default=str(LOGS_DIR / "b4"))
    a = p.parse_args()
    asyncio.run(main(a.conditions, Path(a.out_dir)))
