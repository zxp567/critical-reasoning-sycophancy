"""Experiment configuration: agent roster, conditions, and protocol constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
SCORES_DIR = ROOT / "scores"
FIGURES_DIR = ROOT / "figures"
CACHE_PATH = ROOT / "data" / "llm_cache.jsonl"

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
# The paper uses Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-{3B,7B,14B,32B}. OpenRouter
# serves only three of those six, so we keep the four we can (marked exact) and
# substitute within the same model families to preserve the property the paper
# actually relies on: a wide, monotone capability gradient across the roster.
#
# llama-3.2-1b was screened out: it answered 'incorrect' on 150/150 calibration
# items, a constant responder whose 0.0 sycophancy score is an artifact rather
# than a behaviour. gemma-3-4b-it replaces it at the weak end.
ROSTERS = {
    # The main roster: spans a wide capability gradient, as the paper's does.
    "default": {
        "gemma4b":  "google/gemma-3-4b-it",               # substitute (weak end)
        "llama3b":  "meta-llama/llama-3.2-3b-instruct",   # exact (paper)
        "llama8b":  "meta-llama/llama-3.1-8b-instruct",   # exact (paper)
        "qwen7b":   "qwen/qwen-2.5-7b-instruct",          # exact (paper)
        "llama70b": "meta-llama/llama-3.3-70b-instruct",  # substitute (strong end)
        "qwen72b":  "qwen/qwen-2.5-72b-instruct",         # substitute for qwen32b
    },
    # Strong-only roster, to test whether the interventions behave differently
    # when every agent is individually competent. Screened for degeneracy and
    # parse reliability; no model here scores below 0.62 on the MCQ probe,
    # against a floor of 0.50 in the default roster.
    "strong": {
        "gemma12b":     "google/gemma-3-12b-it",
        "qwen3-8b":     "qwen/qwen3-8b",
        "qwen3-30b":    "qwen/qwen3-30b-a3b-instruct-2507",
        "llama70b":     "meta-llama/llama-3.3-70b-instruct",
        "qwen3next80b": "qwen/qwen3-next-80b-a3b-instruct",
        "qwen72b":      "qwen/qwen-2.5-72b-instruct",
    },
    # A third roster, chosen independently of the two above and used for the GSM8K
    # extension, so that run varies the agents as well as the dataset. Four vendors,
    # a 3B-to-72B capability gradient, and every model sub-second on the stance task.
    #
    # Screened on 24 balanced GSM8K items, as llama-3.2-1b was. Rejected there:
    # gpt-oss-20b (22/24 unparsed - it spends the answer budget reasoning),
    # ministral-8b and mistral-7b-instruct (404), command-r7b-12-2024 (23/24
    # 'correct', a constant responder in the opposite direction to llama-3.2-1b),
    # phi-4 (accurate at 0.67 but ~9s per call, an order of magnitude slower).
    "lite": {
        "llama3b":    "meta-llama/llama-3.2-3b-instruct",
        "qwen7b":     "qwen/qwen-2.5-7b-instruct",
        "nemo12b":    "mistralai/mistral-nemo",
        "novamicro":  "amazon/nova-micro-v1",
        "llama70b":   "meta-llama/llama-3.3-70b-instruct",
        "qwen72b":    "qwen/qwen-2.5-72b-instruct",
    },
}

# Select with the ROSTER environment variable, e.g. ROSTER=strong python3 ...
ROSTER = os.environ.get("ROSTER", "default")
if ROSTER not in ROSTERS:
    raise ValueError(f"unknown ROSTER {ROSTER!r}; choose from {list(ROSTERS)}")

MODELS = ROSTERS[ROSTER]
AGENTS = list(MODELS.keys())

# Score files are per-roster: BSS is a property of a specific line-up, not of a
# model in isolation, so it must never be shared across rosters.
SUFFIX = "" if ROSTER == "default" else f"_{ROSTER}"


def score_path(name: str) -> Path:
    return SCORES_DIR / f"{name}{SUFFIX}.json"

# -----------------------------------------------------------------------------
# Protocol (matches the paper: n = 6 agents, m = 5 rounds)
# -----------------------------------------------------------------------------
N_ROUNDS = 5           # round 0 (independent) + rounds 1..4 (peer-informed)
MAX_TOKENS_SHORT = 8   # one-word stance
# Raised from 200: at 200, ~3% of CoT responses were truncated mid-reasoning
# before emitting the verdict line. Truncation falls back to holding the previous
# stance, which on the paper protocol is an accuracy *advantage* (round 0 is the
# most accurate round), so the cap was silently favouring the CoT arm.
MAX_TOKENS_COT = 512   # scratchpad + one-word stance
TEMPERATURE = 0.0

# The five MMLU subjects the paper selected.
SUBJECTS = [
    "elementary_mathematics",
    "professional_law",
    "machine_learning",
    "business_ethics",
    "high_school_biology",
]

# We run the sycophant-with-knowledge metric: it conditions on the agent
# actually knowing the answer, so it isolates sycophancy from ignorance.
METRIC = "sycophant_with_knowledge"

# -----------------------------------------------------------------------------
# Conditions
# -----------------------------------------------------------------------------
# A 2x2 over {no prior, BSS prior} x {plain, critical}, plus warning_only as the
# midpoint between "no information" and "per-agent sycophancy labels".
CONDITIONS = {
    # --- reproduction of the paper ---
    "baseline":     dict(prior=None,  critical=False, cot=False),
    "bss":          dict(prior="bss", critical=False, cot=False),
    "warning_only": dict(prior="warn", critical=False, cot=False),
    # --- the hypothesis under test ---
    "critical":     dict(prior=None,  critical=True,  cot=False),
    "critical_cot": dict(prior=None,  critical=True,  cot=True),
    "critical_bss": dict(prior="bss", critical=True,  cot=False),
}

SEED = 123

# -----------------------------------------------------------------------------
# Stance balance
# -----------------------------------------------------------------------------
# In the paper the simulated user's asserted option is ALWAYS wrong, so the
# correct stance is always 'incorrect'. That makes a constant 'incorrect'
# responder score 100% accuracy and 0.0 sycophancy without reading the question,
# and it means any prompt that merely biases agents against agreement is
# rewarded for free. The balanced set asserts the CORRECT option on half the
# items, breaking that degeneracy, and is used as a control.
BALANCED_FRACTION_USER_CORRECT = 0.5

