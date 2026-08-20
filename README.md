# Does reducing sycophancy mean improving accuracy?

📄 **Write-up: [zxp567.github.io/sycophancy-vs-accuracy](https://zxp567.github.io/sycophancy-vs-accuracy/)**

## Background

**Sycophancy** is a language model's tendency to go along with a user's stated
position even when it conflicts with what the model itself would otherwise
conclude. Tell a model the answer is B when it privately believes A, and it will
often agree anyway. It is largely a side effect of training: RLHF rewards
responses people rate highly, and people rate agreement highly.

That matters most wherever a model *checks* something rather than produces it. An
assistant that validates a mistaken contract reading or a wrong dosage
calculation fails precisely when the user was already wrong and needed catching.

**Multi-agent systems raise the stakes.** The argument for having several models
discuss a question is that independent errors cancel and the majority lands
closer to the truth — which assumes independence. If every agent is separately
inclined to defer, deference compounds instead of cancelling: one agent's
capitulation becomes peer evidence for the next.

[Kasprova et al. (arXiv:2604.02668)](https://arxiv.org/abs/2604.02668) put numbers
on this, showing that sycophancy *propagates* through multi-agent discussion, and
proposed a fix worth **10.5 accuracy points**: precompute how sycophantic each
model is (a **Base Sycophancy Score**, BSS), then tell every agent how sycophantic
its peers are so it can discount the flatterers.

## The question

BSS is expensive in a particular way: it must be estimated per roster, needs a
calibration set of hundreds of labelled items, and goes stale whenever a model is
swapped. So:

> **Can a critical-reasoning prompt — free, calibration-free, telling agents to
> validate independently rather than telling them whom to distrust — reduce
> sycophancy and improve accuracy the way an estimated BSS prior does?**

**On sycophancy: yes.** The prompt cuts sycophancy about as well as the calibrated
prior (0.299 → 0.155 vs 0.299 → 0.124) at none of the cost, and posts +10.7
accuracy points on benchmark 1 against BSS's +14.0.

**On accuracy: neither does.** In benchmark 1 the simulated user is *always*
wrong, so rejecting the user and being correct are the same event. Under
benchmark 2, where the user is right half the time, every accuracy gain
disappears — for the prompt and for the prior alike — while sycophancy still
falls. What both interventions reliably change is agents' willingness to
disagree, not their ability to tell a correct claim from an incorrect one.

**Then what does?** Not a prompt, it turns out. In every benchmark above, agents
see only each other's *answers*. Let them exchange *reasoning* instead and the
same six models on the same questions gain 8 points — the one configuration here
that clearly beats them answering independently.

See [Results](#results) for the numbers.

## Design

A 2×2 over {no prior, BSS prior} × {plain, critical instruction}, plus two
controls:

| condition | peer prior | critical instruction | purpose |
|---|---|---|---|
| `baseline` | — | — | reference point |
| `bss` | ranked sycophancy labels | — | the calibrated prior |
| `warning_only` | generic caution, no per-agent labels | — | isolates "be skeptical" from "know who to distrust" |
| `critical` | — | yes | **the hypothesis** |
| `critical_cot` | — | yes, with a verification scratchpad | does reasoning room matter? |
| `critical_bss` | ranked sycophancy labels | yes | complementary or redundant? |

`warning_only` is a generic caution with no per-agent attribution. It separates
"be skeptical of your peers" from "know which peers to discount", which is what
tells us whether the prior's *content* matters or merely its presence.

The critical instruction is deliberately **length-matched** to the BSS warning
block (~100 words) so no arm wins on prompt volume, and it contains no claim
about any specific peer — only a procedure.

### Benchmark 2: the balanced control

In benchmark 1 the simulated user's asserted option is **always
wrong**, so the correct stance is always `'incorrect'`. This creates a scoring
loophole: an agent that ignores the question and always answers `'incorrect'`
scores **100% accuracy and 0.0 sycophancy**. (Llama-3.2-1B does exactly this,
150/150 — it was dropped from the roster for this reason.)

That loophole threatens the hypothesis directly: a prompt telling agents not to
go along with peers could raise accuracy purely by inducing contrarianism, with
no reasoning involved. So every condition is also run on **benchmark 2**, where
the user asserts the *correct* answer on half the items. There, reflexive
disagreement costs as much as it gains. `disagreement_rate` in the analysis
tracks this explicitly against the true base rate.

Both benchmarks use the same protocol otherwise. In particular the simulated
user is a **fixed stimulus, not a participant**: they assert one answer at round
0, that assertion is re-injected verbatim into every later round, and they never
observe or respond to the discussion. Only the peer-response block changes
between rounds, and agents never see their own previous answer. Holding the
user's stance constant is what makes the measurement interpretable — a user who
drifted toward the majority would confound persuasion with a moving target.

## Agents

Three of the six models used in the prior work are unavailable on OpenRouter, so substitutes
preserve the two properties the method relies on: a wide capability gradient and
multiple model families.

| key | model | status |
|---|---|---|
| `gemma4b` | `google/gemma-3-4b-it` | substitute (weak end) |
| `llama3b` | `meta-llama/llama-3.2-3b-instruct` | as in prior work |
| `llama8b` | `meta-llama/llama-3.1-8b-instruct` | as in prior work |
| `qwen7b` | `qwen/qwen-2.5-7b-instruct` | as in prior work |
| `llama70b` | `meta-llama/llama-3.3-70b-instruct` | substitute (strong end) |
| `qwen72b` | `qwen/qwen-2.5-72b-instruct` | substitute for `qwen32b` |

Benchmark 1 follows the prior work's protocol: n = 6 agents, m = 5 rounds (round 0 independent,
then 4 peer-informed rounds), 5 MMLU subjects, disjoint calibration/test splits.

## Results

**Headline: every intervention wins on benchmark 1. None survives benchmark 2.**

![headline result](figures/headline.png)

| condition | benchmark 1 | Δ [95% CI] | benchmark 2 | Δ [95% CI] |
|---|---|---|---|---|
| `baseline` | 0.627 | — | 0.720 | — |
| `warning_only` | **0.733** | +0.107 [+0.060, +0.160] | 0.700 | −0.020 [−0.080, +0.040] |
| `bss` *(calibrated prior)* | **0.767** | +0.140 [+0.087, +0.200] | 0.707 | −0.013 [−0.087, +0.060] |
| `critical` *(hypothesis)* | **0.733** | +0.107 [+0.060, +0.153] | 0.740 | +0.020 [−0.040, +0.080] |
| `critical_cot` | **0.840** | +0.213 [+0.147, +0.280] | 0.740 | +0.020 [−0.053, +0.093] |
| `critical_bss` | **0.787** | +0.160 [+0.100, +0.220] | 0.740 | +0.020 [−0.053, +0.093] |

n = 150 questions per cell, majority-vote accuracy at the final round. Differences use
a paired bootstrap over questions (5,000 resamples), since every condition is scored on
the same items — an unpaired test would discard that pairing and overstate uncertainty.
Every benchmark-1 interval excludes zero; every benchmark-2 interval contains it. The
previously reported effect reproduces and slightly exceeds its +10.5 points.

### Why: the interventions shift a threshold, they don't improve judgement

In benchmark 1 the user is *always* wrong, so "rejects the user" and
"is correct" are the same event — the two are numerically identical in our logs
for every condition. The protocol therefore cannot distinguish better reasoning
from plain contrarianism. Benchmark 2 can. Treating "reject the user" as
the positive response:

Every judgement falls into one of four cells (the right-hand column exists only
in benchmark 2):

| | user actually **wrong** | user actually **right** |
|---|---|---|
| agent rejects user | **hit** | **false alarm** |
| agent agrees | miss | correct rejection |

**d′** is how far apart the "user is wrong" and "user is right" cases are for the
agent, in standard deviations — genuine discriminating skill, and unchangeable by
moving a threshold. **Criterion** is where the threshold sits — how readily the
agent rejects at all, regardless of evidence. `d′ = z(hit) − z(fa)` and
`criterion = −½[z(hit) + z(fa)]`.

![what d-prime and criterion mean](figures/sdt_explainer.png)

In raw counts out of 900 decisions, BSS caught 87 more wrong users (295→382
hits) but wrongly contradicted 119 more right ones (115→234 false alarms) — a
net loss of 32. Benchmark 1 can only observe the first number.

![signal detection](figures/signal_detection.png)

| condition | hit rate | false alarm | d′ (skill) | criterion (bias) |
|---|---|---|---|---|
| `baseline` | 0.656 | 0.256 | **1.057** | +0.128 |
| `warning_only` | 0.778 | 0.429 | 0.944 | −0.293 |
| `bss` | 0.849 | 0.520 | 0.982 | −0.541 |
| `critical` | 0.822 | 0.447 | **1.058** | −0.395 |
| `critical_cot` | 0.796 | 0.460 | 0.926 | −0.363 |
| `critical_bss` | 0.853 | 0.542 | 0.945 | −0.578 |

**The criterion moves; discrimination doesn't.** A paired bootstrap over
questions (4,000 resamples) places the criterion shifts far from zero — BSS
−0.674 [−0.797, −0.557], `critical` −0.525 [−0.632, −0.432] — while the d′
shifts are point-estimated near zero with intervals straddling it
(−0.078 [−0.323, +0.157] and −0.001 [−0.211, +0.202]). We therefore claim a
large measured change in *bias* alongside no detectable change in *skill* or
accuracy, not a proof that d′ is unchanged; our data could hide a d′ movement
of ±0.2, but not the criterion shift, which is 3–9x larger.

Without the signal-detection model at all: BSS raises correct rejections by 19
points and *incorrect* rejections by 26. Agents reject more of everything.

### Benchmark 3: no user at all

Both benchmarks above use a binary answer space, so a threshold always exists for
an intervention to slide. Benchmark 3 removes the user entirely: same 150
questions, same six agents, same five rounds, but agents answer the MCQ directly
(A/B/C/D). With no user there is no agree/disagree axis, so the only way to score
better is to pick the right letter more often.

![benchmark 3](figures/benchmark3.png)

| condition | round 0 | final | what discussion did | vs baseline |
|---|---|---|---|---|
| `baseline` | 0.740 | 0.707 | −0.033 [−0.080, +0.013] | — |
| `bss` | 0.740 | 0.667 | **−0.073** [−0.120, −0.033] | −0.040 (p=0.46) |
| `critical` | 0.740 | 0.693 | −0.047 [−0.093, +0.000] | −0.013 (p=0.80) |
| `critical_cot` | 0.740 | 0.733 | −0.007 [−0.073, +0.060] | +0.027 (p=0.61) |

**No intervention beats baseline**, and the critical prompt lands 1.3 points
below it. The +10.7 it earned on benchmark 1 does not survive removing the
agree/disagree axis — which is exactly the claim it was built to test.

Answer-only discussion doesn't help here either. The best score on the benchmark is **round
zero**: six models answering alone with a majority vote, no conversation. The BSS
prior actively hurts, which is unsurprising — it reweights peers using a
credibility signal derived from user-directed sycophancy, in a task with no user.

Across benchmarks 1–3, what discussion does to accuracy:

| | change from discussion |
|---|---|
| Benchmark 1 (user always wrong) | **−0.173** [−0.233, −0.113] |
| Benchmark 2 (user right half the time) | +0.067 [−0.013, +0.147] |
| Benchmark 3 (no user) | −0.033 [−0.080, +0.013] |

This does not describe a system that pools information. It describes one that
moves bias around: amplifying it where it is aligned against the task, correcting
it where it runs the other way, and doing nothing where there is none.

### Benchmark 4: sharing reasoning, not just answers

In benchmarks 1-3 an agent's entire view of its peers is a list of votes
(`llama8b: incorrect`). No agent can transmit *why* it is correct and no peer can
check the claim, so the only signal able to propagate is social. Benchmark 4
widens the channel: agents reason before answering, and peers see the reasoning
alongside the answer. Same 150 questions, same six agents, five rounds, no user.

![benchmark 4](figures/benchmark4.png)

| configuration | reasoning generated | reasoning shared | accuracy |
|---|---|---|---|
| answering alone | ✗ | — | 0.740 |
| answering alone, with reasoning | ✓ | — | 0.740 |
| discussion, answers only | ✗ | ✗ | 0.707 |
| discussion, reasoning kept private | ✓ | ✗ | 0.767 |
| **discussion, reasoning shared** | ✓ | ✓ | **0.820** |

**It isn't just the reasoning, it's the sharing.** Reasoning with *no discussion at
all* nets out to zero (+0.000 [−0.080, +0.080]) — a wash rather than an absence of
effect: it flips 36 of 150 items, 18 each way, helping some agents and hurting others
(`llama3b` −0.120). Inside a discussion it does better, +0.060 [−0.007, +0.133] over
never reasoning, though that interval crosses zero. Sharing that same reasoning is worth
a further **+0.053** [+0.013, +0.100] over the identical private-reasoning setup, and
**+0.113** [+0.053, +0.173] over answer-only discussion; both of those exclude zero.

This is the one configuration in the study that clearly beats agents answering
independently, by +0.080 [+0.020, +0.147]. Private reasoning also edges ahead, but
by +0.027 [−0.047, +0.100] — an interval containing zero. An answer-only protocol looks like collaboration and behaves like
a poll: votes carry a popularity signal but no truth signal, so pooling them
amplifies whatever bias the group already had. Reasoning can be checked against
the question rather than counted.

Strictly the poll analogy is generous. A poll aggregates *independent* responses, and
independent votes do carry information. Iterated discussion produces something worse:
once agents observe and update on peer votes, those votes become correlated, and a vote
no longer distinguishes independent evidence from an echo of an earlier vote.

Caveat: benchmark 4 is not compute-matched to 1-3. The private-reasoning arm is
the control for that — same generation cost, 5.3 points worse.

### Strong-roster replication

Rerun with six models each scoring ≥ 0.62 on the knowledge probe (Qwen3-8B,
Qwen3-30B, Qwen3-Next-80B, Llama-3.3-70B, Qwen2.5-72B, Gemma-3-12B).

| | weak roster | strong roster |
|---|---|---|
| baseline d′ | 1.07 | **1.65** |
| baseline accuracy (balanced) | 0.720 | **0.807** |
| BSS, benchmark 1 | +14.0 (p=0.008) | **+7.3 (p=0.131, n.s.)** |
| `critical`, benchmark 1 | +10.7 (p=0.048) | **+3.3 (p=0.505, n.s.)** |
| BSS, benchmark-2 accuracy shift | −0.036 [−0.090, +0.020] | +0.004 [−0.031, +0.042] |
| BSS, criterion shift | −0.674 | **−0.265** |

Stronger agents are better judges, and the interventions do *less* for them —
the headline gains halve and lose significance even on the original protocol.
The apparent benefit is largest exactly where agents are worst, which is what a
bias correction looks like.

The BSS ranking is also **well estimated on this roster**: Gemma-3-12B is a
genuine outlier (0.350 vs 0.124–0.196), 7 of 15 pairs separate at p < 0.05, and
the full ranking reproduces 19.9% of the time (vs 2.3%). So the priors are not
inherently broken — they are well determined when a roster contains real
sycophancy variation and poorly determined when it does not, with nothing in the
output distinguishing the two cases.

**The sycophancy metric improves anyway.** Post-discussion sycophancy falls from
0.299 to 0.124–0.188 even in benchmark 2, where accuracy does not move. That
metric counts false agreement but not false disagreement, so raising general
skepticism drives it down regardless of whether judgement improved. Both of the
headline metrics are one-sided in the same direction.

### Second task family: GSM8K arithmetic

Benchmarks 1-2 rerun on 150 GSM8K word problems, where the answer is a free-form
integer derived in several steps rather than one of four supplied options. The
user's wrong assertion is constructed by perturbing the correct answer the way an
arithmetic slip would (doubled/dropped step, off-by-small-integer, percentage-sized
miss), landing a median of 20% away.

This run also uses a different set of agents, chosen independently of the main
roster, to test robustness to the roster at the same time: the `lite` roster spans
four vendors and 3B-72B, with MMLU probe accuracy 0.500-0.816, and was screened for
degeneracy exactly as the main roster was. Dataset and roster therefore vary
together — a weaker test of the dataset specifically, a stronger test of the finding
overall.

| condition | user always wrong | user right half the time |
|---|---|---|
| baseline | 0.433 | 0.593 |
| `bss` | **0.573** (+0.140 [+0.080, +0.200]) | 0.573 (−0.020 [−0.080, +0.047]) |
| `critical` | **0.573** (+0.140 [+0.080, +0.207]) | 0.573 (−0.020 [−0.087, +0.053]) |

Same shape as MMLU: both interventions gain +14.0 points one-sided (intervals
exclude zero), lose 2.0 on the balanced control (intervals contain zero).

Signal detection, 900 agent-decisions on the balanced set:

| condition | hit | false alarm | d′ | Δd′ | criterion | Δcriterion |
|---|---|---|---|---|---|---|
| baseline | 0.500 | 0.342 | 0.406 | — | +0.203 | — |
| `bss` | 0.687 | 0.560 | 0.335 | −0.071 [−0.250, +0.113] | −0.319 | **−0.522** [−0.619, −0.434] |
| `critical` | 0.660 | 0.547 | 0.295 | −0.111 [−0.297, +0.065] | −0.265 | **−0.468** [−0.566, −0.378] |

The hit rate rises 0.500 → 0.687 and the false-alarm rate rises 0.342 → 0.560 with
it: agents reject more of the wrong assertions *and* more of the right ones. That is
a threshold sliding. Criterion shifts land in the same −0.53 to −0.67 band measured
on MMLU; neither d′ shift is distinguishable from zero.

Reproduce with `ROSTER=lite python3 gsm_report.py`.

### Channel ablation on GSM8K

The four settings of benchmark 4, rerun on GSM8K with the `lite` roster, user removed.
Answers are free-form integers, so the majority vote runs over an unbounded answer
space; ties break to the numerically smallest value, which is independent of
correctness by construction.

| setting | reasoning | shared | accuracy | 95% CI |
|---|---|---|---|---|
| answering alone | ✗ | — | 0.327 | [0.257, 0.405] |
| answering alone, with reasoning | ✓ | — | 0.947 | [0.898, 0.973] |
| discussion, answers only | ✗ | ✗ | 0.340 | [0.269, 0.419] |
| discussion, reasoning private | ✓ | ✗ | 0.953 | [0.907, 0.977] |
| discussion, reasoning shared | ✓ | ✓ | 0.947 | [0.898, 0.973] |

| contrast | GSM8K | MMLU |
|---|---|---|
| reasoning itself | **+0.620** [+0.540, +0.700] | +0.000 |
| sharing it (shared vs private) | −0.007 [−0.040, +0.027] | **+0.053** [+0.013, +0.100] |

Reasoning matters enormously on arithmetic and not at all on MMLU — expected, since a
free-form integer from a multi-step problem needs a scratchpad and a 4-option pick does
not. These two numbers measure the same manipulation on tasks with different demands,
not a failed replication.

Sharing reasoning does not replicate, and the reason is headroom rather than consensus:

| | GSM8K | MMLU |
|---|---|---|
| round-0 disagreement | 79.3% | 83.3% |
| round-0 majority already correct | 142/150 (94.7%) | 111/150 (74.0%) |
| questions discussion could fix | **8** | **39** |

The largest possible effect on GSM8K is 8/150 = +0.053 — exactly the MMLU effect size —
so replication would have required fixing every remaining error. Restricting to
disagreeing questions gives −0.008 (n=119).

**But the majority vote hides the effect.** Per-agent, shared reasoning is decisive:

![GSM8K channel ablation](figures/channel_gsm.png)

| agent | private | shared | difference |
|---|---|---|---|
| `llama3b` | 0.447 | **0.820** | **+0.373** [+0.273, +0.473] |
| `nemo12b` | 0.800 | **0.947** | **+0.147** [+0.087, +0.207] |
| `novamicro` | 0.813 | **0.940** | **+0.127** [+0.067, +0.193] |
| `qwen7b` | 0.913 | **0.953** | +0.040 [+0.000, +0.080] |
| `qwen72b` | 0.940 | **0.953** | +0.013 [−0.020, +0.053] |
| `llama70b` | **0.947** | 0.933 | −0.013 [−0.053, +0.020] |
| **majority** | **0.953** | 0.947 | −0.007 [−0.040, +0.027] |

Three of six intervals exclude zero. The gain is inversely correlated with an agent's
solo accuracy (r = −0.99), and the roster's spread collapses from 0.500 to 0.133 wide:
shared reasoning levels weak agents up rather than making anyone exceptional. Since the
two strong agents already carried the majority at 94%, that changes almost no votes.

The two measures are not in conflict — the majority vote simply has no room left. With
142 of 150 items already correct at round 0, even a 37-point lift to the weakest agent
reaches a vote that was mostly settled without it. So the group-verdict benefit cannot
be tested here, while the per-agent benefit replicates strongly: across all three
rosters, shared reasoning gives **17 of 18 agents** their best result of the three
settings.

Figure and per-agent table: `ROSTER=lite python3 plots_channel_gsm.py`.

Reproduce with `ROSTER=lite python3 benchmark_gsm.py` then
`ROSTER=lite python3 gsm_channel_report.py`.

### Supporting findings

**Answer-only discussion destroys accuracy when a wrong user is present.**
Baseline majority accuracy by round:
`0.800 → 0.720 → 0.653 → 0.627 → 0.627`. A 17.3-point collapse (bootstrap 95%
CI [0.113, 0.233], P(drop > 0) = 1.000); all six agents end worse than they
started. The channel is what does this, not deliberation as such: over the same
five rounds, agents exchanging *reasoning* instead of votes gain 8 points
(0.740 → 0.820). What collapses here is a poll, not a discussion. Steeper than in the prior work because our roster has no resistant core: it reports
three agents finishing ≥ 0.86, ours has none above 0.733. On the two shared models
the numbers are close (`llama8b` 0.613 vs 0.612).

**The BSS ranking is not firmly determined.** At the enlarged 430-question
calibration, sycophancy rates span just 0.138–0.179 and **0 of 15 pairwise
differences reach p < 0.05**; the full six-way ranking reproduces itself 2.3% of
the time under bootstrap. Enlarging from 150 to 430 questions *reordered* the
ranking and *shrank* the raw spread from 0.074 to 0.041 — consistent with the
original spread being largely noise. A power analysis says separating the extreme
pair needs |K| ≈ 389 (250 questions gives ~218); separating adjacent
models would need |K| ≈ 1.7 million. The roster supports two or three
distinguishable tiers at most, while the method assigns four.

The ranking is also sensitive to a definitional choice: under strict
knowledge-conditioning (K = model picked the *correct* answer, rather than the
the standard "did not pick the user's option"), `gemma4b` and `llama8b` swap two
positions.

**Low MCQ accuracy is not incompetence at this task.** `llama3b` scores 0.493 on
4-way MCQ but 0.827 on the binary stance judgement, because rejecting a wrong
answer is easier than producing the right one. The gap closes to 0.000 for the
strongest model. Round-0 majority accuracy is 0.800 — the vote is reliable before
discussion begins.

### What's released, and what it costs to check

The full apparatus rather than a summary of it: all code and prompts, every cached model
response keyed by `(model, messages, sampling params)`, and per-round logs for every
discussion — covering every condition, benchmark, roster and dataset reported.

Because responses are cached by content, **every table and figure regenerates offline
with no API calls.** A reader can check that a reported number follows from the raw
transcripts, rather than re-running the models and hoping the numbers land in the same
place. Re-running from scratch is inexpensive — the roster is deliberately built from
small open models so that reproducing this doesn't require a budget.

## Layout

```
core
  src/config.py         agent rosters, conditions, protocol constants
  src/prompts.py        stance/discussion templates + critical-reasoning variants
  src/backend.py        async OpenRouter client, disk cache, retries
  src/data.py           MMLU splits; benchmark 1 and benchmark 2 question sets
  src/data_gsm.py       GSM8K splits; plausible-slip wrong answers
  src/parsing.py        robust parsing of the constrained model outputs

experiments
  src/calibrate.py      knowledge/stance probes, BSS computation
  src/discussion.py     6-agent x 5-round engine (benchmarks 1 and 2, both datasets)
  src/benchmark3.py     no user; agents answer A/B/C/D, exchange answers
  src/benchmark4.py     no user; agents exchange reasoning as well as answers
  src/benchmark_gsm.py  the channel ablation on GSM8K (free-form numeric answers)

analysis
  src/analyze.py        accuracy, sycophancy, flips, influence, signal detection,
                        Wilson CIs
  src/reliability.py    is the BSS ranking distinguishable from noise?
  src/gsm_report.py          GSM8K benchmarks 1-2, accuracy + signal detection
  src/gsm_channel_report.py  the four channel settings on GSM8K

figures and output
  src/plots.py                per-agent accuracy, sycophancy, influence
  src/plots_headline.py       benchmark 1 vs 2, signal detection
  src/plots_sdt_explainer.py  what d-prime and criterion mean
  src/plots_syc_vs_acc.py     sycophancy falls, accuracy stays flat
  src/plots_trajectory.py     round-by-round, benchmarks 1 and 2
  src/plots_b3.py             benchmark 3 and the three-benchmark synthesis
  src/plots_b4.py             the reasoning/sharing ladder
  src/plots_b4_traj.py        per-agent trajectories, benchmarks 3 vs 4
  src/plots_ablation.py       what the channel has to carry
  src/plots_channel.py        per-roster channel figures and tables (MMLU)
  src/plots_channel_gsm.py    per-agent channel trajectories (GSM8K)
  src/build_page.py           renders BLOG.md into docs/ for GitHub Pages
```

## Running

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env

./run_remaining.sh                 # benchmarks 1 and 2, all six conditions
ROSTER=strong ./run_strong.sh      # strong-roster replication
cd src && python3 benchmark3.py    # benchmark 3 — no user, answers only
cd src && python3 benchmark4.py    # benchmark 4 — reasoning shared
```

Figures and the published page:

```bash
cd src
for f in plots*.py; do python3 "$f"; done
python3 build_page.py              # renders BLOG.md into docs/
```

Every model call is cached to `data/llm_cache.jsonl`, keyed by
(model, messages, sampling params), so interrupted runs resume for free and
re-analysis never re-spends tokens. Round 0 is shared across conditions wherever
the prompt is identical, so it is computed once.

## Caveats

**How strong is the null?** The central negative claim — that these interventions
move response bias rather than judgement — is an absence of evidence. No individual
benchmark 2 accuracy difference is significant at n = 150, and the d′ intervals
(±0.2–0.3) could hide a modest real improvement in discrimination. So it doesn't
rest on that arm alone: it is carried jointly by the signal-detection decomposition
(~900 agent-decisions per condition), the strong-roster replication, benchmark
3, which tests the same claim without needing the signal-detection model at all,
and the GSM8K rerun on a different task with a different roster. Four lines
agreeing; none decisive by itself. Single seed throughout.

**How much do two datasets buy?** The GSM8K rerun varies roster as well as dataset,
so it shows the result survives both changes together rather than isolating the
dataset; four of six models overlap between rosters. Both task families are English and
academic.

**The mechanism was wrong the first time.** The strong-roster non-replication was
originally explained as the benefit scaling with how much disagreement there is to
resolve. GSM8K tests that and it fails: round-0 disagreement is comparable across
datasets (79.3% vs 83.3%). What tracks the effect is how often the majority is already
correct (74% vs 94.7%). The revised account has survived one test rather than none.

**How solid is the shared-reasoning result?** Benchmark 4 generates roughly ten times more
tokens per question than benchmarks 1–3, so it is not compute-matched to them. The
private-reasoning arm is the control for that — identical generation cost, 5.3
points worse — which is why the gain is attributed to what peers can *see* rather
than to token budget. It does not answer the next question a practitioner would
ask: does six agents sharing reasoning beat one strong model given an equally long
scratchpad? It was rerun on GSM8K with a split outcome: the per-agent benefit
replicated, but with only 8 questions of headroom the majority-vote benefit could not
be tested. And whether a
reasoning channel repairs sycophancy propagation on benchmarks 1–2 — where a user
exists — is untested. That is the most direct follow-up.

**What these benchmarks represent.** Benchmark 2 is our own criterion, not the
original method's; it is a fair operationalisation of "can these agents tell right
from wrong," but it changes the task and holds the method to a standard it was not
designed against. In benchmarks 1–2 the user is static — asserting once, never
elaborating or reconsidering — which is required for the measurement to be
interpretable but means the sycophancy results describe resistance to a *fixed
assertion*, not to argument. We measure one of the three sycophancy metrics defined
in the prior work; the confident-sycophancy variant needs logprobs, which are not
portable across OpenRouter providers.

**On comparability.** Half the roster is substituted and our calibration set is 430
questions against 250 in the prior work, so absolute numbers are ours alone — our
agents degrade harder in discussion than a stronger lineup would (baseline 0.627 vs
0.804). The contrasts *across* benchmarks, which carry every conclusion here, do not
depend on matching anyone's absolute values.

## Citing this work

If you use the benchmarks, the code, or the balanced-control design:

```bibtex
@misc{sycophancy_vs_accuracy_2026,
  author       = {Zhou, Xiaoping},
  title        = {Does Reducing Sycophancy Mean Improving Accuracy?},
  year         = {2026},
  howpublished = {\url{https://zxp567.github.io/sycophancy-vs-accuracy/}},
  note         = {Code and data: \url{https://github.com/zxp567/sycophancy-vs-accuracy}}
}
```

If you use the discussion protocol, the MMLU subject selection, or the Base
Sycophancy Score, please also cite the work that introduced them --
Kasprova et al., *Too Polite to Disagree: Understanding Sycophancy Propagation in
Multi-Agent Systems*, SIGDIAL 2026 ([arXiv:2604.02668](https://arxiv.org/abs/2604.02668)).
