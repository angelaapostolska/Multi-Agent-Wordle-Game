# Multi-Agent Wordle: Cooperative RL Agents with an LLM Moderator

Three specialised reinforcement-learning agents and a moderator cooperate to
solve Wordle puzzles. We study the team in three scenarios (English, Macedonian,
multi-word) and ask a research question on top of it: **what happens when a
local LLM is allowed to argue for the agents' guesses, and to overrule the
trained moderator?**

The learning code is pure `numpy`. The LLM layer is optional and runs locally
through [Ollama](https://ollama.com), with `gemma3:4b` as the default model. We
also ran early experiments with `llama3.2`.

---

## Table of contents
1. [Quick start](#quick-start)
2. [Project structure](#project-structure)
3. [How it works](#how-it-works)
4. [The three scenarios](#the-three-scenarios)
5. [LLM integration](#llm-integration)
6. [Experiments and results](#experiments-and-results)
7. [Hypothesis and reasoning](#hypothesis-and-reasoning)
8. [Limitations and future work](#limitations-and-future-work)
9. [Reproducing the results](#reproducing-the-results)
10. [Team](#team)

---

## Quick start

### Requirements
| Needed for | Requirement |
|---|---|
| Training and playing (all scenarios) | Python 3.9+ and `numpy` |
| LLM debate and moderator vote (optional) | [Ollama](https://ollama.com/download) and a pulled model |

```bash
# 1. Clone and enter the project
git clone https://github.com/angelaapostolska/Multi-Agent-Wordle-Game.git
cd Multi-Agent-Wordle-Game

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the only dependency
pip install numpy

# 4. Train and watch the English scenario
python scenario_english.py
```

The first run trains the agents and saves a `.npz` weights file. Later runs with
`--demo` load the saved weights instead of retraining.

### Optional: enable the LLM layer
```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull gemma3:4b            # default model
ollama pull llama3.2             # used in our early experiments (optional)
ollama serve                     # skip if Ollama is already running

python scenario_english.py --demo --llm
```
Choose a different model or host:
```bash
python scenario_english.py --demo --llm --ollama-model llama3.2
python scenario_english.py --demo --llm --ollama-host http://192.168.1.5:11434
# or set a default for the whole shell:
export OLLAMA_MODEL=llama3.2
```
If Ollama isn't running, every LLM call fails safely and falls back to the
template text and the trained moderator's pick. Nothing crashes.

---

## Project structure

```
.
├── wordle_env_base.py                  # Shared base: word lists, feedback logic, rewards,
│                                       #   Policy network, and the shared Ollama/LLM helpers
├── rl_training.py                      # Shared REINFORCE training loop (Trainer class)
├── scenario_english.py                 # Scenario 1: English 5-letter Wordle
├── scenario_macedonian.py              # Scenario 2: Macedonian (Cyrillic) Wordle
├── scenario_multiword.py               # Scenario 3: several secret words at once
├── scenario_macedonian_heuristics_tests.py  # Action-space ("heuristics limit") experiments
├── words_mk_pythonlist.py, zborle_mk_words.csv   # Macedonian word list
├── weights_*.npz                       # Saved trained weights (delete one to retrain)
├── results/                            # Experiment outputs and transcripts
│   ├── 15k ep/                         #   Trained-agent stats, 300-game LLM comparisons
│   ├── llm/                            #   LLM demo/compare transcripts
│   └── heuristics_limit_tests_result.txt
└── README.md
```

---

## How it works

### The team
| Agent | Strategy |
|---|---|
| **Eliminator** | Picks the guess that eliminates the most remaining candidate words |
| **Probabilist** | Favours common, likely-to-be-the-answer words |
| **RiskTaker** | Maximises how evenly a guess splits the remaining candidates (information gain) |
| **Moderator** | Sees all three proposals and chooses which one is actually played each turn |

Each agent and the moderator is a small two-layer neural network written from
scratch in `numpy`. Agents use 64 hidden units and the moderator 32.

### Learning: REINFORCE
All scenarios use REINFORCE (Monte-Carlo policy gradient), implemented in
`rl_training.py`:
1. Play a full game, recording `(state, action, reward)` at every step.
2. Compute the discounted return `G_t = r_t + γ·r_{t+1} + …`.
3. Raise the probability of actions with `G_t` above a running baseline and
   lower it for actions below it.

Training runs for 15,000 episodes with a learning rate of 0.003, using
epsilon-greedy exploration. The LLM is never called during training.

### Rewards
**Team reward (moderator):**
```
+0.2 per green  +0.05 per yellow  −0.1 per turn
+ letter-match bonus
+ info-gain bonus  (fraction of candidates eliminated this turn)
+5.0 if solved       −1.0 if the last turn ends unsolved
```
**Agent reward:** Eliminator is rewarded for the fraction of candidates removed,
Probabilist for guessing a valid candidate and a common word, and RiskTaker for
the Gini impurity of the feedback partition, a log-free stand-in for entropy.
Each agent also gets a weighted letter-match bonus.

### Action space
Each agent chooses among a heuristically shortlisted set of candidate words
(the `valid_indices` limit, set to 5) rather than the whole vocabulary. We tested
how large this shortlist should be (see
`results/heuristics_limit_tests_result.txt`).

---

## The three scenarios

| | English | Macedonian | Multi-word |
|---|---|---|---|
| Word list | 968 English 5-letter words | 967 Macedonian 5-letter words | English 5-letter words |
| Alphabet | 26 Latin letters | 31 Cyrillic letters | 26 Latin letters |
| Secrets per game | 1 | 1 | 2 or more |
| Max turns | 6 | 6 | 8 or more (default for 2 words) |
| Reward | Standard | Standard (language-agnostic) | Normalised with partial credit |
| State vector | Base | Cyrillic-aware | Base × number of words |

```bash
# English
python scenario_english.py                    # train + demo
python scenario_english.py --demo --word crane --games 3

# Macedonian
python scenario_macedonian.py
python scenario_macedonian.py --games 5

# Multi-word
python scenario_multiword.py                              # 2 words, 8 turns
python scenario_multiword.py --num_words 3 --max_turns 9
python scenario_multiword.py --words crane slate          # fix the secrets
```

Common flags: `--games N`, `--seed S`, `--unseeded`, `--stats-only`
(Macedonian and multi-word), `--no-results` (multi-word), plus the LLM flags below.

> If you have an old `weights_english.npz` from before the 968-word list and the
> info-gain reward, delete it and retrain. Its network shape no longer matches.

---

## LLM integration

Two separate, optional features. Both run only in demo or comparison mode, never
during training, so they add zero cost to training.

1. **Debate (`--llm`).** After the trained agents have chosen their words, the
   LLM writes a short in-character argument for each one and can react to what
   the others said that turn. This is purely explanatory. It does not change
   which word is played.
2. **Moderator vote (`--llm-moderator`).** The LLM also votes on which of the
   three proposals to play. The trained moderator still makes its own pick every
   turn, but if the LLM disagrees, the LLM's choice is played.

```bash
python scenario_english.py --demo --llm
python scenario_english.py --demo --llm-moderator
python scenario_english.py --compare-llm --games 300 --seed 42
python scenario_macedonian.py --llm --llm-moderator --games 5 --unseeded
python scenario_multiword.py --compare-llm --games 300
```

### How the comparison is measured (`--compare-llm`)
For each of N games we play the same game twice using the same RNG seed: once
with the LLM moderator active and once with only the trained moderator. Both
conditions see the same secret word and the same agent proposals, so the only
difference is whether the LLM's override is applied. We report:
- how often the LLM **agreed with** vs **overrode** the trained moderator, and
- the **win rate** with the LLM vs the trained-only baseline.

### Scenario-specific design
- **Macedonian.** The model is asked to argue in Macedonian (Cyrillic). This
  tests whether a small general-purpose model has enough competence in a
  lower-resource language. The vote prompt stays in English since it returns
  only a digit.
- **Multi-word.** One shared guess is played against several active targets, so
  the debate reports elimination power and partition quality **per target**
  instead of merging the targets' candidates. This lets the LLM weigh the
  trade-off between targets.

---

## Experiments and results

### 1. Trained agents without the LLM (30,000 seeded games, seed 123)
| Scenario | Win rate | Dominant agent |
|---|---|---|
| English | 98.2% (29,473 / 30,000) | Eliminator wins 96.6% of games |
| Macedonian | 99.4% | Probabilist wins 99.9% of games |

All three agents propose the secret word at similar raw rates (about 25–28% of
turns), so the moderator's choice between them is what matters.

### 2. LLM moderator vs trained-only moderator (300 matched games each)
| Scenario | LLM overrode trained moderator | Win rate WITH LLM | Win rate baseline |
|---|---|---|---|
| English | 57.3% (637 / 1,111 turns) | 100% (300/300) | 97.7% (293/300) |
| Macedonian | 61.3% (722 / 1,178 turns) | 97.7% (293/300) | 100% (300/300) |
| Multi-word | 55.1% (870 / 1,578 turns) | 98.7% (296/300) | 100% (300/300) |

Early 20-game pilot with `llama3.2`: override rates of ~89% (English), 75.9%
(Macedonian) and 93.2% (multi-word), with the same overall pattern.

### 3. Action-space size (Macedonian, 6,000 training episodes)
A shortlist of 25 words reached 98.8% and no limit reached 97.0%. We settled on
a limit of 5 and re-ran the 15k-episode results with it.

Transcripts, per-game logs and raw outputs are in `results/`.

---

## Hypothesis and reasoning

> **Draft: please align with the research paper.**

**H1 (team).** Splitting the guessing task into specialised agents (eliminate,
exploit probability, explore for information) and letting a learned moderator
arbitrate gives near-ceiling win rates across languages and task structures,
without any hand-written strategy.
*Result: supported.* Win rates are 97–100% in all three scenarios.

**H2 (LLM as moderator).** A language model reasons very differently from a
policy learned from reward-scored games. We therefore expect it to disagree with
the trained moderator often, and we ask whether that disagreement helps, hurts
or changes nothing.
*Result: the disagreement is large, but the effect on winning is small.* The LLM
overrode the trained moderator on 55–61% of turns in every scenario, yet win
rates stayed within 2.3 percentage points of the baseline (at most 7 games out
of 300).

**H3 (language competence).** The LLM's judgement is only as good as its command
of the language. We expect it to do worst on the lower-resource language.
*Result: consistent, but not conclusive.* Macedonian is the scenario where the
LLM's moderation did worse than the baseline (97.7% vs 100%). Its debate text
there is also visibly less fluent. Multi-word, the harder task, also dipped
slightly (98.7%). The one scenario where the LLM did better (English, 100% vs
97.7%) is also the one where the model is strongest.

**Reasoning.**
- The trained team is already near the ceiling, so there is very little
  headroom for any moderator to add wins. This is why large behavioural
  differences produce such small outcome differences.
- The LLM does not become more cautious when it is on weaker ground: its
  override rate stayed above 55% even in Macedonian. Confidence did not track
  competence.
- With differences of a few games in 300, we can't claim the LLM is better or
  worse as a moderator. Its clearest contribution is **explainability**: it
  turns opaque numeric decisions into natural-language arguments.

---

## Limitations and future work
- Win-rate gaps are small and based on 300 games per condition, so they are
  suggestive rather than statistically significant. More games, several seeds
  and confidence intervals are needed.
- Only small local models were tested (`gemma3:4b`, `llama3.2`). A larger or
  explicitly multilingual model may change the Macedonian result.
- Because the baseline is near the ceiling, a harder setup (fewer turns, larger
  word lists, more target words) would leave room for coordination or LLM effects
  to show up.
- The LLM is never trained or fine-tuned on this task. It would be natural to
  test few-shot prompting or fine-tuning.

---

## Reproducing the results
```bash
# 1. Train (15,000 episodes, saves weights)
python scenario_english.py
python scenario_macedonian.py
python scenario_multiword.py

# 2. Trained agents only, 30,000 seeded games
python scenario_english.py --demo --games 30000 --seed 123
python scenario_macedonian.py --stats-only --games 30000 --seed 123

# 3. LLM vs baseline, 300 matched games (needs Ollama running)
python scenario_english.py    --compare-llm --games 300 --seed 42
python scenario_macedonian.py --compare-llm --games 300
python scenario_multiword.py  --compare-llm --games 300
```
LLM comparison runs make one Ollama call per turn and take roughly 1.5–2 hours
per scenario on a laptop. Training itself takes seconds to minutes.

---

## Team
Angela Apostolska, Eva Madzar and Stefani Akimovska.
