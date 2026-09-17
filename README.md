# Multi-Agent Wordle Project

A reinforcement learning project where three specialised AI agents collaborate to solve Wordle puzzles across three different scenarios.

## Project Structure

```
multi_agent_wordle/
│
├── wordle_env_base.py       # Shared base: word lists, feedback logic, reward functions, Policy class
├── rl_training.py           # Shared RL training loop (REINFORCE algorithm)
│
├── scenario_english.py      # Scenario 1: Standard English 5-letter Wordle
├── scenario_macedonian.py   # Scenario 2: Macedonian Wordle (Cyrillic alphabet)
├── scenario_multiword.py    # Scenario 3: Guess multiple words simultaneously
│
└── README.md
```

## The Three Agents

| Agent | Strategy |
|---|---|
| **Eliminator** | Tries to eliminate as many candidate words as possible each turn |
| **Probabilist** | Favours guessing common/likely words from the candidate list |
| **RiskTaker** | Maximises how evenly the guess splits remaining candidates (information gain) |

A **Moderator** agent observes all three proposals and picks the best one to play each turn.

## Reward System

### Task Reward (for the Moderator)
```
reward = 0.2 × greens + 0.05 × yellows - 0.1 (per turn)
       + letter_match_score × weight          (NEW: raw letter proximity)
       + 5.0 if solved
       - 1.0 if last turn and not solved
```

The **letter_match_score** is new — it measures how many letter positions match exactly between the guess and the secret, giving the agents an additional signal about how "close" they are beyond just green/yellow feedback.

### Agent Reward (per agent type)
- **Eliminator**: fraction of candidates eliminated
- **Probabilist**: mix of "is this guess a valid candidate" + word commonness
- **RiskTaker**: Gini impurity of the partition (replaces log-entropy — same idea, simpler math)

Each agent type also gets a weighted **letter_match_score** bonus:
- Eliminator: weight 0.2
- Probabilist: weight 0.35 (highest — letter proximity matters most for likely-word guessing)
- RiskTaker: weight 0.1 (lowest — exploratory guesses shouldn't be penalised)

## LLM Integration (English Scenario)

*A plain-language explanation of what was added and what we learned — see "LLM Debate (Ollama)" further down for the actual setup/usage instructions.*

**The problem this solves:** the three agents and the moderator that picks between them are just small neural networks — every decision they make comes out as a number (a probability, a score), with nothing that explains *why* a word was chosen in a way a person could follow. The LLM integration adds a layer that can actually talk, on top of the existing game, without changing how the game itself is trained or played by default.

**What it actually does — two separate, optional features:**
1. **Debate (`--llm`):** each turn, after the three agents have already picked their words (using the same trained networks as before — nothing about *that* changes), a small LLM running locally via [Ollama](https://ollama.com) writes a short, in-character argument for each agent's word, and can react to what the other agents already said that turn. This is purely for readability — it does not affect which word gets played.
2. **Moderator vote (`--llm-moderator`):** the LLM is also asked to vote on which of the three proposed words the team should actually play. The trained "moderator" network still makes its own pick every turn as normal; if the LLM disagrees, its vote is the one that gets played instead.

**Important boundary:** the LLM is only ever used when watching a demo game — it is never called during training. Training the agents (the 15,000-episode learning process) is completely unaffected, costs nothing extra, and runs at the same speed whether or not any LLM feature is turned on.

**What we tested, and what we found:** using `--compare-llm`, we ran 20 games with the LLM's vote active against 20 matched games using only the trained moderator (same secret words and same agent proposals in both, so the comparison isolates just the effect of the LLM's vote):
- The LLM **overrode the trained moderator's pick on ~89% of turns** — it very rarely agreed with what the trained network wanted to play.
- Despite disagreeing almost every turn, **both approaches won all 20/20 games.**
- Takeaway: the LLM's judgment is built on completely different reasoning than the trained network (language-based impressions vs. a policy learned from thousands of reward-scored games), and the two disagree constantly — but in this game, that disagreement didn't cost any wins. It's a genuinely different way of deciding that turned out to be just as effective here, not a strictly better or worse one.

## Running the Scenarios

### English Wordle
```bash
python scenario_english.py          # Train + demo
python scenario_english.py --demo   # Demo only (loads saved weights)
python scenario_english.py --word crane --games 3
```

> **If you already have a `weights_english.npz` from before this update, delete it and retrain.**
> The word list grew from 535 to 968 words and the reward function changed, so an old weights
> file won't match the current network shape — `--demo` will load stale, incompatible weights
> (or crash) instead of using the current training setup. Just run `rm -f weights_english.npz`
> once, then the next `python scenario_english.py` call will retrain fresh automatically.

#### LLM Debate (Ollama) — English scenario only

The English scenario can optionally use a local [Ollama](https://ollama.com)
model to run the agents' turn-by-turn debate, instead of the fixed template
text.

**Setup (one-time):**
```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a small instruct model
ollama pull llama3.2
# 3. Make sure the Ollama server is running (it starts automatically on
#    most installs; otherwise: `ollama serve`)
```

**Usage:**
```bash
# Natural-language debate text, generated by the LLM each turn
python scenario_english.py --demo --llm

# The LLM also gets a vote: it can override the trained moderator's pick
# when it disagrees with it
python scenario_english.py --demo --llm-moderator

# Use a different model / a remote Ollama host
python scenario_english.py --demo --llm --ollama-model mistral
python scenario_english.py --demo --llm --ollama-host http://192.168.1.5:11434
```

**Where this fits, and where it deliberately doesn't:**
- The LLM is *only* ever called from `demo_game()` — never from `run_episode()`
  or `train()`. Training remains the pure numpy/REINFORCE loop described
  above, completely unaffected by whether `--llm` is passed: no extra API/
  local-model calls are added per training episode, no matter how many
  episodes you train for.
- `--llm` alone is cosmetic/explanatory: the agents' *word choices* still
  come entirely from the trained policy networks — the LLM only puts each
  agent's existing pick into natural language, and (unlike the old
  template) can react to what the other two agents already argued that turn.
- `--llm-moderator` is the only flag that changes gameplay: the trained
  moderator network still runs every turn (so you can see both picks when
  they differ), but the LLM's vote decides which one is actually played.
- If Ollama isn't running or a model isn't pulled, every LLM call fails
  safely and falls back to the deterministic template text (and the trained
  moderator's pick) — `--llm`/`--llm-moderator` never crash the demo, they
  just silently degrade to the non-LLM behavior.

**Measuring how often the LLM actually overrides the trained moderator**, rather than eyeballing individual games:
```bash
python scenario_english.py --compare-llm --games 20
python scenario_english.py --compare-llm --games 50 --seed 7   # different matched-seed sample
```
This runs `--games` games with the LLM moderator active, then the *same*
number of games with only the trained moderator — using a matched RNG seed
per game index across both halves, so game *i* sees the exact same secret
word and the exact same agent proposals in both conditions. The only thing
that can differ is whether the LLM's override actually gets applied, which
isolates its effect from ordinary run-to-run randomness. Instead of
per-turn debate text, it prints one summary:
```
Turns played: 71  |  LLM cast a usable vote on 71 of them (Ollama unreachable/unparsed on 0)
  Agreed with trained moderator:    15 (21.1%)
  Overrode trained moderator:       56 (78.9%)
Win rate WITH LLM moderator:       98.0%  (49/50)
Win rate baseline (trained only): 100.0%  (50/50)
```
This still makes real Ollama calls for every turn of the LLM-moderator
half (bounded by `--games`, same as the normal demo — just multiplied by
it), so a large `--games` will take a while.

### Macedonian Wordle
```bash
python scenario_macedonian.py
python scenario_macedonian.py --games 5
```

### Multi-word Wordle
```bash
python scenario_multiword.py                          # 2 words, 8 turns (default)
python scenario_multiword.py --num_words 3 --max_turns 9
python scenario_multiword.py --words crane slate      # Fix the secret words
```

## Key Differences Between Scenarios

| | English | Macedonian | Multi-word |
|---|---|---|---|
| Word list | English 5-letter | Macedonian Cyrillic 5-letter | English 5-letter |
| Alphabet | 26 Latin letters | 31 Cyrillic letters | 26 Latin letters |
| Secrets per game | 1 | 1 | 2+ |
| Max turns | 6 | 6 | 8+ |
| `task_reward` | Standard | Standard (language-agnostic) | Normalised + partial credit |
| `agent_reward` | Standard | Standard | Averaged across word slots |
| State vector | Base size | Cyrillic-aware | Base × num_words |

## The RL Algorithm: REINFORCE

All scenarios use the REINFORCE policy gradient algorithm (also called Monte Carlo Policy Gradient):

1. Play a full game, recording `(state, action, reward)` at each step
2. Calculate **discounted return** `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`
3. Update network weights: increase probability of actions with `G_t > baseline`, decrease for `G_t < baseline`
4. The **baseline** is a running average of recent returns, which stabilises training

This is implemented in `rl_training.py` as the `Trainer` class.

## Requirements

```
numpy
```

No other dependencies needed — the neural networks are implemented from scratch using only numpy.

## LLM Integration Across All Scenarios

All three scenarios (English, Macedonian, Multi-word) now support the same
`--llm` / `--llm-moderator` / `--compare-llm` feature set. The shared
plumbing — the Ollama HTTP call, the agent personas, and the
moderator-vote prompt — lives once in `wordle_env_base.py`
(`ollama_generate()`, `AGENT_PERSONA`, `format_debate_context()`,
`llm_moderator_vote()`) and each scenario file only supplies its own
argument-generation logic on top of it.

As with the original English-only version: the LLM is only ever called
from `demo_game()`/`compare_llm_moderator()`, never from `run_episode()`
or `train()` — turning `--llm` on adds zero cost to training, regardless
of scenario or episode count.

### English

Unchanged from the original design (see the section above): the LLM
argues in English for each agent's already-chosen word, and can
optionally cast the vote that decides which proposal actually gets
played.

### Macedonian

Same `--llm`/`--llm-moderator` mechanics, but `generate_agent_argument_llm()`
explicitly asks the model to argue **in Macedonian (Cyrillic)**, not
English — a deliberate test of whether a small general-purpose model like
`llama3.2` has enough Macedonian competence to produce coherent, on-topic
Cyrillic text about a Macedonian word, rather than falling back to
English or hallucinating. The moderator-vote prompt itself stays in
English, since it only needs to return a digit (0/1/2) — this keeps the
lower-stakes parsing step language-independent while the actual debate
content is the real cross-language test.

```bash
python scenario_macedonian.py --llm --llm-moderator --games 5 --unseeded
python scenario_macedonian.py --compare-llm --games 20
```

**Open question this is meant to surface, not answer in the code:** if
`llama3.2` turns out to have poor Macedonian competence (English replies,
generic filler, hallucinated words), that's a real finding — it would
mean a low-resource language needs a larger or explicitly multilingual
model to get a meaningful debate layer, which is worth flagging as a
limitation/future-work item rather than working around silently.

### Multi-word

Multi-word plays one shared guess per turn against 2+ simultaneously
active target words (the trained agents only ever produce one guess —
there's no per-target action space), so the moderator vote stays a single
3-way choice just like the other two scenarios. What's different is the
**content** of the debate: `generate_agent_argument()` and
`generate_agent_argument_llm()` compute elimination-power, partition
quality, and candidacy **per active target** rather than merging every
active target's candidates into one flattened set. That per-target
breakdown is what actually makes multi-word interesting to study here —
a guess can help one target far more than another, and the LLM's
argument/vote can reflect that tradeoff (e.g. "eliminates 90% of target
1's candidates but only 20% of target 2's") instead of it being averaged
away before the model ever sees it.

```bash
python scenario_multiword.py --llm --llm-moderator --games 5 --unseeded
python scenario_multiword.py --compare-llm --games 20
```

### Results

`results/llm/` contains one demo transcript and one `--compare-llm`
summary for Macedonian and for multi-word, all captured from a real,
locally running Ollama server (`llama3.2`) — not fallback text. English's
real-Ollama results are already documented in the "What we tested, and
what we found" section above.

**Headline findings across all three scenarios** (LLM override rate vs.
whether that override actually cost games, `--compare-llm --games 20`
except English which used `--games 20` at 50-per-half):

| Scenario   | Override rate | Win rate WITH LLM | Win rate baseline |
|------------|---------------|--------------------|--------------------|
| English    | ~89%          | 100% (20/20)        | 100% (20/20)       |
| Macedonian | 75.9%         | 90% (18/20)         | 100% (20/20)       |
| Multi-word | 93.2%         | 100% (20/20)        | 100% (20/20)       |

The LLM overrides the trained moderator most of the time in all three
scenarios — that part doesn't depend on task or language. What differs is
whether overriding costs anything: it doesn't in English or multi-word
(even though multi-word's override rate is the *highest* of the three,
i.e. harder task ≠ worse outcome), but it does in Macedonian, where the
model's own debate text is visibly less fluent (mixed Latin-script
fragments, invented words, broken grammar — see `mk_llm_demo.txt`).
Notably, the override rate in Macedonian didn't drop to compensate for
that lower reliability — the model kept overriding confidently even
though it was worse at the underlying task, which is arguably the more
interesting finding than the raw win-rate drop itself.

## Notes for Students

- The `Policy` class in `wordle_env_base.py` is a 2-layer neural network written purely in numpy. You can see exactly how forward pass, softmax, and backpropagation work.
- The `partition_quality()` function uses **Gini impurity** instead of Shannon entropy (no logarithms needed), but measures the same thing: how evenly spread the feedback groups are.
- Each scenario file is self-contained — you can read just one to understand the full training loop.
- Saved weights are stored as `.npz` files. Delete them to retrain from scratch.
