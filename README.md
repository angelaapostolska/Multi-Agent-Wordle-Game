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

## Running the Scenarios

### English Wordle
```bash
python scenario_english.py          # Train + demo
python scenario_english.py --demo   # Demo only (loads saved weights)
python scenario_english.py --word crane --games 3
```

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

## Notes for Students

- The `Policy` class in `wordle_env_base.py` is a 2-layer neural network written purely in numpy. You can see exactly how forward pass, softmax, and backpropagation work.
- The `partition_quality()` function uses **Gini impurity** instead of Shannon entropy (no logarithms needed), but measures the same thing: how evenly spread the feedback groups are.
- Each scenario file is self-contained — you can read just one to understand the full training loop.
- Saved weights are stored as `.npz` files. Delete them to retrain from scratch.
