"""
scenario_multiword.py — Scenario 2: Multi-Word Wordle (e.g., Dordle)

Three agents (Eliminator, Probabilist, RiskTaker) collaborate to solve
multiple target 5-letter English words simultaneously (default: 2 words in 8 tries).
A moderator picks which agent's suggestion to play each turn.

To run training:
    python scenario_multiword.py

To watch a single game after training:
    python scenario_multiword.py --demo

To watch a game with an LLM-generated debate (requires local Ollama):
    python scenario_multiword.py --demo --llm
    python scenario_multiword.py --demo --llm-moderator
"""

import os
import sys
import time
import shutil
import datetime
import argparse
import numpy as np

# Force UTF-8 stdout/stderr. This script prints box-drawing characters,
# checkmarks, and medal emojis (═, ─, ✓, ✗, 🥇🥈🥉); on Windows these crash
# with a UnicodeEncodeError the moment output is redirected to a file
# (e.g. by the results-folder capture below) or run under a non-UTF-8
# system locale, because Python then falls back to the system's ANSI
# codepage instead of UTF-8. reconfigure() needs Python 3.7+.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from wordle_env_base import (
    WORDS_EN, AGENT_NAMES, ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, GREY, build_pattern_matrix, filter_candidates,
    task_reward, agent_reward, expected_remaining_frac, partition_quality,
    fb_to_str, Policy,
    AGENT_PERSONA, ollama_generate, format_debate_context, llm_moderator_vote,
)

# ── Setup & Dimensions ────────────────────────────────────────────────────────

WORDS    = WORDS_EN
N        = len(WORDS)
WORD_IDX = {w: i for i, w in enumerate(WORDS)}

print(f"[MultiWord] Building pattern matrix for {N} words...")
PATTERN = build_pattern_matrix(WORDS)
print("[MultiWord] Pattern matrix ready.")

_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR  = _prior / _prior.max()

DEFAULT_NUM_WORDS = 2
DEFAULT_MAX_TURNS = 8

SINGLE_WORD_DIM = N + DEFAULT_MAX_TURNS + 26 + (5 * 26) + 26
AGENT_STATE_DIM = (SINGLE_WORD_DIM * DEFAULT_NUM_WORDS) + DEFAULT_NUM_WORDS
MOD_STATE_DIM   = (3 * 7 + 2) * DEFAULT_NUM_WORDS

# Training hyperparameters — matched to scenario_english.py / scenario_macedonian.py
EPISODES  = 15000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99        # Baseline smoothing factor
GRAD_CLIP = 5.0
LOG_EVERY = 1000
WINDOW    = 500
SEED      = 42

# ── Ollama config ───────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 20  # seconds


# ── LLM Helpers ───────────────────────────────────────────────────────────────
#
# Slot-aware: with num_words >= 2 targets active at once, a single shared
# guess can help one target far more than another. Rather than flattening
# every active target's candidates into one merged set (which would hide
# that tradeoff), these functions compute and expose per-target stats, so
# the debate/vote can reflect a guess that's strong for target 1 but weak
# for target 2.

def generate_agent_argument(agent_id, guess_idx, cands_list, active_slots):
    """Deterministic argument text fallback — one figure per active target."""
    word = WORDS[guess_idx]

    if agent_id == ELIMINATOR:
        per_target = ", ".join(
            f"target {i+1}: {int(100 * (1.0 - expected_remaining_frac(guess_idx, cands_list[i], PATTERN)))}%"
            for i in active_slots
        )
        return f"I propose '{word}'. Elimination power per active target — {per_target}."

    elif agent_id == PROBABILIST:
        commonness = int(PRIOR[guess_idx] * 100)
        candidate_for = [i + 1 for i in active_slots if guess_idx in cands_list[i]]
        return (f"I propose '{word}'. Frequency rank: {commonness}%, still a candidate "
                f"for target(s): {candidate_for if candidate_for else 'none'}.")

    else:  # RISKTAKER
        per_target = ", ".join(
            f"target {i+1}: {round(partition_quality(guess_idx, cands_list[i], PATTERN), 2)}"
            for i in active_slots
        )
        return f"I propose '{word}'. Partition quality (Gini) per active target — {per_target}."


def generate_agent_argument_llm(agent_id, guess_idx, cands_list, active_slots,
                                 prior_arguments, warn=True):
    """Generate in-character debate arguments via local Ollama LLM, giving the
    model per-target stats so it can reason about cross-target tradeoffs."""
    word = WORDS[guess_idx]
    per_target_stats = {
        f"target_{i+1}": {
            "elimination_%":        int(100 * (1.0 - expected_remaining_frac(guess_idx, cands_list[i], PATTERN))),
            "partition_quality":    round(partition_quality(guess_idx, cands_list[i], PATTERN), 2),
            "still_a_candidate":    guess_idx in cands_list[i],
            "candidates_remaining": len(cands_list[i]),
        }
        for i in active_slots
    }

    context = format_debate_context(AGENT_NAMES, prior_arguments)

    prompt = (
        f"You are playing a multi-target Wordle-style game as {AGENT_PERSONA[agent_id]}, "
        f"where {len(active_slots)} target word(s) must be guessed simultaneously with one "
        f"shared guess per turn. Your strategy already picked the word '{word.upper()}' as "
        f"this turn's guess. Per-target stats for this word: {per_target_stats}. {context}"
        f"In 1-2 short sentences, argue in character for why '{word.upper()}' is a good "
        f"guess right now, noting if it helps one target much more than another"
        + (", briefly reacting to what the other agents said" if prior_arguments else "")
        + ". Do not propose a different word — only argue for this one."
    )

    text = ollama_generate(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST,
                            timeout=OLLAMA_TIMEOUT, warn=warn)
    if text is None:
        return generate_agent_argument(agent_id, guess_idx, cands_list, active_slots)
    return text.replace("\n", " ").strip()


class AgentStats:
    """
    Tracks per-agent performance across episodes — SAME schema as
    scenario_english.py's / scenario_macedonian.py's AgentStats, so
    results are directly comparable across all three scenarios.

    For multi-word games, record_turn()/record_reward() are called once
    per still-active word slot per turn (see run_episode/demo_game) —
    the class itself is completely unmodified from the single-word
    version, so the printed table lines up column-for-column with the
    English/Macedonian scenarios.
    """

    def __init__(self, names):
        self.names = names
        n = len(names)
        self.proposals = [0] * n
        self.chosen = [0] * n
        self.wins = [0] * n
        self.correct_proposals = [0] * n
        self.reward_sum = [0.0] * n
        self.reward_count = [0] * n
        self.win_turns = [[] for _ in range(n)]

    def record_turn(self, proposals, secret, choice, final_guess, turn_idx, solved):
        for ai, g in enumerate(proposals):
            self.proposals[ai] += 1
            if g == secret:
                self.correct_proposals[ai] += 1
        self.chosen[choice] += 1
        if solved:
            self.wins[choice] += 1
            self.win_turns[choice].append(turn_idx + 1)

    def record_reward(self, which, r):
        self.reward_sum[which] += r
        self.reward_count[which] += 1

    def avg_reward(self, i):
        return self.reward_sum[i] / self.reward_count[i] if self.reward_count[i] else float("nan")

    def avg_win_turn(self, i):
        return np.mean(self.win_turns[i]) if self.win_turns[i] else float("nan")

    def pick_rate(self, i):
        total = sum(self.chosen)
        return self.chosen[i] / total if total else 0.0

    def win_share(self, i):
        total_wins = sum(self.wins)
        return self.wins[i] / total_wins if total_wins else 0.0

    def hit_rate(self, i):
        """Of the times this agent's guess was chosen, what fraction won?"""
        return self.wins[i] / self.chosen[i] if self.chosen[i] else 0.0

    def proposal_accuracy(self, i):
        """Of all guesses this agent proposed, what fraction were a secret word?"""
        return self.correct_proposals[i] / self.proposals[i] if self.proposals[i] else 0.0

    def ranked_by_wins(self):
        return sorted(range(len(self.names)), key=lambda i: self.wins[i], reverse=True)

    def print_summary(self, total_episodes=None, elapsed=None):
        print("\n" + "═" * 72)
        print("  AGENT PERFORMANCE SUMMARY")
        print("═" * 72)

        order = self.ranked_by_wins()
        header = (f"{'Rank':<5}{'Agent':<13}{'Wins':>6}{'Win %':>8}{'Chosen':>8}"
                  f"{'Pick %':>8}{'Hit %':>8}{'Avg Turn':>10}{'Avg Rwd':>10}")
        print(header)
        print("─" * 72)
        for rank, i in enumerate(order, start=1):
            name = self.names[i]
            wins = self.wins[i]
            win_pct = self.win_share(i) * 100
            chosen = self.chosen[i]
            pick_pct = self.pick_rate(i) * 100
            hit_pct = self.hit_rate(i) * 100
            avg_turn = self.avg_win_turn(i)
            avg_turn_s = f"{avg_turn:.2f}" if not np.isnan(avg_turn) else "  n/a"
            avg_rwd = self.avg_reward(i)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            print(f"{medal} {rank:<3}{name:<13}{wins:>6}{win_pct:>7.1f}%{chosen:>8}"
                  f"{pick_pct:>7.1f}%{hit_pct:>7.1f}%{avg_turn_s:>10}{avg_rwd:>10.3f}")

        print("─" * 72)
        best_i = order[0]
        worst_i = order[-1]
        print(f"  Most winning guesses:  {self.names[best_i]} ({self.wins[best_i]} wins)")
        print(f"  Fewest winning guesses: {self.names[worst_i]} ({self.wins[worst_i]} wins)")

        print("\n  Raw proposal accuracy (proposed a secret word, regardless of "
              "whether the moderator chose it):")
        for i in range(len(self.names)):
            print(f"    {self.names[i]:<13} {self.correct_proposals[i]:>5} / {self.proposals[i]:<6} "
                  f"({self.proposal_accuracy(i) * 100:5.1f}%)")

        if total_episodes is not None:
            print(f"\n  Episodes played: {total_episodes}")
        if elapsed is not None:
            print(f"  Total time: {elapsed:.1f}s  ({elapsed / max(total_episodes, 1) * 1000:.2f} ms/episode)")
        print("═" * 72)


class GameTimer:
    """Times individual episodes and reports distribution stats — SAME
    schema as scenario_english.py's / scenario_macedonian.py's GameTimer.
    `solved` here means ALL target words were solved (whole-game win),
    and `n_guesses` is the number of turns the whole multi-word game took."""

    def __init__(self):
        self.episode_times = []
        self.guess_counts = []
        self.win_flags = []

    def record(self, elapsed, n_guesses, solved):
        self.episode_times.append(elapsed)
        self.guess_counts.append(n_guesses)
        self.win_flags.append(1 if solved else 0)

    def print_summary(self):
        if not self.episode_times:
            return
        times = np.array(self.episode_times)
        wins = np.array(self.win_flags)
        won_guesses = np.array([g for g, w in zip(self.guess_counts, self.win_flags) if w])

        print("\n" + "─" * 72)
        print("  TIMING & GUESS-COUNT STATS")
        print("─" * 72)
        print(f"  Episodes:            {len(times)}")
        print(f"  Total time:          {times.sum():.2f}s")
        print(f"  Avg time / episode:  {times.mean() * 1000:.3f} ms")
        print(f"  Fastest episode:     {times.min() * 1000:.3f} ms")
        print(f"  Slowest episode:     {times.max() * 1000:.3f} ms")
        print(f"  Win rate:            {wins.mean() * 100:.1f}%")
        if len(won_guesses):
            print(f"  Avg guesses (won):   {won_guesses.mean():.2f}")
            print(f"  Best win (fewest guesses): {won_guesses.min()}")
            print(f"  Worst win (most guesses):  {won_guesses.max()}")
        print("─" * 72)


# ── State Representation ───────────────────────────────────────────────────────

def multiword_agent_state(
    cands_list, turn, absent_list, known_greens, yellows_list, solved_list,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Constructs concatenated agent state vector across all word slots."""
    parts = []
    for i in range(num_words):
        mask = np.zeros(N)
        if not solved_list[i]:
            mask[cands_list[i]] = 1.0

        t = np.zeros(max_turns)
        if turn < max_turns:
            t[turn] = 1.0

        absent = absent_list[i]

        green_enc = np.zeros(5 * 26)
        for pos, ch in enumerate(known_greens[i]):
            if ch:
                green_enc[pos * 26 + (ord(ch) - 97)] = 1.0

        yellow_enc = np.zeros(26)
        for ch in yellows_list[i]:
            yellow_enc[ord(ch) - 97] = 1.0

        parts.extend([mask, t, absent, green_enc, yellow_enc])

    solved_flags = np.array([1.0 if s else 0.0 for s in solved_list])
    parts.append(solved_flags)
    return np.concatenate(parts)


def multiword_mod_state(
    proposals, cands_list, turn,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Constructs moderator state feature vector across candidate sets and proposals."""
    feats = []
    for i in range(num_words):
        cands = cands_list[i]
        for agent_id, g in enumerate(proposals):
            elim   = 1.0 - expected_remaining_frac(g, cands, PATTERN) if cands else 0.0
            pq     = partition_quality(g, cands, PATTERN) if cands else 0.0
            in_c   = 1.0 if (cands and g in cands) else 0.0
            common = float(PRIOR[g])

            one_hot = [0.0, 0.0, 0.0]
            one_hot[agent_id] = 1.0

            feats += [elim, pq, in_c, common] + one_hot

        feats += [turn / max_turns, len(cands) / max(N, 1)]
    return np.array(feats)


# ── Single Episode Execution ──────────────────────────────────────────────────

def run_episode(
    agents, moderator, rng, train_mode=True,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS, stats=None
):
    """
    Runs a single episode of Multi-Word Wordle.

    Reward system matches scenario_english.py / scenario_macedonian.py
    exactly in HOW it's computed — task_reward()/agent_reward() are the
    same shared functions, called the same way (same arguments, including
    cands_before/cands_after for the info-gain term) — just once per
    STILL-ACTIVE word slot instead of once for a single secret:

      - A word slot only contributes a reward this turn if it wasn't
        already solved BEFORE this turn. An already-solved slot has
        nothing left to reward — including it (e.g. with a fake all-green
        feedback) would silently inflate the average with a free +5.0
        every remaining turn, rewarding the team for doing nothing.
      - Each active slot's `solved` flag is whether THIS guess matched
        THAT slot's secret (not whether every word is solved) — the same
        per-guess meaning task_reward() uses in the single-word scenarios.
      - The final team reward is the mean task_reward() across the slots
        that were active this turn, keeping the reward's scale comparable
        to the single-word case regardless of num_words.
      - The same "did the moderator pick the best eliminator?" bonus from
        English/Macedonian is applied, using the same >20-candidates
        threshold (checked against the hardest-still-active slot).
    """
    secrets = list(rng.choice(N, size=num_words, replace=False))

    cands_list   = [list(range(N)) for _ in range(num_words)]
    absent_list  = [np.zeros(26)   for _ in range(num_words)]
    known_greens = [[None] * 5     for _ in range(num_words)]
    yellows_list = [set()          for _ in range(num_words)]
    solved_list  = [False]         * num_words
    n_guesses    = 0

    agent_mem, mod_mem = [], []

    for turn in range(max_turns):
        n_guesses += 1

        # Which slots are still being played going INTO this turn — only
        # these get rewarded/updated for this turn's guess.
        active_before = [i for i in range(num_words) if not solved_list[i]]

        a_state = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list, solved_list,
            num_words, max_turns
        )

        valid = set()
        for i in active_before:
            valid.update(cands_list[i])
        valid = list(valid) if valid else list(range(N))

        proposals = []
        for ai, ag in enumerate(agents):
            val_idx = valid if (ai == PROBABILIST or len(valid) <= 5) else None
            guess_idx, _ = ag.sample(a_state, valid_indices=val_idx, rng=rng)
            proposals.append(guess_idx)

        m_state = multiword_mod_state(proposals, cands_list, turn, num_words, max_turns)

        if train_mode and rng.random() < 0.20:
            choice = int(rng.integers(3))
        else:
            choice, _ = moderator.sample(m_state, rng=rng)

        final_guess = proposals[choice]
        last = (turn == max_turns - 1)

        per_slot_task_rewards = []
        for i in active_before:
            fb = PATTERN[final_guess, secrets[i]]
            cands_before_i = len(cands_list[i])
            new_cands = filter_candidates(cands_list[i], final_guess, fb, PATTERN)
            cands_after_i = len(new_cands) if new_cands else cands_before_i
            solved_i = (final_guess == secrets[i])

            tr_i = task_reward(
                fb, solved_i, last,
                guess_word=WORDS[final_guess],
                secret_word=WORDS[secrets[i]],
                which=choice,
                cands_before=cands_before_i,
                cands_after=cands_after_i,
            )
            per_slot_task_rewards.append(tr_i)

            for pos in range(5):
                ch = WORDS[final_guess][pos]
                if fb[pos] == GREEN:
                    known_greens[i][pos] = ch
                elif fb[pos] == YELLOW:
                    yellows_list[i].add(ch)
                else:
                    absent_list[i][ord(ch) - 97] = 1.0

            cands_list[i] = new_cands if new_cands else cands_list[i]
            if solved_i:
                solved_list[i] = True

        tr = float(np.mean(per_slot_task_rewards)) if per_slot_task_rewards else 0.0

        # Same "did the moderator pick the best eliminator?" bonus as
        # English/Macedonian, using the same 20-candidate threshold —
        # checked against whichever active slot still has the most
        # candidates left, so it fires under the same conditions a
        # single-word game would.
        hardest_active_len = max((len(cands_list[i]) for i in active_before), default=0)
        if hardest_active_len > 20:
            elim_powers = [1.0 - expected_remaining_frac(p, valid, PATTERN) for p in proposals]
            best_agent  = int(np.argmax(elim_powers))
            tr += 0.6 if choice == best_agent else -0.2

        mod_mem.append((m_state, choice, tr))

        for ai in range(3):
            ars = [
                agent_reward(ai, proposals[ai], cands_list[i], secrets[i], WORDS, PATTERN)
                for i in active_before
            ]
            ar = float(np.mean(ars)) if ars else 0.0
            agent_mem.append((ai, a_state, proposals[ai], ar))
            if stats is not None:
                stats.record_reward(ai, ar)

        if stats is not None:
            for i in active_before:
                stats.record_turn(proposals, secrets[i], choice, final_guess, turn,
                                   final_guess == secrets[i])

        if all(solved_list):
            break

    return all(solved_list), n_guesses, agent_mem, mod_mem


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, seed=SEED, track_stats=True):
    """Train multi-word Wordle policy networks using REINFORCE policy gradient.
    Structured identically to scenario_english.py's / scenario_macedonian.py's
    train(): same hyperparameters, same AgentStats/GameTimer tracking, same
    printed summaries at the end — so results line up for direct comparison."""
    rng = np.random.default_rng(seed)

    agents = [
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=1),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=2),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=3),
    ]
    moderator = Policy(MOD_STATE_DIM, 32, 3, lr, seed=4)

    base_agent = [0.0, 0.0, 0.0]
    base_mod   = 0.0

    wins, guess_log = [], []
    stats = AgentStats(AGENT_NAMES) if track_stats else None
    timer = GameTimer()
    t0 = time.time()

    print(f"\n[MultiWord] Starting training: {episodes} episodes, lr={lr}")
    print("─" * 60)

    for ep in range(episodes):
        ep_t0 = time.time()
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agents, moderator, rng, train_mode=True, stats=stats
        )
        ep_elapsed = time.time() - ep_t0
        timer.record(ep_elapsed, n_guesses, solved)

        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else DEFAULT_MAX_TURNS)

        # Moderator Policy Gradient Step
        G, returns = 0.0, []
        for (_, _, r) in reversed(mod_mem):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()

        for (s, c, _), Gt in zip(mod_mem, returns):
            base_mod = BETA * base_mod + (1 - BETA) * Gt
            moderator.pg_step(s, c, Gt - base_mod, grad_clip=GRAD_CLIP)

        # Agent Policy Gradient Steps
        for (which, s, a, r) in agent_mem:
            base_agent[which] = BETA * base_agent[which] + (1 - BETA) * r
            agents[which].pg_step(s, a, r - base_agent[which], grad_clip=GRAD_CLIP)

        if (ep + 1) % LOG_EVERY == 0:
            win_rate  = np.mean(wins[-WINDOW:])
            solved_gs = [g for g, w in zip(guess_log[-WINDOW:], wins[-WINDOW:]) if w]
            avg_g     = np.mean(solved_gs) if solved_gs else float("nan")
            elapsed   = time.time() - t0
            print(f"  ep {ep+1:6d} | win% {win_rate*100:5.1f} | "
                  f"avg guesses (won) {avg_g:.2f} | {elapsed:.1f}s")

    print("─" * 60)
    final_wr  = float(np.mean(wins[-WINDOW:]))
    solved_gs = [g for g, w in zip(guess_log, wins) if w]
    avg_g     = float(np.mean(solved_gs[-WINDOW:])) if solved_gs else float("nan")
    print(f"[MultiWord] Training done. Win rate: {final_wr*100:.1f}%  "
          f"Avg guesses: {avg_g:.2f}")

    if stats is not None:
        stats.print_summary(total_episodes=episodes, elapsed=time.time() - t0)
    timer.print_summary()

    return agents, moderator, stats, timer


# ── Demo & Evaluation Modes ───────────────────────────────────────────────────

def demo_game(
    agents, moderator, use_llm=False, llm_moderator=False,
    rng=None, verbose=True, override_log=None, stats=None,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Play one multi-word game and display board visual feedback.

    stats : optional AgentStats instance — pass the same one across many
    demo_game() calls (e.g. in a --games N batch) to build up a
    performance table identical in format to English's/Macedonian's
    post-training/--games summaries.
    """
    if rng is None:
        rng = np.random.default_rng()

    secrets = list(rng.choice(N, size=num_words, replace=False))
    cands_list   = [list(range(N)) for _ in range(num_words)]
    absent_list  = [np.zeros(26)   for _ in range(num_words)]
    known_greens = [[None] * 5     for _ in range(num_words)]
    yellows_list = [set()          for _ in range(num_words)]
    solved_list  = [False]         * num_words

    use_llm = use_llm or llm_moderator

    if verbose:
        print(f"\n{'='*55}")
        print(f"  MULTI-WORD WORDLE DEMO ({num_words} target words)")
        print(f"  Secrets: {', '.join(WORDS[s].upper() for s in secrets)}")
        if use_llm:
            tag = f"Ollama ({OLLAMA_MODEL})" + (" + moderator override" if llm_moderator else "")
            print(f"  LLM debate: {tag}")
        print(f"{'='*55}")
        print(f"  {'AGENT':<12} {'GUESS':<8} {'BOARDS FEEDBACK':<22} {'CANDS'}")
        print(f"  {'─'*12} {'─'*8} {'─'*22} {'─'*10}")

    for turn in range(max_turns):
        active_before = [i for i in range(num_words) if not solved_list[i]]

        a_state = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list, solved_list,
            num_words, max_turns
        )

        valid = set()
        for i in active_before:
            valid.update(cands_list[i])
        valid = list(valid) if valid else list(range(N))

        proposals = []
        for ai, ag in enumerate(agents):
            val_idx = valid if (ai == PROBABILIST or len(valid) <= 5) else None
            guess_idx, _ = ag.sample(a_state, valid_indices=val_idx, rng=rng)
            proposals.append(guess_idx)

        arguments = []
        if use_llm:
            if verbose:
                cand_counts = "/".join(str(len(c)) for c in cands_list)
                print(f"\n  --- Turn {turn+1} debate (cands: {cand_counts}) ---")
            for ai in range(3):
                arg = generate_agent_argument_llm(
                    ai, proposals[ai], cands_list, active_before,
                    prior_arguments=list(enumerate(arguments)), warn=verbose
                )
                arguments.append(arg)
                if verbose:
                    print(f"    [{AGENT_NAMES[ai]}] {WORDS[proposals[ai]].upper()}: {arg}")
        else:
            arguments = [generate_agent_argument(ai, proposals[ai], cands_list, active_before)
                         for ai in range(3)]

        m_state = multiword_mod_state(proposals, cands_list, turn, num_words, max_turns)
        trained_choice, _ = moderator.sample(m_state, rng=rng)
        choice = trained_choice

        if llm_moderator:
            proposal_words = [WORDS[g].upper() for g in proposals]
            situation = (f"{len(active_before)} active target word(s), candidates remaining "
                         f"per target: {'/'.join(str(len(cands_list[i])) for i in active_before)}")
            llm_choice = llm_moderator_vote(proposal_words, arguments, AGENT_NAMES, situation,
                                             turn, model=OLLAMA_MODEL, host=OLLAMA_HOST,
                                             timeout=OLLAMA_TIMEOUT, warn=verbose)
            if override_log is not None:
                override_log.append({"trained_choice": trained_choice, "llm_choice": llm_choice})
            if llm_choice is not None and llm_choice != trained_choice:
                if verbose:
                    print(f"    [LLM moderator] would play {AGENT_NAMES[llm_choice]}'s "
                          f"'{WORDS[proposals[llm_choice]].upper()}' instead of trained pick — overriding.")
                choice = llm_choice
            elif llm_choice is not None and verbose:
                print(f"    [LLM moderator] agrees with trained pick: {AGENT_NAMES[choice]}.")

        final_guess = proposals[choice]
        fb_strs = []

        for i in range(num_words):
            if solved_list[i]:
                fb_strs.append("SOLVED")
            else:
                fb = PATTERN[final_guess, secrets[i]]
                fb_strs.append(fb_to_str(fb))

                for pos in range(5):
                    ch = WORDS[final_guess][pos]
                    if fb[pos] == GREEN:
                        known_greens[i][pos] = ch
                    elif fb[pos] == YELLOW:
                        yellows_list[i].add(ch)
                    else:
                        absent_list[i][ord(ch) - 97] = 1.0

                new_cands = filter_candidates(cands_list[i], final_guess, fb, PATTERN)
                cands_list[i] = new_cands if new_cands else cands_list[i]

                if final_guess == secrets[i]:
                    solved_list[i] = True

        if stats is not None:
            for i in active_before:
                stats.record_turn(proposals, secrets[i], choice, final_guess, turn,
                                   final_guess == secrets[i])

        if verbose:
            res_str = " | ".join(fb_strs)
            cand_str = "/".join(str(len(c)) for c in cands_list)
            print(f"  {AGENT_NAMES[choice]:<12} {WORDS[final_guess].upper():<8} {res_str:<22} {cand_str}")

        if all(solved_list):
            if verbose:
                print(f"\n  ✓ All words solved in {turn+1} guess{'es' if turn > 0 else ''}!")
            return True

    if verbose:
        unsolved = [WORDS[secrets[i]].upper() for i in range(num_words) if not solved_list[i]]
        print(f"\n  ✗ Failed! Unsolved words: {', '.join(unsolved)}")
    return False


def compare_llm_moderator(agents, moderator, n_games=20, base_seed=123):
    """Compare performance of LLM moderator vs trained moderator network."""
    print(f"\n[MultiWord] Comparing LLM moderator vs trained moderator over {n_games} games each...")
    print("(this makes real Ollama calls for the LLM-moderator half — expect it to take a while)")

    t0 = time.time()
    override_log = []
    llm_wins = 0
    for i in range(n_games):
        rng = np.random.default_rng(base_seed + i)
        won = demo_game(agents, moderator, use_llm=True, llm_moderator=True,
                        rng=rng, verbose=False, override_log=override_log)
        llm_wins += int(won)
        print(f"  [LLM-moderator {i+1:>3}/{n_games}] {'won ' if won else 'lost'} "
              f"| {time.time()-t0:6.1f}s elapsed")

    print(f"  LLM-moderator half done in {time.time()-t0:.1f}s. "
          f"Running the {n_games}-game baseline (no Ollama calls, should be quick)...")

    t1 = time.time()
    baseline_wins = 0
    for i in range(n_games):
        rng = np.random.default_rng(base_seed + i)
        won = demo_game(agents, moderator, use_llm=False, llm_moderator=False,
                        rng=rng, verbose=False)
        baseline_wins += int(won)
    print(f"  Baseline half done in {time.time()-t1:.1f}s.")

    voted    = [e for e in override_log if e["llm_choice"] is not None]
    no_vote  = len(override_log) - len(voted)
    agreed   = sum(1 for e in voted if e["llm_choice"] == e["trained_choice"])
    overrode = len(voted) - agreed

    print("─" * 60)
    print(f"Turns played: {len(override_log)}  |  LLM cast a usable vote on {len(voted)} of them "
          f"(Ollama unreachable/unparsed on {no_vote})")
    if voted:
        print(f"  Agreed with trained moderator:  {agreed:4d} ({100*agreed/len(voted):.1f}%)")
        print(f"  Overrode trained moderator:     {overrode:4d} ({100*overrode/len(voted):.1f}%)")
    print(f"Win rate WITH LLM moderator:      {100*llm_wins/n_games:5.1f}%  ({llm_wins}/{n_games})")
    print(f"Win rate baseline (trained only): {100*baseline_wins/n_games:5.1f}%  ({baseline_wins}/{n_games})")
    print("─" * 60)

    return {
        "n_games":           n_games,
        "turns_played":      len(override_log),
        "turns_with_vote":   len(voted),
        "agreed":            agreed,
        "overrode":          overrode,
        "no_vote":           no_vote,
        "llm_win_rate":      llm_wins / n_games,
        "baseline_win_rate": baseline_wins / n_games,
    }


# ── Weight Persistence ────────────────────────────────────────────────────────

def save_weights(agents, moderator, filename="weights_multiword.npz"):
    np.savez(filename,
        e_W1=agents[0].W1, e_b1=agents[0].b1,
        e_W2=agents[0].W2, e_b2=agents[0].b2,
        p_W1=agents[1].W1, p_b1=agents[1].b1,
        p_W2=agents[1].W2, p_b2=agents[1].b2,
        r_W1=agents[2].W1, r_b1=agents[2].b1,
        r_W2=agents[2].W2, r_b2=agents[2].b2,
        m_W1=moderator.W1, m_b1=moderator.b1,
        m_W2=moderator.W2, m_b2=moderator.b2,
    )
    print(f"Weights saved → {filename}")


def load_weights(filename="weights_multiword.npz", lr=LR):
    if not os.path.exists(filename):
        return None, None
    d      = np.load(filename, allow_pickle=True)
    agents = [
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=1),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=2),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=3),
    ]
    for ag, px in zip(agents, ["e", "p", "r"]):
        ag.W1 = d[f"{px}_W1"]; ag.b1 = d[f"{px}_b1"]
        ag.W2 = d[f"{px}_W2"]; ag.b2 = d[f"{px}_b2"]
    mod = Policy(MOD_STATE_DIM, 32, 3, lr, seed=4)
    mod.W1 = d["m_W1"]; mod.b1 = d["m_b1"]
    mod.W2 = d["m_W2"]; mod.b2 = d["m_b2"]
    print(f"Weights loaded ← {filename}")
    return agents, mod


# ── Results folder: save the full console transcript + weights together ────
#
# Requirement: everything this run prints to the console should ALSO be
# saved as a .txt file in results/, alongside the weights that came out of
# that same run — so a training run and its results/weights stay paired
# and reviewable later without re-running anything.

class _Tee:
    """Duplicates writes to multiple streams (e.g. the real console AND a
    results/*.txt file), so nothing about normal console output changes —
    it just also gets saved."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def start_results_capture(tag="run"):
    """
    Call once at the very start of the script. Mirrors everything printed
    from here on into results/multiword_<tag>_<timestamp>.txt, in addition
    to printing normally. Returns (file_handle, txt_path, timestamp) —
    pass the timestamp to save_weights_to_results() at the end so the
    weights filename lines up with the transcript filename.
    """
    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join("results", f"multiword_{tag}_{ts}.txt")
    f = open(txt_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f, txt_path, ts


def stop_results_capture(f, txt_path):
    """Restores normal stdout/stderr and closes the transcript file."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    f.close()
    print(f"[MultiWord] Full console transcript saved -> {txt_path}")


def save_weights_to_results(agents, moderator, ts, weights_filename="weights_multiword.npz"):
    """Saves a timestamped COPY of the weights into results/, alongside
    the transcript from the same run (see start_results_capture)."""
    results_weights_path = os.path.join("results", f"weights_multiword_{ts}.npz")
    save_weights(agents, moderator, filename=results_weights_path)
    return results_weights_path


def play_many_demo_games(agents, moderator, n_games, seed=None, verbose=True,
                          num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS):
    """
    Play n_games demo games back-to-back, collecting AgentStats + GameTimer
    across all of them, then print a combined ranked summary at the end.
    Mirrors scenario_macedonian.py's play_many_demo_games() exactly, so
    --stats-only behaves the same way here as it does there.

    seed    : if given, game i uses np.random.default_rng(seed + i) — same
              seeded-per-game reproducibility convention as English/
              Macedonian, so the same seed + same weights file reproduces
              the exact same sequence of secret words and samples.
    verbose : set False to suppress per-game WIN/LOSS lines.
    """
    stats = AgentStats(AGENT_NAMES)
    timer = GameTimer()

    for g in range(n_games):
        rng = np.random.default_rng(seed + g) if seed is not None else np.random.default_rng()
        t0 = time.time()
        solved, n_guesses, _, _ = run_episode(
            agents, moderator, rng, train_mode=False, stats=stats,
            num_words=num_words, max_turns=max_turns,
        )
        elapsed = time.time() - t0
        timer.record(elapsed, n_guesses, solved)
        if verbose:
            print(f"  Game {g+1}/{n_games}: {'WIN' if solved else 'LOSS'} "
                  f"in {n_guesses} guess(es), {elapsed*1000:.1f} ms")

    stats.print_summary(total_episodes=n_games, elapsed=sum(timer.episode_times))
    timer.print_summary()
    return stats, timer


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a demo game only")
    parser.add_argument("--games", type=int, default=300,
                         help="Number of demo games (default bumped up so the "
                              "AgentStats summary is statistically stable enough "
                              "to compare across machines/runs; use a small number "
                              "like 1-5 with --unseeded if you just want to watch "
                              "printed output for a few random games)")
    parser.add_argument("--stats-only", action="store_true",
                         help="Skip printed per-turn text; just play --games games "
                              "and print the ranked agent leaderboard + timing stats.")
    parser.add_argument("--llm", action="store_true",
                        help="Generate debate arguments with a local Ollama model")
    parser.add_argument("--llm-moderator", action="store_true",
                        help="Let Ollama vote on proposals and override trained moderator")
    parser.add_argument("--ollama-model", type=str, default=None,
                        help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help=f"Ollama server URL (default: {OLLAMA_HOST})")
    parser.add_argument("--compare-llm", action="store_true",
                        help="Compare LLM moderator vs trained moderator over games")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed for demo games. Game i uses seed "
                             "(--seed + i), so the SAME seed on different machines "
                             "with the SAME weights_multiword.npz reproduces the "
                             "exact same sequence of secret words and samples — "
                             "the recommended way to compare runs across machines.")
    parser.add_argument("--unseeded", action="store_true",
                        help="Opt out of seeding and use a fresh, unseeded RNG per "
                             "game instead — different secret words every run, not "
                             "comparable across machines or repeats.")
    parser.add_argument("--no-results", action="store_true",
                        help="Skip saving a results/*.txt transcript + weights copy "
                             "for this run (still saves/loads the normal "
                             "weights_multiword.npz as before).")
    args = parser.parse_args()

    if args.ollama_model:
        OLLAMA_MODEL = args.ollama_model
    if args.ollama_host:
        OLLAMA_HOST = args.ollama_host

    capture = None
    if not args.no_results:
        tag = "compare" if args.compare_llm else ("stats" if args.stats_only else "demo")
        capture = start_results_capture(tag=tag)

    trained_this_run = False
    agents, moderator = load_weights()
    if agents is None:
        agents, moderator, _, _ = train()
        save_weights(agents, moderator)
        trained_this_run = True

    run_seed = None if args.unseeded else args.seed

    if args.compare_llm:
        compare_llm_moderator(agents, moderator, n_games=args.games, base_seed=args.seed)
    elif args.stats_only:
        tag_s = "UNSEEDED" if run_seed is None else f"SEEDED (base seed={run_seed})"
        print(f"\n[MultiWord] Playing {args.games} {tag_s} games for stats only...")
        play_many_demo_games(agents, moderator, args.games, seed=run_seed,
                              verbose=(args.games <= 20))
    else:
        tag_s = "unseeded" if run_seed is None else f"seeded, base seed={run_seed}"
        print(f"\n[MultiWord] Running {args.games} demo game(s) ({tag_s})...")
        game_stats = AgentStats(AGENT_NAMES)
        # Only print full per-turn board text when the batch is small — a
        # large --games run with full text would flood the terminal, same
        # auto-drop convention as scenario_macedonian.py.
        verbose = args.games <= 20
        wins = 0
        for i in range(args.games):
            rng = np.random.default_rng(run_seed + i) if run_seed is not None else None
            wins += demo_game(agents, moderator, use_llm=args.llm,
                               llm_moderator=args.llm_moderator, stats=game_stats,
                               rng=rng, verbose=verbose)
        print(f"\nResult: {wins}/{args.games} games won")
        if args.games > 1:
            game_stats.print_summary(total_episodes=args.games)

    if capture is not None:
        f, txt_path, ts = capture
        # Always save a results/-folder copy of whatever weights are in
        # play this run (freshly trained or previously loaded), so the
        # transcript and the weights that produced it stay paired.
        save_weights_to_results(agents, moderator, ts)
        stop_results_capture(f, txt_path)