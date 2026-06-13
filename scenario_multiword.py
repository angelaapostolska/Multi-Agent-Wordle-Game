"""
scenario_multiword.py — Scenario 3: Multi-word, Multi-try Wordle

The agents must solve MULTIPLE secret words simultaneously, each with
their own feedback, within a shared pool of turns.

Example with 2 words, 8 turns:
    Turn 1: guess "crane" → gets feedback for both secrets at once
    Turn 2: guess "slate" → gets feedback for both secrets at once
    ...

This is the most complex scenario because:
  - There is one candidate list PER secret word
  - The task_reward sums across all words and normalises
  - Partial credit is given if only some words are solved
  - agent_reward aggregates across all word slots

To run:
    python scenario_multiword.py
    python scenario_multiword.py --num_words 3 --max_turns 9
"""

import numpy as np
import time
import argparse

from wordle_env_base import (
    WORDS_EN as WORDS, MAX_TURNS as BASE_MAX_TURNS,
    AGENT_NAMES, ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, GREY,
    build_pattern_matrix, filter_candidates,
    build_agent_state, build_mod_state,
    task_reward, agent_reward,
    fb_to_str, Policy, partition_quality,
    expected_remaining_frac, letter_match_score,
)

N        = len(WORDS)
WORD_IDX = {w: i for i, w in enumerate(WORDS)}

print(f"[Multi-word] Building pattern matrix for {N} words...")
PATTERN = build_pattern_matrix(WORDS)
print("[Multi-word] Pattern matrix ready.")

_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR  = _prior / _prior.max()

# Training config
EPISODES  = 15000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99
GRAD_CLIP = 5.0
LOG_EVERY = 1000
WINDOW    = 500
SEED      = 42

# Default multi-word settings
DEFAULT_NUM_WORDS = 2     # How many secret words to guess simultaneously
DEFAULT_MAX_TURNS = 8     # More turns allowed since there are more words


# ── Multi-word reward functions ────────────────────────────────────────────────

def task_reward_multi(fb_list, solved_list, last_turn,
                      guess_word=None, secret_words=None, which=None):
    """
    Reward signal for a multi-word turn.

    fb_list      : list of feedback tuples, one per secret word
    solved_list  : list of booleans, True if that word was solved
    last_turn    : True if this is the final allowed guess
    guess_word   : the word that was guessed (same word tested on all secrets)
    secret_words : list of secret word strings (for letter-diff score)
    which        : agent type for diff_weight selection

    The reward is normalised by the number of words so the scale
    stays the same regardless of how many words we're solving.
    """
    from wordle_env_base import DIFF_WEIGHTS

    n_words = len(fb_list)
    total_greens  = sum(sum(1 for x in fb if x == GREEN)  for fb in fb_list)
    total_yellows = sum(sum(1 for x in fb if x == YELLOW) for fb in fb_list)

    # Base signal — normalised per word so scale doesn't grow with n_words
    r = (0.2 * total_greens + 0.05 * total_yellows - 0.1 * n_words) / n_words

    # Letter-difference score averaged over all unsolved secrets
    if guess_word is not None and secret_words is not None:
        diff_weight = DIFF_WEIGHTS.get(which, 0.2)
        avg_diff = sum(
            letter_match_score(guess_word, sw)
            for sw, solved in zip(secret_words, solved_list)
            if not solved          # Only count unsolved ones
        )
        remaining = sum(1 for s in solved_list if not s)
        if remaining > 0:
            avg_diff /= remaining
            r += diff_weight * avg_diff

    n_solved = sum(solved_list)

    if n_solved == n_words:
        r += 5.0          # All words solved — full bonus
    elif n_solved > 0:
        # Partial credit: proportional to how many words were solved
        r += 5.0 * (n_solved / n_words)

    if last_turn:
        # Penalty for each unsolved word
        n_unsolved = n_words - n_solved
        r -= 1.0 * n_unsolved / n_words

    return r


def agent_reward_multi(which, guess_idx, cands_list, secret_list, words, pattern):
    """
    Agent reward for a multi-word turn.

    cands_list  : list of candidate lists, one per secret word
    secret_list : list of secret word indices

    We aggregate the single-word agent_reward across all word slots
    and average it.
    """
    total = 0.0
    for cands, secret in zip(cands_list, secret_list):
        total += agent_reward(which, guess_idx, cands, secret, words, pattern)
    return total / max(len(cands_list), 1)


# ── State builder for multi-word ───────────────────────────────────────────────

def multiword_agent_state(cands_list, turn, absent_list, known_greens, yellows_list,
                          n_words, max_turns):
    """
    Build a state vector that covers all word slots at once.

    We concatenate the state from each word slot so the agent can see
    the full picture. This makes the state bigger but more informative.
    """
    parts = []
    for i in range(n_words):
        part = build_agent_state(
            cands_list[i], turn,
            absent_list[i], known_greens[i], yellows_list[i], N
        )
        parts.append(part)
    # Also tell the agent how many words are still unsolved
    solved_flags = np.array([
        1.0 if len(c) == 1 and list(c)[0] == list(c)[0] else 0.0
        for c in cands_list
    ])
    return np.concatenate(parts + [solved_flags])


def multiword_mod_state(proposals, cands_list, turn, n_words):
    """
    Moderator state: aggregate proposal quality across all word slots.
    """
    feats = []
    for g in proposals:
        # Average the features across all active (unsolved) word slots
        elim_list, pq_list, in_c_list, common_list = [], [], [], []
        for cands in cands_list:
            elim_list.append(1.0 - expected_remaining_frac(g, cands, PATTERN))
            pq_list.append(partition_quality(g, cands, PATTERN))
            in_c_list.append(1.0 if g in cands else 0.0)
            common_list.append(float(PRIOR[g]))
        feats += [
            np.mean(elim_list),
            np.mean(pq_list),
            np.mean(in_c_list),
            np.mean(common_list),
        ]
    # Global features
    avg_cands = np.mean([len(c) for c in cands_list]) / max(N, 1)
    feats += [turn / BASE_MAX_TURNS, avg_cands]
    return np.array(feats)


# ── Single episode ─────────────────────────────────────────────────────────────

def run_episode(agents, moderator, rng, num_words=DEFAULT_NUM_WORDS,
                max_turns=DEFAULT_MAX_TURNS):
    """
    Play one multi-word game.

    Each turn the agents propose one word. That word is tested against
    ALL secret words simultaneously and gets feedback from each.
    A word is "solved" when the guess exactly matches that secret.
    """
    # Pick num_words distinct secret words
    secrets = list(rng.choice(N, size=num_words, replace=False))

    # Each secret word has its own candidate list
    cands_list   = [list(range(N)) for _ in range(num_words)]
    absent_list  = [np.zeros(26)   for _ in range(num_words)]
    known_greens = [[None] * 5     for _ in range(num_words)]
    yellows_list = [set()          for _ in range(num_words)]
    solved_list  = [False]         * num_words

    n_guesses = 0
    agent_mem, mod_mem = [], []

    # State/action dimension: one base state per word slot + solved flags
    single_state_dim = N + BASE_MAX_TURNS + 26 + 5 * 26 + 26
    agent_state_dim  = single_state_dim * num_words + num_words

    for turn in range(max_turns):
        n_guesses += 1

        # Valid actions: union of all unsolved candidate lists
        unsolved_cands = set()
        for i, s in enumerate(solved_list):
            if not s:
                unsolved_cands.update(cands_list[i])
        valid = list(unsolved_cands)

        a_state   = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list,
            num_words, max_turns
        )
        proposals = [ag.sample(a_state, valid_indices=valid, rng=rng)[0]
                     for ag in agents]

        m_state        = multiword_mod_state(proposals, cands_list, turn, num_words)
        choice, _      = moderator.sample(m_state, rng=rng)
        final_guess    = proposals[choice]

        # Get feedback from each unsolved secret
        fb_this_turn   = []
        newly_solved   = []
        for i, secret in enumerate(secrets):
            if solved_list[i]:
                continue   # Already done — skip
            fb = PATTERN[final_guess, secret]
            fb_this_turn.append(fb)

            # Update this word slot's constraints
            for pos in range(5):
                ch = WORDS[final_guess][pos]
                if fb[pos] == GREEN:
                    known_greens[i][pos] = ch
                elif fb[pos] == YELLOW:
                    yellows_list[i].add(ch)
                else:
                    absent_list[i][ord(ch) - 97] = 1.0

            new_c = filter_candidates(cands_list[i], final_guess, fb, PATTERN)
            cands_list[i] = new_c if new_c else cands_list[i]

            if final_guess == secret:
                solved_list[i] = True
                newly_solved.append(i)

        last      = (turn == max_turns - 1)
        all_done  = all(solved_list)

        # Build reward — only using feedback from still-active secrets
        active_secrets = [WORDS[secrets[i]] for i in range(num_words)
                          if not solved_list[i] or i in [
                              j for j, s in enumerate(solved_list) if s and
                              final_guess == secrets[j]
                          ]]
        tr = task_reward_multi(
            fb_this_turn, [final_guess == secrets[i] for i in range(num_words)
                           if not (solved_list[i] and final_guess != secrets[i])],
            last,
            guess_word=WORDS[final_guess],
            secret_words=active_secrets,
            which=choice,
        )
        mod_mem.append((m_state, choice, tr))

        for ai in range(3):
            ar = agent_reward_multi(ai, proposals[ai], cands_list, secrets, WORDS, PATTERN)
            agent_mem.append((ai, a_state, proposals[ai], ar))

        if all_done:
            break

    return all(solved_list), n_guesses, agent_mem, mod_mem


# ── Training ───────────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, num_words=DEFAULT_NUM_WORDS,
          max_turns=DEFAULT_MAX_TURNS, seed=SEED):

    rng = np.random.default_rng(seed)

    single_dim      = N + BASE_MAX_TURNS + 26 + 5 * 26 + 26
    agent_state_dim = single_dim * num_words + num_words
    mod_state_dim   = 3 * 4 + 2

    agents = [
        Policy(agent_state_dim, 64, N, lr, seed=1),
        Policy(agent_state_dim, 64, N, lr, seed=2),
        Policy(agent_state_dim, 64, N, lr, seed=3),
    ]
    moderator = Policy(mod_state_dim, 32, 3, lr, seed=4)

    base_agent = [0.0, 0.0, 0.0]
    base_mod   = 0.0
    wins, guess_log = [], []
    t0 = time.time()

    print(f"\n[Multi-word] Training: {episodes} ep, {num_words} words, "
          f"{max_turns} turns, lr={lr}")
    print("─" * 60)

    for ep in range(episodes):
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agents, moderator, rng, num_words=num_words, max_turns=max_turns
        )

        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else max_turns)

        G, returns = 0.0, []
        for (_, _, r) in reversed(mod_mem):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()
        for (s, c, _), Gt in zip(mod_mem, returns):
            base_mod = BETA * base_mod + (1 - BETA) * Gt
            moderator.pg_step(s, c, Gt - base_mod, grad_clip=GRAD_CLIP)

        for (which, s, a, r) in agent_mem:
            base_agent[which] = BETA * base_agent[which] + (1 - BETA) * r
            agents[which].pg_step(s, a, r - base_agent[which], grad_clip=GRAD_CLIP)

        if (ep + 1) % LOG_EVERY == 0:
            win_rate  = np.mean(wins[-WINDOW:])
            solved_gs = [g for g, w in zip(guess_log[-WINDOW:], wins[-WINDOW:]) if w]
            avg_g     = np.mean(solved_gs) if solved_gs else float("nan")
            print(f"  ep {ep+1:6d} | all-solved% {win_rate*100:5.1f} | "
                  f"avg guesses {avg_g:.2f} | {time.time()-t0:.1f}s")

    print("─" * 60)
    final_wr = float(np.mean(wins[-WINDOW:]))
    solved_gs = [g for g, w in zip(guess_log, wins) if w]
    avg_g = float(np.mean(solved_gs[-WINDOW:])) if solved_gs else float("nan")
    print(f"[Multi-word] Done. All-solved rate: {final_wr*100:.1f}%  "
          f"Avg guesses: {avg_g:.2f}")

    return agents, moderator


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo_game(agents, moderator, num_words=DEFAULT_NUM_WORDS,
              max_turns=DEFAULT_MAX_TURNS, secret_words=None):
    rng = np.random.default_rng()

    if secret_words:
        secrets = [WORD_IDX[w.lower()] for w in secret_words[:num_words]]
    else:
        secrets = list(rng.choice(N, size=num_words, replace=False))

    cands_list   = [list(range(N)) for _ in range(num_words)]
    absent_list  = [np.zeros(26)   for _ in range(num_words)]
    known_greens = [[None] * 5     for _ in range(num_words)]
    yellows_list = [set()          for _ in range(num_words)]
    solved_list  = [False]         * num_words

    print(f"\n{'='*55}")
    print(f"  MULTI-WORD WORDLE DEMO  ({num_words} words, {max_turns} turns)")
    for i, s in enumerate(secrets):
        print(f"  Word {i+1}: {WORDS[s].upper()}")
    print(f"{'='*55}")
    header = f"  {'AGENT':<12} {'GUESS':<8} " + \
             " ".join(f"{'W'+str(i+1):<8}" for i in range(num_words))
    print(header)
    print("  " + "─" * (len(header) - 2))

    for turn in range(max_turns):
        valid = set()
        for i, s in enumerate(solved_list):
            if not s:
                valid.update(cands_list[i])
        valid = list(valid)

        a_state   = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list,
            num_words, max_turns
        )
        proposals = [ag.sample(a_state, valid_indices=valid, rng=rng)[0]
                     for ag in agents]
        m_state   = multiword_mod_state(proposals, cands_list, turn, num_words)
        choice, _ = moderator.sample(m_state, rng=rng)
        final     = proposals[choice]

        fb_strs = []
        for i, secret in enumerate(secrets):
            if solved_list[i]:
                fb_strs.append("✓✓✓✓✓ ")
                continue
            fb      = PATTERN[final, secret]
            fb_strs.append(fb_to_str(fb))

            for pos in range(5):
                ch = WORDS[final][pos]
                if fb[pos] == GREEN:
                    known_greens[i][pos] = ch
                elif fb[pos] == YELLOW:
                    yellows_list[i].add(ch)
                else:
                    absent_list[i][ord(ch) - 97] = 1.0

            new_c = filter_candidates(cands_list[i], final, fb, PATTERN)
            cands_list[i] = new_c if new_c else cands_list[i]
            if final == secret:
                solved_list[i] = True

        row = (f"  {AGENT_NAMES[choice]:<12} {WORDS[final].upper():<8} " +
               " ".join(f"{s:<8}" for s in fb_strs))
        print(row)

        if all(solved_list):
            print(f"\n  ✓ All {num_words} words solved in {turn+1} turns!")
            return True
        if turn == max_turns - 1:
            unsolved = [WORDS[secrets[i]].upper()
                        for i, s in enumerate(solved_list) if not s]
            print(f"\n  ✗ Failed! Unsolved words: {', '.join(unsolved)}")
            return False

    return False


# ── Save / load ────────────────────────────────────────────────────────────────

def save_weights(agents, moderator, num_words, filename=None):
    if filename is None:
        filename = f"weights_multiword_{num_words}w.npz"
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


def load_weights(num_words=DEFAULT_NUM_WORDS, lr=LR,
                 max_turns=DEFAULT_MAX_TURNS, filename=None):
    import os
    if filename is None:
        filename = f"weights_multiword_{num_words}w.npz"
    if not os.path.exists(filename):
        return None, None

    single_dim      = N + BASE_MAX_TURNS + 26 + 5 * 26 + 26
    agent_state_dim = single_dim * num_words + num_words
    mod_state_dim   = 3 * 4 + 2

    d      = np.load(filename, allow_pickle=True)
    agents = [
        Policy(agent_state_dim, 64, N, lr, seed=1),
        Policy(agent_state_dim, 64, N, lr, seed=2),
        Policy(agent_state_dim, 64, N, lr, seed=3),
    ]
    for ag, px in zip(agents, ["e", "p", "r"]):
        ag.W1 = d[f"{px}_W1"]; ag.b1 = d[f"{px}_b1"]
        ag.W2 = d[f"{px}_W2"]; ag.b2 = d[f"{px}_b2"]
    mod = Policy(mod_state_dim, 32, 3, lr, seed=4)
    mod.W1 = d["m_W1"]; mod.b1 = d["m_b1"]
    mod.W2 = d["m_W2"]; mod.b2 = d["m_b2"]
    print(f"Weights loaded ← {filename}")
    return agents, mod


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_words", type=int, default=DEFAULT_NUM_WORDS)
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--games",     type=int, default=3)
    parser.add_argument("--words",     nargs="+", default=None,
                        help="Fix the secret words e.g. --words crane slate")
    args = parser.parse_args()

    agents, moderator = load_weights(num_words=args.num_words,
                                     max_turns=args.max_turns)
    if agents is None:
        agents, moderator = train(num_words=args.num_words,
                                  max_turns=args.max_turns)
        save_weights(agents, moderator, num_words=args.num_words)

    print(f"\n[Multi-word] Running {args.games} demo game(s) "
          f"({args.num_words} words, {args.max_turns} turns)...")
    wins = sum(
        demo_game(agents, moderator,
                  num_words=args.num_words,
                  max_turns=args.max_turns,
                  secret_words=args.words)
        for _ in range(args.games)
    )
    print(f"\nResult: {wins}/{args.games} games won (all words solved)")
