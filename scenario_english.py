"""
scenario_english.py — Scenario 1: Standard English Wordle

Three agents (Eliminator, Probabilist, RiskTaker) collaborate to guess
a 5-letter English word in 6 tries. A moderator picks which agent's
suggestion to actually play each turn.

To run training:
    python scenario_english.py

To watch a single game after training:
    python scenario_english.py --demo
"""

import numpy as np
import json
import time
import argparse

from wordle_env_base import (
    WORDS_EN, MAX_TURNS, AGENT_NAMES,
    ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, GREY,
    build_pattern_matrix, filter_candidates,
    build_agent_state, build_mod_state,
    task_reward, agent_reward,
    fb_to_str, Policy,
)

# ── Setup ──────────────────────────────────────────────────────────────────────

WORDS    = WORDS_EN
N        = len(WORDS)
WORD_IDX = {w: i for i, w in enumerate(WORDS)}

print(f"[English] Building pattern matrix for {N} words...")
PATTERN = build_pattern_matrix(WORDS)
print("[English] Pattern matrix ready.")

# Prior: words earlier in the list are assumed more common
_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR  = _prior / _prior.max()

# State dimensions
AGENT_STATE_DIM = N + MAX_TURNS + 26 + 5 * 26 + 26
MOD_STATE_DIM   = 3 * 4 + 2   # 3 agents × 4 features each + 2 global features

# Training hyperparameters
EPISODES  = 15000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99        # Baseline smoothing factor
GRAD_CLIP = 5.0
LOG_EVERY = 1000
WINDOW    = 500
SEED      = 42


# ── Single game (one episode) ─────────────────────────────────────────────────

def run_episode(agents, moderator, rng, train_mode=True):
    """
    Play one full Wordle game.

    Returns:
        won       : True/False
        n_guesses : how many guesses were needed
        memories  : list of (state, action, reward) tuples for training
    """
    secret      = int(rng.integers(N))
    cands       = list(range(N))
    absent      = np.zeros(26)
    known_green = [None] * 5
    yellows     = set()
    solved      = False
    n_guesses   = 0

    agent_mem = []   # (agent_id, state, action, reward)
    mod_mem   = []   # (state, action, reward)

    for turn in range(MAX_TURNS):
        n_guesses += 1

        a_state = build_agent_state(cands, turn, absent, known_green, yellows, N)

        # Each agent proposes a word
        proposals = [ag.sample(a_state, valid_indices=cands, rng=rng)[0]
                     for ag in agents]

        # Moderator picks which agent's word to play
        m_state        = build_mod_state(proposals, cands, turn, WORDS, PATTERN)
        choice, _      = moderator.sample(m_state, rng=rng)
        final_guess    = proposals[choice]

        fb      = PATTERN[final_guess, secret]
        last    = (turn == MAX_TURNS - 1)
        solved  = (final_guess == secret)

        # Task reward goes to the moderator
        tr = task_reward(
            fb, solved, last,
            guess_word=WORDS[final_guess],
            secret_word=WORDS[secret],
            which=choice,
        )
        mod_mem.append((m_state, choice, tr))

        # Individual rewards go to each agent
        for ai in range(3):
            ar = agent_reward(ai, proposals[ai], cands, secret, WORDS, PATTERN)
            agent_mem.append((ai, a_state, proposals[ai], ar))

        # Update game state based on feedback
        for i in range(5):
            ch = WORDS[final_guess][i]
            if fb[i] == GREEN:
                known_green[i] = ch
            elif fb[i] == YELLOW:
                yellows.add(ch)
            else:
                absent[ord(ch) - 97] = 1.0

        new_cands = filter_candidates(cands, final_guess, fb, PATTERN)
        cands     = new_cands if new_cands else cands

        if solved:
            break

    return solved, n_guesses, agent_mem, mod_mem


# ── Training loop ──────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, seed=SEED):
    """Train all agents using REINFORCE policy gradient."""
    rng = np.random.default_rng(seed)

    # Create one policy network per agent + one for the moderator
    agents = [
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=1),  # Eliminator
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=2),  # Probabilist
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=3),  # RiskTaker
    ]
    moderator = Policy(MOD_STATE_DIM, 32, 3, lr, seed=4)

    # Running baselines for variance reduction (makes training more stable)
    base_agent = [0.0, 0.0, 0.0]
    base_mod   = 0.0

    wins, guess_log = [], []
    t0 = time.time()

    print(f"\n[English] Starting training: {episodes} episodes, lr={lr}")
    print("─" * 60)

    for ep in range(episodes):
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agents, moderator, rng, train_mode=True
        )

        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else MAX_TURNS)

        # ── Train moderator (episodic REINFORCE with discounting) ──
        G, returns = 0.0, []
        for (_, _, r) in reversed(mod_mem):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()
        for (s, c, _), Gt in zip(mod_mem, returns):
            base_mod = BETA * base_mod + (1 - BETA) * Gt
            moderator.pg_step(s, c, Gt - base_mod, grad_clip=GRAD_CLIP)

        # ── Train agents (per-turn bandit update) ──
        for (which, s, a, r) in agent_mem:
            base_agent[which] = BETA * base_agent[which] + (1 - BETA) * r
            agents[which].pg_step(s, a, r - base_agent[which], grad_clip=GRAD_CLIP)

        # ── Logging ──
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
    print(f"[English] Training done. Win rate: {final_wr*100:.1f}%  "
          f"Avg guesses: {avg_g:.2f}")

    return agents, moderator


# ── Demo: watch one game ───────────────────────────────────────────────────────

def demo_game(agents, moderator, secret_word=None):
    """Play one game and print the board so you can see what happened."""
    rng = np.random.default_rng()

    if secret_word is not None:
        secret = WORD_IDX[secret_word.lower()]
    else:
        secret = int(rng.integers(N))

    cands       = list(range(N))
    absent      = np.zeros(26)
    known_green = [None] * 5
    yellows     = set()

    print(f"\n{'='*45}")
    print(f"  ENGLISH WORDLE DEMO")
    print(f"  Secret word: {WORDS[secret].upper()}")
    print(f"{'='*45}")
    print(f"  {'AGENT':<12} {'GUESS':<8} {'RESULT':<8} {'CANDS'}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*6}")

    for turn in range(MAX_TURNS):
        a_state   = build_agent_state(cands, turn, absent, known_green, yellows, N)
        proposals = [ag.sample(a_state, valid_indices=cands, rng=rng)[0]
                     for ag in agents]
        m_state   = build_mod_state(proposals, cands, turn, WORDS, PATTERN)
        choice, _ = moderator.sample(m_state, rng=rng)
        final     = proposals[choice]

        fb      = PATTERN[final, secret]
        fb_str  = fb_to_str(fb)
        solved  = (final == secret)

        print(f"  {AGENT_NAMES[choice]:<12} {WORDS[final].upper():<8} {fb_str:<8} {len(cands)}")

        for i in range(5):
            ch = WORDS[final][i]
            if fb[i] == GREEN:
                known_green[i] = ch
            elif fb[i] == YELLOW:
                yellows.add(ch)
            else:
                absent[ord(ch) - 97] = 1.0

        new_cands = filter_candidates(cands, final, fb, PATTERN)
        cands     = new_cands if new_cands else cands

        if solved:
            print(f"\n  ✓ Solved in {turn+1} guess{'es' if turn > 0 else ''}!")
            return True

    print(f"\n  ✗ Failed! The word was {WORDS[secret].upper()}")
    return False


# ── Save / load weights ────────────────────────────────────────────────────────

def save_weights(agents, moderator, filename="weights_english.npz"):
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


def load_weights(filename="weights_english.npz", lr=LR):
    import os
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true", help="Run a demo game only")
    parser.add_argument("--word",  type=str, default=None, help="Fix the secret word")
    parser.add_argument("--games", type=int, default=5,   help="Number of demo games")
    args = parser.parse_args()

    agents, moderator = load_weights()
    if agents is None:
        agents, moderator = train()
        save_weights(agents, moderator)

    print(f"\n[English] Running {args.games} demo game(s)...")
    wins = sum(demo_game(agents, moderator, args.word) for _ in range(args.games))
    print(f"\nResult: {wins}/{args.games} games won")
