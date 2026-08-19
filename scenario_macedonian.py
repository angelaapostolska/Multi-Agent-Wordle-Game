"""
scenario_macedonian.py — Scenario 2: Macedonian Wordle with LLM-Style Debate & Negotiation
"""

import numpy as np
import time
import argparse

from wordle_env_base import (
    WORDS_MK, MAX_TURNS, AGENT_NAMES,
    ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW,
    build_pattern_matrix, filter_candidates,
    task_reward, agent_reward,
    fb_to_str, Policy,
    expected_remaining_frac, partition_quality
)

# ── Macedonian alphabet (31 letters) ──────────────────────────────────────────
MK_ALPHABET = [
    'а','б','в','г','д','ѓ','е','ж','з','ѕ',
    'и','ј','к','л','љ','м','н','њ','о','п',
    'р','с','т','ќ','у','ф','х','ц','ч','џ','ш'
]
MK_ALPHA_IDX = {ch: i for i, ch in enumerate(MK_ALPHABET)}
MK_ALPHA_LEN = len(MK_ALPHABET)

# ── Setup ──────────────────────────────────────────────────────────────────────
WORDS    = WORDS_MK
N        = len(WORDS)
WORD_IDX = {w: i for i, w in enumerate(WORDS)}

print(f"[Macedonian] Building pattern matrix for {N} words...")
PATTERN = build_pattern_matrix(WORDS)
print("[Macedonian] Pattern matrix ready.")

_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR  = _prior / _prior.max()

AGENT_STATE_DIM = N + MAX_TURNS + MK_ALPHA_LEN + 5 * MK_ALPHA_LEN + MK_ALPHA_LEN
MOD_STATE_DIM   = 3 * 7 + 2

EPISODES  = 6000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99
GRAD_CLIP = 5.0
LOG_EVERY = 1000
WINDOW    = 500
SEED      = 42


# ── Macedonian-aware state builder ────────────────────────────────────────────

def mk_agent_state(cands, turn, absent_mk, known_green, yellows):
    mask = np.zeros(N)
    mask[cands] = 1.0

    t = np.zeros(MAX_TURNS)
    if turn < MAX_TURNS:
        t[turn] = 1.0

    green_enc = np.zeros(5 * MK_ALPHA_LEN)
    for pos, ch in enumerate(known_green):
        if ch and ch in MK_ALPHA_IDX:
            green_enc[pos * MK_ALPHA_LEN + MK_ALPHA_IDX[ch]] = 1.0

    yellow_enc = np.zeros(MK_ALPHA_LEN)
    for ch in yellows:
        if ch in MK_ALPHA_IDX:
            yellow_enc[MK_ALPHA_IDX[ch]] = 1.0

    return np.concatenate([mask, t, absent_mk, green_enc, yellow_enc])


def mk_build_mod_state(proposals, cands, turn):
    feats = []
    for agent_id, g in enumerate(proposals):
        elim   = 1.0 - expected_remaining_frac(g, cands, PATTERN)
        pq     = partition_quality(g, cands, PATTERN)
        in_c   = 1.0 if g in cands else 0.0
        common = float(PRIOR[g])

        one_hot = [0.0, 0.0, 0.0]
        one_hot[agent_id] = 1.0

        feats += [elim, pq, in_c, common] + one_hot

    feats += [turn / MAX_TURNS, len(cands) / max(N, 1)]
    return np.array(feats)


# ── LLM-Style Debate & Argument Generation (Option B) ──────────────────────────

def generate_agent_argument(agent_id, guess_idx, cands):
    word = WORDS[guess_idx]

    if agent_id == ELIMINATOR:
        frac = expected_remaining_frac(guess_idx, cands, PATTERN)
        elim_pct = int(100 * (1.0 - frac))
        return f"Предлагам '{word}'. Елиминатор: овој збор отстранува околу {elim_pct}% од кандидатите."

    elif agent_id == PROBABILIST:
        commonness = int(PRIOR[guess_idx] * 100)
        in_play = "Да" if guess_idx in cands else "Не"
        return f"Предлагам '{word}'. Веројатност: индекс на честота е {commonness}%, во игра: {in_play}."

    else:  # RISKTAKER
        pq = round(partition_quality(guess_idx, cands, PATTERN), 2)
        return f"Предлагам '{word}'. Ризикер: коефициент на партиционирање е {pq}."


# ── Single game with Debate Phase ──────────────────────────────────────────────

def run_episode(agent_models, moderator, rng, train_mode=True):
    secret      = int(rng.integers(N))
    cands       = list(range(N))
    absent      = np.zeros(MK_ALPHA_LEN)
    known_green = [None] * 5
    yellows     = set()
    solved      = False
    n_guesses   = 0

    agent_mem, mod_mem = [], []

    for turn in range(MAX_TURNS):
        n_guesses += 1

        a_state   = mk_agent_state(cands, turn, absent, known_green, yellows)
        proposals = [ag.sample(a_state, valid_indices=cands, rng=rng)[0] for ag in agent_models]

        # Debate round
        for ai in range(3):
            _ = generate_agent_argument(ai, proposals[ai], cands)

        m_state        = mk_build_mod_state(proposals, cands, turn)
        choice, _      = moderator.sample(m_state, rng=rng)
        final_guess    = proposals[choice]

        fb      = PATTERN[final_guess, secret]
        last    = (turn == MAX_TURNS - 1)
        solved  = (final_guess == secret)

        tr = task_reward(
            fb, solved, last,
            guess_word=WORDS[final_guess],
            secret_word=WORDS[secret],
            which=choice,
        )
        if len(cands) > 100:
            if choice == ELIMINATOR:
                tr += 0.08
            elif choice == RISKTAKER:
                tr += 0.04
        mod_mem.append((m_state, choice, tr))

        for ai in range(3):
            ar = agent_reward(ai, proposals[ai], cands, secret, WORDS, PATTERN)
            agent_mem.append((ai, a_state, proposals[ai], ar))

        for i in range(5):
            ch = WORDS[final_guess][i]
            if fb[i] == GREEN:
                known_green[i] = ch
            elif fb[i] == YELLOW:
                yellows.add(ch)
            else:
                if ch in MK_ALPHA_IDX:
                    absent[MK_ALPHA_IDX[ch]] = 1.0

        new_cands = filter_candidates(cands, final_guess, fb, PATTERN)
        cands     = new_cands if new_cands else cands

        if solved:
            break

    return solved, n_guesses, agent_mem, mod_mem


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, seed=SEED):
    rng = np.random.default_rng(seed)

    agent_models = [
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=1),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=2),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=3),
    ]
    moderator = Policy(MOD_STATE_DIM, 32, 3, lr, seed=4)

    base_agent = [0.0, 0.0, 0.0]
    base_mod   = 0.0
    wins, guess_log = [], []
    t0 = time.time()

    print(f"\n[Macedonian] Starting training with LLM Debate: {episodes} episodes")
    print("─" * 60)

    for ep in range(episodes):
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agent_models, moderator, rng, train_mode=True
        )

        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else MAX_TURNS)

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
            agent_models[which].pg_step(s, a, r - base_agent[which], grad_clip=GRAD_CLIP)

        if (ep + 1) % LOG_EVERY == 0:
            win_rate  = np.mean(wins[-WINDOW:])
            solved_gs = [g for g, w in zip(guess_log[-WINDOW:], wins[-WINDOW:]) if w]
            avg_g     = np.mean(solved_gs) if solved_gs else float("nan")
            print(f"  ep {ep+1:6d} | win% {win_rate*100:5.1f} | "
                  f"avg guesses (won) {avg_g:.2f} | {time.time()-t0:.1f}s")

    print("─" * 60)
    final_wr  = float(np.mean(wins[-WINDOW:]))
    solved_gs = [g for g, w in zip(guess_log, wins) if w]
    avg_g     = float(np.mean(solved_gs[-WINDOW:])) if solved_gs else float("nan")
    print(f"[Macedonian] Done. Win rate: {final_wr*100:.1f}%  Avg guesses: {avg_g:.2f}")

    return agent_models, moderator


# ── Interactive Demo with Debate Printing ──────────────────────────────────────

def demo_game(agent_models, moderator, secret_word=None):
    rng = np.random.default_rng()
    secret = WORD_IDX.get(secret_word, int(rng.integers(N))) if secret_word else int(rng.integers(N))

    cands       = list(range(N))
    absent      = np.zeros(MK_ALPHA_LEN)
    known_green = [None] * 5
    yellows     = set()

    print(f"\n{'='*65}")
    print(f"  МАКЕДОНСКИ ВОРДЛ — ДЕБАТА И КООРДИНАЦИЈА НА АГЕНТИ")
    print(f"  Тајниот збор: {WORDS[secret].upper()}")
    print(f"{'='*65}")

    for turn in range(MAX_TURNS):
        print(f"\n--- Обид {turn+1} (Преостанати кандидати: {len(cands)}) ---")

        a_state   = mk_agent_state(cands, turn, absent, known_green, yellows)
        proposals = [ag.sample(a_state, valid_indices=cands, rng=rng)[0] for ag in agent_models]

        # Debate round
        for ai in range(3):
            arg = generate_agent_argument(ai, proposals[ai], cands)
            print(f"  [{AGENT_NAMES[ai]}] -> {arg}")

        m_state   = mk_build_mod_state(proposals, cands, turn)
        choice, _ = moderator.sample(m_state, rng=rng)
        final     = proposals[choice]

        fb      = PATTERN[final, secret]
        fb_str  = fb_to_str(fb)
        solved  = (final == secret)

        print(f" МОДЕРАТОРОТ го избра агентот **{AGENT_NAMES[choice]}** со зборот **{WORDS[final].upper()}** | Резултат: {fb_str}")

        for i in range(5):
            ch = WORDS[final][i]
            if fb[i] == GREEN:
                known_green[i] = ch
            elif fb[i] == YELLOW:
                yellows.add(ch)
            else:
                if ch in MK_ALPHA_IDX:
                    absent[MK_ALPHA_IDX[ch]] = 1.0

        new_cands = filter_candidates(cands, final, fb, PATTERN)
        cands     = new_cands if new_cands else cands

        if solved:
            print(f"\n  ✓ Успех! Зборот е погоден за {turn+1} обид/и!")
            return True

    print(f"\n  ✗ Неуспех. Зборот беше {WORDS[secret]}")
    return False


# ── Save / Load Weights ────────────────────────────────────────────────────────

def save_weights(agent_models, moderator, filename="weights_macedonian_debate.npz"):
    np.savez(filename,
        e_W1=agent_models[0].W1, e_b1=agent_models[0].b1, e_W2=agent_models[0].W2, e_b2=agent_models[0].b2,
        p_W1=agent_models[1].W1, p_b1=agent_models[1].b1, p_W2=agent_models[1].W2, p_b2=agent_models[1].b2,
        r_W1=agent_models[2].W1, r_b1=agent_models[2].b1, r_W2=agent_models[2].W2, r_b2=agent_models[2].b2,
        m_W1=moderator.W1, m_b1=moderator.b1, m_W2=moderator.W2, m_b2=moderator.b2,
    )
    print(f"Weights saved → {filename}")


def load_weights(filename="weights_macedonian_debate.npz", lr=LR):
    import os
    if not os.path.exists(filename):
        return None, None
    d      = np.load(filename, allow_pickle=True)
    agent_models = [
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=1),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=2),
        Policy(AGENT_STATE_DIM, 64, N, lr, seed=3),
    ]
    for ag, px in zip(agent_models, ["e", "p", "r"]):
        ag.W1 = d[f"{px}_W1"]; ag.b1 = d[f"{px}_b1"]
        ag.W2 = d[f"{px}_W2"]; ag.b2 = d[f"{px}_b2"]
    mod = Policy(MOD_STATE_DIM, 32, 3, lr, seed=4)
    mod.W1 = d["m_W1"]; mod.b1 = d["m_b1"]
    mod.W2 = d["m_W2"]; mod.b2 = d["m_b2"]
    print(f"Weights loaded ← {filename}")
    return agent_models, mod


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word",  type=str, default=None)
    parser.add_argument("--games", type=int, default=1)
    args = parser.parse_args()

    agent_models, moderator = load_weights()
    if agent_models is None:
        agent_models, moderator = train()
        save_weights(agent_models, moderator)

    print(f"\n[Macedonian] Demonstrating multi-agent debate and negotiation...")
    for _ in range(args.games):
        demo_game(agent_models, moderator, args.word)