"""
scenario_macedonian.py — Scenario 2: Macedonian Wordle with LLM-Style Debate & Negotiation

Includes agent performance statistics: which agent proposed the most winning
guesses, how often each agent was chosen by the moderator, timing, guess-count
distribution, and a ranked leaderboard printed at the end of training/demo.
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


# ── Stats tracking ──────────────────────────────────────────────────────────────

class AgentStats:
    """
    Tracks per-agent performance across episodes:
      - proposals: how many times this agent proposed a guess at all
      - chosen:    how many times the moderator picked this agent's proposal
      - wins:      how many times the *chosen, played* guess was correct
                    (i.e. this agent gets credit for the winning guess)
      - correct_proposals: how many times this agent PROPOSED the secret word,
                    whether or not the moderator picked it
      - reward_sum / reward_count: running total for average agent_reward
      - win_turns: list of turn numbers (1-indexed) on which this agent's
                    chosen guess won
    """

    def __init__(self, names):
        self.names = names
        n = len(names)
        self.proposals          = [0] * n
        self.chosen             = [0] * n
        self.wins               = [0] * n
        self.correct_proposals  = [0] * n
        self.reward_sum         = [0.0] * n
        self.reward_count       = [0] * n
        self.win_turns          = [[] for _ in range(n)]

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
        self.reward_sum[which]   += r
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
        """Of all guesses this agent proposed, what fraction were the secret word?"""
        return self.correct_proposals[i] / self.proposals[i] if self.proposals[i] else 0.0

    def ranked_by_wins(self):
        order = sorted(range(len(self.names)), key=lambda i: self.wins[i], reverse=True)
        return order

    def print_summary(self, total_episodes=None, elapsed=None):
        print("\n" + "═" * 72)
        print("  AGENT PERFORMANCE SUMMARY")
        print("═" * 72)

        order = self.ranked_by_wins()
        header = f"{'Rank':<5}{'Agent':<13}{'Wins':>6}{'Win %':>8}{'Chosen':>8}{'Pick %':>8}{'Hit %':>8}{'Avg Turn':>10}{'Avg Rwd':>10}"
        print(header)
        print("─" * 72)
        for rank, i in enumerate(order, start=1):
            name       = self.names[i]
            wins       = self.wins[i]
            win_pct    = self.win_share(i) * 100
            chosen     = self.chosen[i]
            pick_pct   = self.pick_rate(i) * 100
            hit_pct    = self.hit_rate(i) * 100
            avg_turn   = self.avg_win_turn(i)
            avg_turn_s = f"{avg_turn:.2f}" if not np.isnan(avg_turn) else "  n/a"
            avg_rwd    = self.avg_reward(i)
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            print(f"{medal} {rank:<3}{name:<13}{wins:>6}{win_pct:>7.1f}%{chosen:>8}{pick_pct:>7.1f}%{hit_pct:>7.1f}%{avg_turn_s:>10}{avg_rwd:>10.3f}")

        print("─" * 72)
        best_i  = order[0]
        worst_i = order[-1]
        print(f"  Most winning guesses:  {self.names[best_i]} ({self.wins[best_i]} wins)")
        print(f"  Fewest winning guesses: {self.names[worst_i]} ({self.wins[worst_i]} wins)")

        print("\n  Raw proposal accuracy (proposed the secret word, regardless of "
              "whether the moderator chose it):")
        for i in range(len(self.names)):
            print(f"    {self.names[i]:<13} {self.correct_proposals[i]:>5} / {self.proposals[i]:<6} "
                  f"({self.proposal_accuracy(i)*100:5.1f}%)")

        if total_episodes is not None:
            print(f"\n  Episodes played: {total_episodes}")
        if elapsed is not None:
            print(f"  Total time: {elapsed:.1f}s  ({elapsed/max(total_episodes,1)*1000:.2f} ms/episode)")
        print("═" * 72)


class GameTimer:
    """Simple wrapper to time individual episodes and report distribution stats."""

    def __init__(self):
        self.episode_times = []
        self.guess_counts  = []
        self.win_flags     = []

    def record(self, elapsed, n_guesses, solved):
        self.episode_times.append(elapsed)
        self.guess_counts.append(n_guesses)
        self.win_flags.append(1 if solved else 0)

    def print_summary(self):
        if not self.episode_times:
            return
        times = np.array(self.episode_times)
        wins  = np.array(self.win_flags)
        won_guesses = np.array([g for g, w in zip(self.guess_counts, self.win_flags) if w])

        print("\n" + "─" * 72)
        print("  TIMING & GUESS-COUNT STATS")
        print("─" * 72)
        print(f"  Episodes:            {len(times)}")
        print(f"  Total time:          {times.sum():.2f}s")
        print(f"  Avg time / episode:  {times.mean()*1000:.3f} ms")
        print(f"  Fastest episode:     {times.min()*1000:.3f} ms")
        print(f"  Slowest episode:     {times.max()*1000:.3f} ms")
        print(f"  Win rate:            {wins.mean()*100:.1f}%")
        if len(won_guesses):
            print(f"  Avg guesses (won):   {won_guesses.mean():.2f}")
            print(f"  Best win (fewest guesses): {won_guesses.min()}")
            print(f"  Worst win (most guesses):  {won_guesses.max()}")
        print("─" * 72)


# ── Single game with Debate Phase ──────────────────────────────────────────────

def run_episode(agent_models, moderator, rng, secret=None, train_mode=True, stats=None):
    if secret is None:
        secret = int(rng.integers(N))

    cands = list(range(N))
    absent = np.zeros(MK_ALPHA_LEN)
    known_green = [None] * 5
    yellows = set()
    solved = False
    n_guesses = 0

    agent_mem, mod_mem = [], []

    for turn in range(MAX_TURNS):
        n_guesses += 1

        a_state = mk_agent_state(cands, turn, absent, known_green, yellows)
        proposals = []
        for ai, ag in enumerate(agent_models):
            # Restrict Eliminator & RiskTaker only when candidate pool is <= 25
            if ai == PROBABILIST or len(cands) <= 25:
                valid_idx = cands
            else:
                valid_idx = None  # Allow full vocabulary search for exploratory info gain

            guess_idx, _ = ag.sample(a_state, valid_indices=valid_idx, rng=rng)
            proposals.append(guess_idx)

        # Debate round
        for ai in range(3):
            _ = generate_agent_argument(ai, proposals[ai], cands)

        m_state = mk_build_mod_state(proposals, cands, turn)

        # ── 1. EPSILON-GREEDY EXPLORATION (20% early exploration during training)
        if train_mode and rng.random() < 0.20:
            choice = int(rng.integers(3))
        else:
            choice, _ = moderator.sample(m_state, rng=rng)

        final_guess = proposals[choice]

        fb = PATTERN[final_guess, secret]
        last = (turn == MAX_TURNS - 1)
        solved = (final_guess == secret)

        new_cands = filter_candidates(cands, final_guess, fb, PATTERN)

        tr = task_reward(
            fb, solved, last,
            guess_word=WORDS[final_guess],
            secret_word=WORDS[secret],
            which=choice,
        )

        # ── 2. RELATIVE INFORMATION GAIN REWARD ─────────────────────────────
        if len(cands) > 20:
            elim_powers = [1.0 - expected_remaining_frac(p, cands, PATTERN) for p in proposals]
            best_agent = int(np.argmax(elim_powers))

            if choice == best_agent:
                tr += 0.6
            else:
                tr -= 0.2

        mod_mem.append((m_state, choice, tr))

        for ai in range(3):
            ar = agent_reward(ai, proposals[ai], cands, secret, WORDS, PATTERN)
            agent_mem.append((ai, a_state, proposals[ai], ar))
            if stats is not None:
                stats.record_reward(ai, ar)

        if stats is not None:
            stats.record_turn(proposals, secret, choice, final_guess, turn, solved)

        for i in range(5):
            ch = WORDS[final_guess][i]
            if fb[i] == GREEN:
                known_green[i] = ch
            elif fb[i] == YELLOW:
                yellows.add(ch)
            else:
                if ch in MK_ALPHA_IDX:
                    absent[MK_ALPHA_IDX[ch]] = 1.0

        cands = new_cands if new_cands else cands

        if solved:
            break

    return solved, n_guesses, agent_mem, mod_mem


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, seed=SEED, track_stats=True):
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

    stats = AgentStats(AGENT_NAMES) if track_stats else None
    timer = GameTimer()

    t0 = time.time()

    print(f"\n[Macedonian] Starting training with LLM Debate: {episodes} episodes")
    print("─" * 60)

    for ep in range(episodes):
        ep_t0 = time.time()
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agent_models, moderator, rng, train_mode=True, stats=stats
        )
        ep_elapsed = time.time() - ep_t0
        timer.record(ep_elapsed, n_guesses, solved)

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

    total_elapsed = time.time() - t0

    print("─" * 60)
    final_wr  = float(np.mean(wins[-WINDOW:]))
    solved_gs = [g for g, w in zip(guess_log, wins) if w]
    avg_g     = float(np.mean(solved_gs[-WINDOW:])) if solved_gs else float("nan")
    print(f"[Macedonian] Done. Win rate: {final_wr*100:.1f}%  Avg guesses: {avg_g:.2f}")

    if stats is not None:
        stats.print_summary(total_episodes=episodes, elapsed=total_elapsed)
    timer.print_summary()

    return agent_models, moderator, stats, timer


# ── Interactive Demo with Debate Printing ──────────────────────────────────────

def demo_game(agent_models, moderator, secret_word=None, stats=None):
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

    game_t0 = time.time()

    for turn in range(MAX_TURNS):
        print(f"\n--- Обид {turn+1} (Преостанати кандидати: {len(cands)}) ---")

        a_state   = mk_agent_state(cands, turn, absent, known_green, yellows)

        proposals = []
        for ai, ag in enumerate(agent_models):
            if ai == PROBABILIST or len(cands) <= 25:
                valid_idx = cands
            else:
                valid_idx = None

            guess_idx, _ = ag.sample(a_state, valid_indices=valid_idx, rng=rng)
            proposals.append(guess_idx)

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

        if stats is not None:
            stats.record_turn(proposals, secret, choice, final, turn, solved)

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
            elapsed = time.time() - game_t0
            print(f"\n  ✓ Успех! Зборот е погоден за {turn+1} обид/и! ({elapsed*1000:.1f} ms)")
            print(f"  Winning guess proposed & played by: {AGENT_NAMES[choice]}")
            return True

    elapsed = time.time() - game_t0
    print(f"\n  ✗ Неуспех. Зборот беше {WORDS[secret]} ({elapsed*1000:.1f} ms)")
    return False


def play_many_demo_games(agent_models, moderator, n_games, secret_word=None):
    """
    Play n_games demo games back-to-back, collecting AgentStats + GameTimer
    across all of them, then print a combined ranked summary at the end.
    """
    stats = AgentStats(AGENT_NAMES)
    timer = GameTimer()

    for g in range(n_games):
        rng = np.random.defaultrng()
        secret = WORD_IDX.get(secret_word, int(rng.integers(N))) if secret_word else int(rng.integers(N))
        t0 = time.time()
        solved, n_guesses, _, _ = run_episode(agent_models, moderator, rng, stats=stats)
        elapsed = time.time() - t0
        timer.record(elapsed, n_guesses, solved)
        print(f"  Game {g+1}/{n_games}: {'WIN' if solved else 'LOSS'} "
              f"in {n_guesses} guess(es), {elapsed*1000:.1f} ms")

    stats.print_summary(total_episodes=n_games, elapsed=sum(timer.episode_times))
    timer.print_summary()
    return stats, timer


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
    parser.add_argument("--stats-only", action="store_true",
                         help="Skip printed debate text; just play --games games "
                              "and print the ranked agent leaderboard + timing stats.")
    args = parser.parse_args()

    agent_models, moderator = load_weights()
    if agent_models is None:
        agent_models, moderator, _, _ = train()
        save_weights(agent_models, moderator)

    if args.stats_only:
        print(f"\n[Macedonian] Playing {args.games} games for stats only...")
        play_many_demo_games(agent_models, moderator, args.games, args.word)
    else:
        print(f"\n[Macedonian] Demonstrating multi-agent debate and negotiation...")
        game_stats = AgentStats(AGENT_NAMES)
        for _ in range(args.games):
            demo_game(agent_models, moderator, args.word, stats=game_stats)
        if args.games > 1:
            game_stats.print_summary(total_episodes=args.games)