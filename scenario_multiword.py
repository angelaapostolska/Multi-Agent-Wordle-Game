import time
import numpy as np
from wordle_env_base import (
    WORDS_EN, AGENT_NAMES, ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, build_pattern_matrix, filter_candidates,
    task_reward, agent_reward, Policy,
    expected_remaining_frac, partition_quality
)

WORDS = WORDS_EN
N = len(WORDS)
PATTERN = build_pattern_matrix(WORDS)

_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR = _prior / _prior.max()

DEFAULT_NUM_WORDS = 2
DEFAULT_MAX_TURNS = 8

SINGLE_WORD_DIM = N + DEFAULT_MAX_TURNS + 26 + (5 * 26) + 26
AGENT_STATE_DIM = (SINGLE_WORD_DIM * DEFAULT_NUM_WORDS) + DEFAULT_NUM_WORDS
MOD_STATE_DIM   = (3 * 7 + 2) * DEFAULT_NUM_WORDS


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


def generate_agent_argument(agent_id, guess_idx, cands_list):
    """Generates text argument explaining proposed word selection."""
    word = WORDS[guess_idx]
    active_cands = set()
    for cands in cands_list:
        active_cands.update(cands)
    active_cands = list(active_cands) if active_cands else list(range(N))

    if agent_id == ELIMINATOR:
        frac = expected_remaining_frac(guess_idx, active_cands, PATTERN)
        elim_pct = int(100 * (1.0 - frac))
        return f"Eliminator: Proposing '{word}'. Expected candidate removal: {elim_pct}%."

    elif agent_id == PROBABILIST:
        commonness = int(PRIOR[guess_idx] * 100)
        in_play = "Yes" if guess_idx in active_cands else "No"
        return f"Probabilist: Proposing '{word}'. Frequency percentile: {commonness}%, Active candidate: {in_play}."

    else:  # RISKTAKER
        pq = round(partition_quality(guess_idx, active_cands, PATTERN), 2)
        return f"RiskTaker: Proposing '{word}'. Partition entropy score: {pq}."


def run_episode(
    agent_models, moderator, rng, train_mode=True,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Runs a single episode of multi-word Wordle with multi-agent debate."""
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

        a_state = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list, solved_list,
            num_words, max_turns
        )

        # Union of active candidates across unsolved boards
        valid = set()
        for i in range(num_words):
            if not solved_list[i]:
                valid.update(cands_list[i])
        valid = list(valid) if valid else list(range(N))

        # Agent proposal sampling
        proposals = [ag.sample(a_state, valid_indices=valid, rng=rng)[0] for ag in agent_models]

        # Generate debate arguments
        arguments = [generate_agent_argument(ai, proposals[ai], cands_list) for ai in range(3)]

        # Moderator decision
        m_state   = multiword_mod_state(proposals, cands_list, turn, num_words, max_turns)
        choice, _ = moderator.sample(m_state, rng=rng)
        final_guess = proposals[choice]

        # Update candidate space and feedback across active boards
        fb_list = []
        for i in range(num_words):
            if solved_list[i]:
                fb_list.append((GREEN, GREEN, GREEN, GREEN, GREEN))
            else:
                fb = PATTERN[final_guess, secrets[i]]
                fb_list.append(fb)

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

        all_solved = all(solved_list)
        last = (turn == max_turns - 1)

        # Calculate environment task reward across boards
        trs = [
            task_reward(fb_list[i], all_solved, last, guess_word=WORDS[final_guess], secret_word=WORDS[secrets[i]], which=choice)
            for i in range(num_words)
        ]
        tr = float(np.mean(trs))

        # Strategic reward shaping for early game candidate pruning
        active_count = sum(len(c) for i, c in enumerate(cands_list) if not solved_list[i])
        if active_count > 100:
            if choice == ELIMINATOR:
                tr += 0.08
            elif choice == RISKTAKER:
                tr += 0.04

        mod_mem.append((m_state, choice, tr))

        for ai in range(3):
            ars = [agent_reward(ai, proposals[ai], cands_list[i], secrets[i], WORDS, PATTERN) for i in range(num_words)]
            ar = float(np.mean(ars))
            agent_mem.append((ai, a_state, proposals[ai], ar))

        if all_solved:
            break

    return all(solved_list), n_guesses, agent_mem, mod_mem


def train_multiword(
    episodes=100, lr=0.003, gamma=0.97, beta=0.99, grad_clip=5.0, seed=42
):
    """Executes Policy Gradient (REINFORCE) training for multi-word Wordle agent and moderator network."""
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

    for ep in range(1, episodes + 1):
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agent_models, moderator, rng, train_mode=True
        )

        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else DEFAULT_MAX_TURNS)

        # Moderator Policy Gradient Step
        G, returns = 0.0, []
        for (_, _, r) in reversed(mod_mem):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        for (s, c, _), Gt in zip(mod_mem, returns):
            base_mod = beta * base_mod + (1 - beta) * Gt
            moderator.pg_step(s, c, Gt - base_mod, grad_clip=grad_clip)

        # Agent Policy Gradient Steps
        for (which, s, a, r) in agent_mem:
            base_agent[which] = beta * base_agent[which] + (1 - beta) * r
            agent_models[which].pg_step(s, a, r - base_agent[which], grad_clip=grad_clip)

        if ep % 20 == 0 or ep == episodes:
            recent_win = np.mean(wins[-20:])
            recent_g = np.mean([g for g, w in zip(guess_log[-20:], wins[-20:]) if w])
            print(f"Episode {ep:3d}/{episodes} | Win Rate (last 20): {recent_win*100:5.1f}% | Avg Guesses: {recent_g:.2f}")

    tot_time = time.time() - t0
    final_wr = np.mean(wins)
    avg_guesses = np.mean([g for g, w in zip(guess_log, wins) if w])

    print(f"\nTraining completed in {tot_time:.2f}s")
    print(f"Overall Win Rate: {final_wr*100:.1f}% | Avg Guesses: {avg_guesses:.2f}")

    return agent_models, moderator


if __name__ == "__main__":
    agents, mod = train_multiword(episodes=100)