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
import json
import re
import time
import argparse
import urllib.request
import numpy as np

from wordle_env_base import (
    WORDS_EN, AGENT_NAMES, ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, GREY, build_pattern_matrix, filter_candidates,
    task_reward, agent_reward, expected_remaining_frac, partition_quality,
    fb_to_str, Policy
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

# Training hyperparameters matching scenario_english.py
EPISODES  = 6000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99        # Baseline smoothing factor
GRAD_CLIP = 5.0
LOG_EVERY = 1000
WINDOW    = 500
SEED      = 42

# ── Ollama config & Agent Personas ─────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 20  # seconds

AGENT_PERSONA = {
    ELIMINATOR:  "the Eliminator, who cares most about ruling out wrong words fast",
    PROBABILIST: "the Probabilist, who cares most about guessing common, likely real words",
    RISKTAKER:   "the RiskTaker, who cares most about splitting the remaining candidates "
                 "as evenly as possible to gather information",
}


# ── LLM Helpers ───────────────────────────────────────────────────────────────

def _ollama_generate(prompt, warn=True):
    """Send a prompt to a local Ollama server and return the model's reply text."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "").strip()
        return text or None
    except Exception as e:
        if warn:
            print(f"    [Ollama unavailable, using template text — {e}]")
        return None


def generate_agent_argument(agent_id, guess_idx, cands_list):
    """Deterministic argument text fallback."""
    if isinstance(cands_list, list) and cands_list and isinstance(cands_list[0], list):
        active = set()
        for cl in cands_list:
            active.update(cl)
        cands = list(active) if active else list(range(N))
    else:
        cands = cands_list

    word = WORDS[guess_idx]

    if agent_id == ELIMINATOR:
        frac     = expected_remaining_frac(guess_idx, cands, PATTERN)
        elim_pct = int(100 * (1.0 - frac))
        return f"I propose '{word}'. It should eliminate about {elim_pct}% of the remaining candidates."

    elif agent_id == PROBABILIST:
        commonness = int(PRIOR[guess_idx] * 100)
        in_play    = "yes" if guess_idx in cands else "no"
        return f"I propose '{word}'. Frequency rank: {commonness}%, still a valid candidate: {in_play}."

    else:  # RISKTAKER
        pq = round(partition_quality(guess_idx, cands, PATTERN), 2)
        return f"I propose '{word}'. Partition quality (Gini) score: {pq}."


def generate_agent_argument_llm(agent_id, guess_idx, cands_list, prior_arguments, warn=True):
    """Generate in-character debate arguments via local Ollama LLM."""
    if isinstance(cands_list, list) and cands_list and isinstance(cands_list[0], list):
        active = set()
        for cl in cands_list:
            active.update(cl)
        cands = list(active) if active else list(range(N))
    else:
        cands = cands_list

    word  = WORDS[guess_idx]
    stats = {
        "elimination_%":        int(100 * (1.0 - expected_remaining_frac(guess_idx, cands, PATTERN))),
        "partition_quality":    round(partition_quality(guess_idx, cands, PATTERN), 2),
        "still_a_candidate":    guess_idx in cands,
        "frequency_rank_%":     int(PRIOR[guess_idx] * 100),
        "candidates_remaining": len(cands),
    }

    context = ""
    if prior_arguments:
        context = "So far in this debate:\n" + "\n".join(
            f"- {AGENT_NAMES[i]}: {text}" for i, text in prior_arguments
        ) + "\n\n"

    prompt = (
        f"You are playing Wordle as {AGENT_PERSONA[agent_id]}. "
        f"Your strategy already picked the word '{word.upper()}' as this turn's guess. "
        f"Stats for this word: {stats}. {context}"
        f"In 1-2 short sentences, argue in character for why '{word.upper()}' is a good "
        f"guess right now"
        + (", briefly reacting to what the other agents said" if prior_arguments else "")
        + ". Do not propose a different word — only argue for this one."
    )

    text = _ollama_generate(prompt, warn=warn)
    if text is None:
        return generate_agent_argument(agent_id, guess_idx, cands)
    return text.replace("\n", " ").strip()


def llm_moderator_vote(proposals, arguments, cands_list, turn, warn=True):
    """Let Ollama LLM vote on which agent proposal to play."""
    if isinstance(cands_list, list) and cands_list and isinstance(cands_list[0], list):
        active = set()
        for cl in cands_list:
            active.update(cl)
        cands = list(active) if active else list(range(N))
    else:
        cands = cands_list

    lines = [
        f"{i}: {AGENT_NAMES[i]} proposes '{WORDS[g].upper()}' — {arguments[i]}"
        for i, g in enumerate(proposals)
    ]
    prompt = (
        f"It's turn {turn + 1} of a Wordle game with {len(cands)} candidate words left. "
        f"Three teammates each propose a guess:\n" + "\n".join(lines) +
        "\n\nWhich proposal should the team actually play? "
        "Reply with ONLY the number 0, 1, or 2 — nothing else."
    )
    text = _ollama_generate(prompt, warn=warn)
    if text is None:
        return None
    match = re.search(r"[0-2]", text)
    return int(match.group()) if match else None


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
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Runs a single episode of Multi-Word Wordle."""
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

        valid = set()
        for i in range(num_words):
            if not solved_list[i]:
                valid.update(cands_list[i])
        valid = list(valid) if valid else list(range(N))

        proposals = []
        for ai, ag in enumerate(agents):
            guess_idx, _ = ag.sample(a_state, valid_indices=valid, rng=rng)
            proposals.append(guess_idx)

        m_state = multiword_mod_state(proposals, cands_list, turn, num_words, max_turns)

        if train_mode and rng.random() < 0.20:
            choice = int(rng.integers(3))
        else:
            choice, _ = moderator.sample(m_state, rng=rng)

        final_guess = proposals[choice]

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

        trs = [
            task_reward(fb_list[i], all_solved, last, guess_word=WORDS[final_guess], secret_word=WORDS[secrets[i]], which=choice)
            for i in range(num_words)
        ]
        tr = float(np.mean(trs))

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


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(episodes=EPISODES, lr=LR, seed=SEED):
    """Train multi-word Wordle policy networks using REINFORCE policy gradient."""
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
    t0 = time.time()

    print(f"\n[MultiWord] Starting training: {episodes} episodes, lr={lr}")
    print("─" * 60)

    for ep in range(episodes):
        solved, n_guesses, agent_mem, mod_mem = run_episode(
            agents, moderator, rng, train_mode=True
        )

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

    return agents, moderator


# ── Demo & Evaluation Modes ───────────────────────────────────────────────────

def demo_game(
    agents, moderator, use_llm=False, llm_moderator=False,
    rng=None, verbose=True, override_log=None,
    num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS
):
    """Play one multi-word game and display board visual feedback."""
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
        a_state = multiword_agent_state(
            cands_list, turn, absent_list, known_greens, yellows_list, solved_list,
            num_words, max_turns
        )

        valid = set()
        for i in range(num_words):
            if not solved_list[i]:
                valid.update(cands_list[i])
        valid = list(valid) if valid else list(range(N))

        proposals = [ag.sample(a_state, valid_indices=valid, rng=rng)[0] for ag in agents]

        arguments = []
        if use_llm:
            if verbose:
                cand_counts = "/".join(str(len(c)) for c in cands_list)
                print(f"\n  --- Turn {turn+1} debate (cands: {cand_counts}) ---")
            for ai in range(3):
                arg = generate_agent_argument_llm(
                    ai, proposals[ai], cands_list, prior_arguments=list(enumerate(arguments)),
                    warn=verbose
                )
                arguments.append(arg)
                if verbose:
                    print(f"    [{AGENT_NAMES[ai]}] {WORDS[proposals[ai]].upper()}: {arg}")
        else:
            arguments = [generate_agent_argument(ai, proposals[ai], cands_list) for ai in range(3)]

        m_state = multiword_mod_state(proposals, cands_list, turn, num_words, max_turns)
        trained_choice, _ = moderator.sample(m_state, rng=rng)
        choice = trained_choice

        if llm_moderator:
            llm_choice = llm_moderator_vote(proposals, arguments, cands_list, turn, warn=verbose)
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


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a demo game only")
    parser.add_argument("--games", type=int, default=5, help="Number of demo games")
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
    parser.add_argument("--seed", type=int, default=123,
                        help="Base RNG seed for --compare-llm's matched game pairs")
    args = parser.parse_args()

    if args.ollama_model:
        OLLAMA_MODEL = args.ollama_model
    if args.ollama_host:
        OLLAMA_HOST = args.ollama_host

    agents, moderator = load_weights()
    if agents is None:
        agents, moderator = train()
        save_weights(agents, moderator)

    if args.compare_llm:
        compare_llm_moderator(agents, moderator, n_games=args.games, base_seed=args.seed)
    else:
        print(f"\n[MultiWord] Running {args.games} demo game(s)...")
        wins = sum(
            demo_game(agents, moderator, use_llm=args.llm, llm_moderator=args.llm_moderator)
            for _ in range(args.games)
        )
        print(f"\nResult: {wins}/{args.games} games won")