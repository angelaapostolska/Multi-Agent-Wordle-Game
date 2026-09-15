"""
scenario_english.py — Scenario 1: Standard English Wordle

Three agents (Eliminator, Probabilist, RiskTaker) collaborate to guess
a 5-letter English word in 6 tries. A moderator picks which agent's
suggestion to actually play each turn.

To run training:
    python scenario_english.py

To watch a single game after training:
    python scenario_english.py --demo

To watch a game with an LLM-generated debate (requires a local Ollama
server — see the "LLM Debate" section in README.md):
    python scenario_english.py --demo --llm
    python scenario_english.py --demo --llm-moderator   # LLM can also override the pick
"""

import numpy as np
import json
import os
import re
import time
import argparse
import urllib.request

from wordle_env_base import (
    WORDS_EN, MAX_TURNS, AGENT_NAMES,
    ELIMINATOR, PROBABILIST, RISKTAKER,
    GREEN, YELLOW, GREY,
    build_pattern_matrix, filter_candidates,
    build_agent_state, build_mod_state,
    task_reward, agent_reward,
    expected_remaining_frac, partition_quality,
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

# ── Ollama config (LLM debate — demo/eval only, never used in train()) ─────────
# Override with env vars, or --ollama-model / --ollama-host on the CLI.
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 20  # seconds

AGENT_PERSONA = {
    ELIMINATOR:  "the Eliminator, who cares most about ruling out wrong words fast",
    PROBABILIST: "the Probabilist, who cares most about guessing common, likely real words",
    RISKTAKER:   "the RiskTaker, who cares most about splitting the remaining candidates "
                 "as evenly as possible to gather information",
}


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

        proposals = []
        for ai, ag in enumerate(agents):
            val_idx = cands if (ai == PROBABILIST or len(cands) <= 25) else None
            guess_idx, _ = ag.sample(a_state, valid_indices=val_idx, rng=rng)
            proposals.append(guess_idx)

        m_state = build_mod_state(proposals, cands, turn, WORDS, PATTERN)

        if train_mode and rng.random() < 0.20:
            choice = int(rng.integers(3))
        else:
            choice, _ = moderator.sample(m_state, rng=rng)

        final_guess    = proposals[choice]

        fb      = PATTERN[final_guess, secret]
        last    = (turn == MAX_TURNS - 1)
        solved  = (final_guess == secret)

        new_cands  = filter_candidates(cands, final_guess, fb, PATTERN)
        next_cands = new_cands if new_cands else cands

        tr = task_reward(
            fb, solved, last,
            guess_word=WORDS[final_guess],
            secret_word=WORDS[secret],
            which=choice,
            cands_before=len(cands),
            cands_after=len(next_cands),
        )

        if len(cands) > 20:
            elim_powers = [1.0 - expected_remaining_frac(p, cands, PATTERN) for p in proposals]
            best_agent  = int(np.argmax(elim_powers))
            tr += 0.6 if choice == best_agent else -0.2

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
                absent[ord(ch) - 97] = 1.0

        cands = next_cands

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


# ── LLM debate (Ollama) — used only by demo_game(), never by train() ───────────
#
# IMPORTANT: this whole block only ever runs inside demo_game(), which is
# called a handful of times (--games N). It is never called from run_episode()
# or train(), so turning it on does not add a single extra call to the
# 15,000-episode training loop — the trained agents/moderator are completely
# unaffected either way.

def _ollama_generate(prompt, warn=True):
    """
    Send a prompt to a local Ollama server and return the model's reply text,
    or None on any failure (server not running, model not pulled, timeout...).
    Callers fall back to the deterministic template text when this returns None,
    so a missing/slow Ollama install never crashes the demo.

    warn : print the failure reason (set False by compare_llm_moderator's
    quiet games so a flaky Ollama server doesn't spam dozens of identical
    lines across many games).
    """
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


def generate_agent_argument(agent_id, guess_idx, cands):
    """Fixed, deterministic argument text — no LLM involved. Used as the
    fallback when --llm is on but Ollama can't be reached, and as the
    default debate text when --llm isn't passed at all."""
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


def generate_agent_argument_llm(agent_id, guess_idx, cands, prior_arguments, warn=True):
    """
    Ask the local Ollama model to argue, in character, for this agent's
    already-chosen word — optionally responding to what the other two
    agents already argued this same turn (prior_arguments: list of
    (agent_id, text) tuples, in the order they spoke).

    The agent's WORD is never decided by the LLM here — that's still the
    trained policy network's job. The LLM only puts the case into words.
    """
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


def llm_moderator_vote(proposals, arguments, cands, turn, warn=True):
    """
    Ask the local Ollama model to pick which of the 3 proposed words the
    team should actually play, given each agent's word and argument.

    Returns an index into `proposals`, or None if Ollama is unreachable or
    its reply can't be parsed — callers should keep the trained moderator's
    pick in that case.
    """
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


# ── Demo: watch one game ───────────────────────────────────────────────────────

def demo_game(agents, moderator, secret_word=None, use_llm=False, llm_moderator=False,
              rng=None, verbose=True, override_log=None):
    """
    Play one game and print the board so you can see what happened.

    use_llm       : generate each agent's debate argument with a local Ollama
                    model instead of the fixed template (auto-falls back to
                    the template if Ollama isn't reachable).
    llm_moderator : also ask the Ollama model to vote on the three proposals,
                    and play its pick instead of the trained moderator's when
                    they disagree (implies use_llm). The trained moderator
                    network still runs every turn regardless — this only
                    decides which of the two picks gets played and printed.
    rng           : pass a seeded np.random.default_rng(...) to make this game
                    reproducible (same secret + same agent proposals every
                    time). Defaults to a fresh, unseeded generator.
    verbose       : set False to suppress all printing — used by
                    compare_llm_moderator() to run many games without
                    flooding the terminal with per-turn debate text.
    override_log  : optional list; when llm_moderator is on, one dict per
                    turn ({"trained_choice", "llm_choice"}) is appended to it
                    so callers can compute agreement/override stats afterward.
    """
    if rng is None:
        rng = np.random.default_rng()

    if secret_word is not None:
        secret = WORD_IDX[secret_word.lower()]
    else:
        secret = int(rng.integers(N))

    cands       = list(range(N))
    absent      = np.zeros(26)
    known_green = [None] * 5
    yellows     = set()

    use_llm = use_llm or llm_moderator  # --llm-moderator implies --llm

    if verbose:
        print(f"\n{'='*45}")
        print(f"  ENGLISH WORDLE DEMO")
        print(f"  Secret word: {WORDS[secret].upper()}")
        if use_llm:
            tag = f"Ollama ({OLLAMA_MODEL})" + (" + moderator override" if llm_moderator else "")
            print(f"  LLM debate: {tag}")
        print(f"{'='*45}")
        print(f"  {'AGENT':<12} {'GUESS':<8} {'RESULT':<8} {'CANDS'}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*6}")

    for turn in range(MAX_TURNS):
        a_state   = build_agent_state(cands, turn, absent, known_green, yellows, N)
        proposals = [ag.sample(a_state, valid_indices=cands, rng=rng)[0]
                     for ag in agents]

        # ── Debate round ──
        arguments = []
        if use_llm:
            if verbose:
                print(f"\n  --- Turn {turn+1} debate ({len(cands)} candidates left) ---")
            for ai in range(3):
                arg = generate_agent_argument_llm(
                    ai, proposals[ai], cands, prior_arguments=list(enumerate(arguments)),
                    warn=verbose,
                )
                arguments.append(arg)
                if verbose:
                    print(f"    [{AGENT_NAMES[ai]}] {WORDS[proposals[ai]].upper()}: {arg}")
        else:
            arguments = [generate_agent_argument(ai, proposals[ai], cands) for ai in range(3)]

        # ── Moderator: trained network always runs; LLM can override it ──
        m_state        = build_mod_state(proposals, cands, turn, WORDS, PATTERN)
        trained_choice, _ = moderator.sample(m_state, rng=rng)
        choice = trained_choice

        if llm_moderator:
            llm_choice = llm_moderator_vote(proposals, arguments, cands, turn, warn=verbose)
            if override_log is not None:
                override_log.append({"trained_choice": trained_choice, "llm_choice": llm_choice})
            if llm_choice is not None and llm_choice != trained_choice:
                if verbose:
                    print(f"    [LLM moderator] would play {AGENT_NAMES[llm_choice]}'s "
                          f"'{WORDS[proposals[llm_choice]].upper()}' instead of the trained "
                          f"moderator's {AGENT_NAMES[trained_choice]} pick — overriding.")
                choice = llm_choice
            elif llm_choice is not None and verbose:
                print(f"    [LLM moderator] agrees with the trained pick: {AGENT_NAMES[choice]}.")

        final = proposals[choice]

        fb      = PATTERN[final, secret]
        fb_str  = fb_to_str(fb)
        solved  = (final == secret)

        if verbose:
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
            if verbose:
                print(f"\n  ✓ Solved in {turn+1} guess{'es' if turn > 0 else ''}!")
            return True

    if verbose:
        print(f"\n  ✗ Failed! The word was {WORDS[secret].upper()}")
    return False


# ── Compare: LLM moderator vs trained moderator, over many games ───────────────

def compare_llm_moderator(agents, moderator, n_games=20, base_seed=123, secret_word=None):
    """
    Run n_games with the LLM moderator active, then the same n_games with
    only the trained moderator — using a matched RNG seed per game index
    across both runs, so game i sees the EXACT same secret word and the
    EXACT same agent proposals in both conditions. The only thing that can
    differ between the two runs of game i is whether the LLM's override
    actually gets applied, which isolates its effect instead of mixing it
    in with ordinary run-to-run randomness.

    Prints one summary at the end instead of per-turn debate text (which
    would be unreadable across many games) — see demo_game(verbose=...).

    NOTE: this still makes real Ollama calls for every turn of every LLM
    game (3 arguments + 1 moderator vote each) — it's bounded by n_games,
    same as demo_game(), just multiplied by it, so a large n_games will
    take a while.
    """
    print(f"\n[English] Comparing LLM moderator vs trained moderator over {n_games} games each...")
    print("(this makes real Ollama calls for the LLM-moderator half — expect it to take a while)")

    t0 = time.time()
    override_log = []
    llm_wins = 0
    for i in range(n_games):
        rng = np.random.default_rng(base_seed + i)
        won = demo_game(agents, moderator, secret_word,
                         use_llm=True, llm_moderator=True,
                         rng=rng, verbose=False, override_log=override_log)
        llm_wins += int(won)
        print(f"  [LLM-moderator {i+1:>3}/{n_games}] {'won ' if won else 'lost'} "
              f"| {time.time()-t0:6.1f}s elapsed")

    print(f"  LLM-moderator half done in {time.time()-t0:.1f}s. "
          f"Running the {n_games}-game baseline (no Ollama calls, should be quick)...")

    t1 = time.time()
    baseline_wins = 0
    for i in range(n_games):
        rng = np.random.default_rng(base_seed + i)  # same seed -> same secret + same proposals
        won = demo_game(agents, moderator, secret_word,
                         use_llm=False, llm_moderator=False,
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
    parser.add_argument("--llm", action="store_true",
                         help="Generate each turn's debate arguments with a local "
                              "Ollama model instead of fixed templates. Demo-only — "
                              "never used during training or evaluate_agents().")
    parser.add_argument("--llm-moderator", action="store_true",
                         help="Also let the Ollama model vote on the three proposals "
                              "and override the trained moderator's pick when they "
                              "disagree (implies --llm). Demo-only.")
    parser.add_argument("--ollama-model", type=str, default=None,
                         help=f"Ollama model name (default: {OLLAMA_MODEL}, or $OLLAMA_MODEL)")
    parser.add_argument("--ollama-host", type=str, default=None,
                         help=f"Ollama server URL (default: {OLLAMA_HOST}, or $OLLAMA_HOST)")
    parser.add_argument("--compare-llm", action="store_true",
                         help="Instead of the normal per-turn demo, run --games games "
                              "with the LLM moderator active and the same number with "
                              "only the trained moderator (matched RNG seeds -> same "
                              "secrets/proposals in both), then print one agreement-rate "
                              "/ win-rate summary. Ignores --llm/--llm-moderator. Makes "
                              "real Ollama calls, so large --games will take a while.")
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
        compare_llm_moderator(agents, moderator, n_games=args.games,
                               base_seed=args.seed, secret_word=args.word)
    else:
        print(f"\n[English] Running {args.games} demo game(s)...")
        wins = sum(
            demo_game(agents, moderator, args.word,
                      use_llm=args.llm, llm_moderator=args.llm_moderator)
            for _ in range(args.games)
        )
        print(f"\nResult: {wins}/{args.games} games won")