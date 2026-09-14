"""
scenario_multiword.py — Scenario 3: Alternating Agents + Scratchpad (Coordination)

Two agents — Agent A and Agent B — take ALTERNATING turns at the same
Wordle puzzle (or puzzles, if --num_words > 1): Agent A guesses on turns
1, 3, 5, ... and Agent B guesses on turns 2, 4, 6, ...

Before every guess, the active agent reads a shared, append-only
scratchpad containing every note either agent has written so far, then
writes its own hypothesis / strategy note before the next agent's turn.
This is what makes the two agents *actively coordinate* instead of just
taking turns in silence:

  - Before guessing, the active agent signals its current hypothesis
  - Agents explicitly divide the search space (vowel-focus / consonant-
    focus) instead of duplicating each other's work
  - The two agents negotiate an opening strategy before Turn 1
  - The scratchpad is append-only — entries are never overwritten

Both agents are trained JOINTLY with REINFORCE: every turn produces one
shared (team) reward, and at the end of the episode both networks are
updated from the SAME discounted-return trajectory. If the team wins,
both agents are pushed toward what they did that game — this is what
forces genuine cooperation instead of two agents optimising themselves
independently.

Research question this file is built to answer (see --compare):
    Do two agents that actively coordinate through a shared scratchpad
    outperform agents who simply alternate turns without communicating?

To run:
    python scenario_multiword.py                  # train/demo, coordination ON
    python scenario_multiword.py --no_scratchpad   # train/demo the baseline (no coordination)
    python scenario_multiword.py --compare         # train/load BOTH and compare them
    python scenario_multiword.py --demo --llm      # narrate the scratchpad with a local Ollama model
    python scenario_multiword.py --num_words 2 --max_turns 8
"""

import numpy as np
import json
import os
import re
import sys
import time
import argparse
import urllib.request

# Force UTF-8 stdout/stderr. Without this, redirecting output on Windows
# (e.g. `python scenario_multiword.py > out.txt`) falls back to the
# system's default locale codec (e.g. cp1251), which can't encode the
# arrows/checkmarks/box-drawing characters used below and crashes with a
# UnicodeEncodeError. Printing straight to a console usually works by
# luck; redirecting to a file is what exposes it.
#
# When output IS redirected to a file (not a live console), we also add
# a UTF-8 BOM. Without it, Notepad/Excel on Windows often guess the
# file's encoding as the system ANSI codepage instead of UTF-8, which
# turns "←" into mojibake like "тЖР" even though the file's bytes were
# correct all along — the BOM tells them unambiguously "this is UTF-8".
# Live console output is left as plain UTF-8 (no BOM) so a stray BOM
# character doesn't show up as a glyph in the terminal itself.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _enc = "utf-8" if _stream.isatty() else "utf-8-sig"
        _stream.reconfigure(encoding=_enc, errors="replace")

from wordle_env_base import (
    WORDS_EN as WORDS, MAX_TURNS as BASE_MAX_TURNS,
    GREEN, YELLOW, GREY,
    build_pattern_matrix, filter_candidates, build_agent_state,
    fb_to_str, Policy, letter_match_score,
    expected_remaining_frac, partition_quality, DIFF_WEIGHTS,
)

# ── Setup ────────────────────────────────────────────────────────────────────

N        = len(WORDS)
WORD_IDX = {w: i for i, w in enumerate(WORDS)}

print(f"[Multiword/Coordination] Building pattern matrix for {N} words...")
PATTERN = build_pattern_matrix(WORDS)
print("[Multiword/Coordination] Pattern matrix ready.")

_prior = np.array([N - i for i in range(N)], dtype=np.float64)
PRIOR  = _prior / _prior.max()

# ── Two agents, not three — this scenario is about coordination, not roles ────
AGENT_NAMES = ["Agent A", "Agent B"]
AGENT_A, AGENT_B = 0, 1

VOWELS = set("aeiou")

# Training config (per the Scenario 3 spec: 3,000-5,000 episodes)
EPISODES  = 4000
LR        = 0.003
GAMMA     = 0.97
BETA      = 0.99          # baseline smoothing factor
GRAD_CLIP = 5.0
LOG_EVERY = 500
WINDOW    = 300
SEED      = 42

# Default game settings
DEFAULT_NUM_WORDS = 1      # standard single-secret Wordle by default
DEFAULT_MAX_TURNS = 6

# ── Ollama config (LLM-narrated scratchpad — demo/eval only, never in train()) ─
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = 20  # seconds


# ── Multi-word-aware joint task reward ──────────────────────────────────────
#
# Both agents share this exact same reward every turn (see run_episode) —
# that's what "joint" means here: there is only one reward channel, not one
# per agent as in the debate/moderator scenarios.

def task_reward_multi(fb_list, solved_list, last_turn,
                       guess_word=None, secret_words=None, which=None):
    """
    Team reward for one turn, normalised by the number of secret words so
    the scale stays comparable whether we're solving 1 word or several.
    """
    n_words       = len(fb_list)
    total_greens  = sum(sum(1 for x in fb if x == GREEN)  for fb in fb_list)
    total_yellows = sum(sum(1 for x in fb if x == YELLOW) for fb in fb_list)

    r = (0.2 * total_greens + 0.05 * total_yellows - 0.1 * n_words) / n_words

    if guess_word is not None and secret_words is not None:
        diff_weight = DIFF_WEIGHTS.get(which, 0.2)
        remaining_scores = [
            letter_match_score(guess_word, sw)
            for sw, solved in zip(secret_words, solved_list) if not solved
        ]
        if remaining_scores:
            r += diff_weight * (sum(remaining_scores) / len(remaining_scores))

    n_solved = sum(solved_list)
    if n_solved == n_words:
        r += 5.0
    elif n_solved > 0:
        r += 5.0 * (n_solved / n_words)

    if last_turn:
        r -= 1.0 * (n_words - n_solved) / n_words

    return r


# ── State construction ──────────────────────────────────────────────────────
#
# Every agent state is built from two parts:
#   1. the per-word "board" state (candidates / greens / yellows / turn) —
#      this is visible in BOTH the coordination and the no-scratchpad
#      baseline conditions, because it comes from the game itself, not
#      from the agents talking to each other.
#   2. coordination features (only added when use_scratchpad=True) — this
#      is the part that specifically represents "reading the scratchpad":
#      whose turn it is, what the other agent guessed last, and which
#      letter-group this agent has been assigned to focus on.

SINGLE_WORD_DIM = N + BASE_MAX_TURNS + 26 + 5 * 26 + 26
COORD_DIM       = 2 + N + 26   # active-agent one-hot + partner's last guess + division-of-labor mask


def base_state_dim(num_words):
    return SINGLE_WORD_DIM * num_words + num_words


def agent_state_dim(num_words, use_scratchpad):
    return base_state_dim(num_words) + (COORD_DIM if use_scratchpad else 0)


def division_mask(agent_id):
    """
    The letter-group each agent has agreed to focus on, e.g. "You focus on
    vowel patterns, I'll handle consonants." This is only ever fed to the
    network as a SOFT hint (never used to hard-restrict which words an
    agent is allowed to guess) — hard-restricting would risk making the
    secret word literally unreachable for whichever agent's "half" doesn't
    contain it.
    """
    mask = np.zeros(26)
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        is_vowel = ch in VOWELS
        if (agent_id == AGENT_A) == is_vowel:
            mask[i] = 1.0
    return mask


def word_slot_state(cands_list, turn, absent_list, known_greens, yellows_list,
                     solved_list, num_words):
    """The shared, scratchpad-independent part of the state."""
    parts = [
        build_agent_state(cands_list[i], turn, absent_list[i],
                           known_greens[i], yellows_list[i], N)
        for i in range(num_words)
    ]
    solved_flags = np.array([1.0 if s else 0.0 for s in solved_list])
    return np.concatenate(parts + [solved_flags])


def coord_state(active_agent, partner_last_guess):
    """The scratchpad-derived part of the state — omitted entirely in the
    no-scratchpad baseline condition."""
    active_onehot = np.zeros(2)
    active_onehot[active_agent] = 1.0

    partner_vec = np.zeros(N)
    if partner_last_guess is not None:
        partner_vec[partner_last_guess] = 1.0

    return np.concatenate([active_onehot, partner_vec, division_mask(active_agent)])


def build_state(cands_list, turn, absent_list, known_greens, yellows_list,
                 solved_list, num_words, use_scratchpad, active_agent=None,
                 partner_last_guess=None):
    base = word_slot_state(cands_list, turn, absent_list, known_greens,
                            yellows_list, solved_list, num_words)
    if not use_scratchpad:
        return base
    return np.concatenate([base, coord_state(active_agent, partner_last_guess)])


# ── Opening negotiation ──────────────────────────────────────────────────────
#
# "Agents can agree on an opening strategy before Turn 1": both agents
# independently propose an opening word from their own (identical, since
# nothing has been guessed yet) policy networks, and whichever proposal
# eliminates more candidates on average across all word slots is the one
# actually played. Whichever agent proposed the winning word gets the
# training credit for that turn (it really was sampled from that agent's
# own distribution, so the REINFORCE update stays valid/on-policy).

def negotiate_opening(agents, state0, cands_list, num_words, rng):
    proposals = [ag.sample(state0, valid_indices=list(range(N)), rng=rng)[0]
                 for ag in agents]
    scores = []
    for g in proposals:
        elim = np.mean([1.0 - expected_remaining_frac(g, cands_list[i], PATTERN)
                         for i in range(num_words)])
        scores.append(elim)
    winner = int(np.argmax(scores))
    return winner, proposals[winner], proposals


# ── Single episode (training — fast, no text, no printing) ─────────────────

def run_episode(agents, rng, num_words=DEFAULT_NUM_WORDS,
                 max_turns=DEFAULT_MAX_TURNS, use_scratchpad=True):
    """
    Play one game of alternating-turn Wordle.

    Returns:
        solved_all : True if every secret word got guessed
        n_guesses  : how many turns were used
        mem        : list of (agent_id, state, action, reward) — one entry
                     per turn, ready for the joint REINFORCE update.
    """
    secrets = list(rng.choice(N, size=num_words, replace=False))

    cands_list   = [list(range(N)) for _ in range(num_words)]
    absent_list  = [np.zeros(26)   for _ in range(num_words)]
    known_greens = [[None] * 5     for _ in range(num_words)]
    yellows_list = [set()          for _ in range(num_words)]
    solved_list  = [False]         * num_words

    mem            = []
    last_guess_idx = None
    n_guesses      = 0

    for turn in range(max_turns):
        n_guesses += 1
        active = turn % 2  # Agent A on turns 0,2,4.. ; Agent B on 1,3,5..

        state = build_state(cands_list, turn, absent_list, known_greens,
                             yellows_list, solved_list, num_words,
                             use_scratchpad, active_agent=active,
                             partner_last_guess=last_guess_idx)

        if turn == 0 and use_scratchpad:
            credited_agent, final, _ = negotiate_opening(
                agents, state, cands_list, num_words, rng
            )
        else:
            valid = set()
            for i in range(num_words):
                if not solved_list[i]:
                    valid.update(cands_list[i])
            valid = list(valid) if valid else list(range(N))
            final, _ = agents[active].sample(state, valid_indices=valid, rng=rng)
            credited_agent = active

        fb_list = []
        for i in range(num_words):
            if solved_list[i]:
                fb_list.append(tuple([GREEN] * 5))
                continue
            fb = PATTERN[final, secrets[i]]
            fb_list.append(fb)
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
            if final == secrets[i]:
                solved_list[i] = True

        last = (turn == max_turns - 1)
        r = task_reward_multi(
            fb_list, solved_list, last,
            guess_word=WORDS[final],
            secret_words=[WORDS[s] for s in secrets],
            which=credited_agent,
        )
        mem.append((credited_agent, state, final, r))
        last_guess_idx = final

        if all(solved_list):
            break

    return all(solved_list), n_guesses, mem


# ── Joint training loop (REINFORCE, joint update at end of episode) ─────────

def train(episodes=EPISODES, lr=LR, num_words=DEFAULT_NUM_WORDS,
          max_turns=DEFAULT_MAX_TURNS, use_scratchpad=True, seed=SEED):
    rng = np.random.default_rng(seed)

    dim    = agent_state_dim(num_words, use_scratchpad)
    agents = [
        Policy(dim, 64, N, lr, seed=1),   # Agent A
        Policy(dim, 64, N, lr, seed=2),   # Agent B
    ]
    base_agent = [0.0, 0.0]
    wins, guess_log = [], []
    t0 = time.time()

    tag = "WITH scratchpad coordination" if use_scratchpad else "WITHOUT scratchpad (baseline)"
    print(f"\n[Multiword/Coordination] Training {tag}: {episodes} ep, "
          f"{num_words} word(s), {max_turns} turns, lr={lr}")
    print("─" * 60)

    for ep in range(episodes):
        solved, n_guesses, mem = run_episode(
            agents, rng, num_words=num_words, max_turns=max_turns,
            use_scratchpad=use_scratchpad,
        )
        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else max_turns)

        # ── Joint REINFORCE update: ONE discounted-return trajectory,
        # shared by both agents, applied to whichever network acted at
        # each step. This is what "both agents share one reward signal"
        # and "joint update at end of episode" mean in practice.
        G, returns = 0.0, []
        for (_, _, _, r) in reversed(mem):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()

        for (aid, s, a, _), Gt in zip(mem, returns):
            base_agent[aid] = BETA * base_agent[aid] + (1 - BETA) * Gt
            agents[aid].pg_step(s, a, Gt - base_agent[aid], grad_clip=GRAD_CLIP)

        if (ep + 1) % LOG_EVERY == 0:
            win_rate  = np.mean(wins[-WINDOW:])
            solved_gs = [g for g, w in zip(guess_log[-WINDOW:], wins[-WINDOW:]) if w]
            avg_g     = np.mean(solved_gs) if solved_gs else float("nan")
            print(f"  ep {ep+1:6d} | all-solved% {win_rate*100:5.1f} | "
                  f"avg guesses {avg_g:.2f} | {time.time()-t0:.1f}s")

    print("─" * 60)
    final_wr  = float(np.mean(wins[-WINDOW:]))
    solved_gs = [g for g, w in zip(guess_log, wins) if w]
    avg_g     = float(np.mean(solved_gs[-WINDOW:])) if solved_gs else float("nan")
    print(f"[Multiword/Coordination] Done ({tag}). "
          f"All-solved rate: {final_wr*100:.1f}%  Avg guesses: {avg_g:.2f}")

    return agents


# ── Scratchpad narration (template + optional Ollama LLM) ──────────────────
#
# IMPORTANT: exactly like the English scenario's LLM debate, this whole
# block is only ever called from demo_game() — never from run_episode() or
# train(). The word each agent guesses always comes from the trained
# policy network; the LLM (or the template) only puts the coordination
# into words for the transcript, and never changes which word gets played.

def _ollama_generate(prompt, warn=True):
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


def _stats_for(guess_idx, cands_list, num_words):
    elim = np.mean([1.0 - expected_remaining_frac(guess_idx, cands_list[i], PATTERN)
                     for i in range(num_words)]) if cands_list[0] else 0.0
    pq   = np.mean([partition_quality(guess_idx, cands_list[i], PATTERN)
                     for i in range(num_words)])
    return elim, pq


def generate_note(agent_id, guess_idx, cands_list, num_words, scratchpad, is_opening):
    """Fixed, deterministic scratchpad note — no LLM. Used as the default
    text, and as the fallback when --llm can't reach Ollama."""
    word      = WORDS[guess_idx].upper()
    elim, pq  = _stats_for(guess_idx, cands_list, num_words)
    focus     = "vowel patterns" if agent_id == AGENT_A else "consonant patterns"

    if is_opening:
        other       = AGENT_NAMES[1 - agent_id]
        other_focus = "consonant patterns" if agent_id == AGENT_A else "vowel patterns"
        return (f"Opening move: I'll guess {word} (~{int(elim*100)}% expected "
                f"elimination). {other}, you focus on {other_focus} once we get feedback.")

    if scratchpad:
        prev = scratchpad[-1]
        return (f"Based on your {WORDS[prev['guess_idx']].upper()} result, I'm guessing "
                f"{word}. Partition quality {pq:.2f}, focusing on {focus}.")
    return f"Guessing {word}. Partition quality {pq:.2f}, focusing on {focus}."


def generate_note_llm(agent_id, guess_idx, cands_list, num_words, scratchpad,
                       is_opening, warn=True):
    word     = WORDS[guess_idx].upper()
    elim, pq = _stats_for(guess_idx, cands_list, num_words)
    persona  = ("Agent A, who is focusing on vowel patterns" if agent_id == AGENT_A
                else "Agent B, who is focusing on consonant patterns")

    history = ""
    if scratchpad:
        history = "Shared scratchpad so far:\n" + "\n".join(
            f"- Turn {e['turn']} ({AGENT_NAMES[e['agent']]}): guessed "
            f"{WORDS[e['guess_idx']].upper()} -> {' '.join(e['feedback'])}. "
            f"Note: \"{e['note']}\""
            for e in scratchpad
        ) + "\n\n"

    prompt = (
        f"You are {persona} in a cooperative two-agent Wordle game. "
        f"{history}"
        f"Your strategy just picked '{word}' as this turn's guess "
        f"(elimination power ~{int(elim*100)}%, partition quality {pq:.2f}). "
        + ("This is the opening guess, before any feedback exists. "
           if is_opening else
           "Build on what's already in the scratchpad above. ")
        + "In 1-2 short sentences, write the note you'd append to the shared "
          "scratchpad: state your hypothesis and, if useful, suggest how you "
          "and your teammate should divide the remaining search space. "
          "Do not propose a different word — only justify this one."
    )
    text = _ollama_generate(prompt, warn=warn)
    if text is None:
        return generate_note(agent_id, guess_idx, cands_list, num_words, scratchpad, is_opening)
    return text.replace("\n", " ").strip()


# ── Demo: watch one game, with the scratchpad printed turn by turn ─────────

def demo_game(agents, num_words=DEFAULT_NUM_WORDS, max_turns=DEFAULT_MAX_TURNS,
              secret_words=None, use_scratchpad=True, use_llm=False,
              rng=None, verbose=True):
    if rng is None:
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

    scratchpad     = []   # append-only: every entry stays forever
    last_guess_idx = None

    if verbose:
        mode = "SCRATCHPAD COORDINATION" if use_scratchpad else "NO COORDINATION (baseline)"
        print(f"\n{'='*70}")
        print(f"  ALTERNATING AGENTS — {mode}")
        for i, s in enumerate(secrets):
            print(f"  Word {i+1}: {WORDS[s].upper()}")
        print(f"{'='*70}")

    for turn in range(max_turns):
        active = turn % 2
        state = build_state(cands_list, turn, absent_list, known_greens,
                             yellows_list, solved_list, num_words,
                             use_scratchpad, active_agent=active,
                             partner_last_guess=last_guess_idx)

        is_opening = (turn == 0)
        if is_opening and use_scratchpad:
            credited_agent, final, _ = negotiate_opening(
                agents, state, cands_list, num_words, rng
            )
        else:
            valid = set()
            for i in range(num_words):
                if not solved_list[i]:
                    valid.update(cands_list[i])
            valid = list(valid) if valid else list(range(N))
            final, _ = agents[active].sample(state, valid_indices=valid, rng=rng)
            credited_agent = active

        if verbose:
            print(f"\n--- Turn {turn+1} ({AGENT_NAMES[credited_agent]}"
                  f"{' — chosen by opening negotiation' if is_opening and use_scratchpad else ''}) ---")
            if use_scratchpad:
                if scratchpad:
                    print(f"  Reads scratchpad ({len(scratchpad)} note(s) so far).")
                else:
                    print("  Reads empty scratchpad.")
            print(f"  Guesses {WORDS[final].upper()}.")

        # Stats/hypothesis text reflect what the agent knew GOING INTO this
        # guess (pre-filter candidate lists) — not the already-narrowed
        # list produced by this very guess's own feedback.
        if use_scratchpad:
            if use_llm:
                note = generate_note_llm(credited_agent, final, cands_list, num_words,
                                          scratchpad, is_opening, warn=verbose)
            else:
                note = generate_note(credited_agent, final, cands_list, num_words,
                                      scratchpad, is_opening)

        fb_strs = []
        for i, secret in enumerate(secrets):
            if solved_list[i]:
                fb_strs.append("✓✓✓✓✓")
                continue
            fb = PATTERN[final, secret]
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

        if use_scratchpad:
            scratchpad.append({
                "turn": turn + 1, "agent": credited_agent,
                "guess_idx": final, "feedback": fb_strs, "note": note,
            })
            if verbose:
                print(f"  Writes: \"{note}\"")

        if verbose:
            print(f"  Feedback: {' '.join(fb_strs)}")

        if all(solved_list):
            if verbose:
                print(f"\n  ✓ All {num_words} word(s) solved in {turn+1} turn(s)!")
            return True, turn + 1

        if turn == max_turns - 1:
            if verbose:
                unsolved = [WORDS[secrets[i]].upper()
                            for i, s in enumerate(solved_list) if not s]
                print(f"\n  ✗ Failed! Unsolved: {', '.join(unsolved)}")
            return False, max_turns

    return False, max_turns


# ── Compare: scratchpad coordination vs. no-coordination baseline ──────────

def compare_scratchpad(episodes=EPISODES, num_words=DEFAULT_NUM_WORDS,
                        max_turns=DEFAULT_MAX_TURNS, n_games=200,
                        seed=SEED, lr=LR):
    """
    Trains (or loads) both conditions, then runs `n_games` fast, silent
    games per condition with MATCHED seeds — game i uses the exact same
    secret word(s) in both conditions — and prints a head-to-head summary.
    This is the direct answer to this scenario's research question.
    """
    print("\n[Multiword/Coordination] Preparing WITH-scratchpad agents...")
    agents_coord = load_weights(num_words, max_turns, True, lr) or \
        train(episodes, lr, num_words, max_turns, True, seed)
    if not weights_exist(num_words, max_turns, True):
        save_weights(agents_coord, num_words, max_turns, True)

    print("\n[Multiword/Coordination] Preparing NO-scratchpad (baseline) agents...")
    agents_base = load_weights(num_words, max_turns, False, lr) or \
        train(episodes, lr, num_words, max_turns, False, seed)
    if not weights_exist(num_words, max_turns, False):
        save_weights(agents_base, num_words, max_turns, False)

    def eval_condition(agents, use_scratchpad):
        rng = np.random.default_rng(seed + 9999)
        wins, guesses = 0, []
        for i in range(n_games):
            game_rng = np.random.default_rng(int(rng.integers(1_000_000_000)) + i)
            solved, n = demo_game(agents, num_words=num_words, max_turns=max_turns,
                                   use_scratchpad=use_scratchpad, rng=game_rng,
                                   verbose=False)
            wins += int(solved)
            if solved:
                guesses.append(n)
        win_rate = wins / n_games
        avg_g    = float(np.mean(guesses)) if guesses else float("nan")
        return win_rate, avg_g

    wr_coord, ag_coord = eval_condition(agents_coord, True)
    wr_base,  ag_base  = eval_condition(agents_base,  False)

    print("\n" + "─" * 60)
    print(f"RESEARCH QUESTION: does active scratchpad coordination help?")
    print("─" * 60)
    print(f"  WITH scratchpad    : {wr_coord*100:5.1f}% solved | avg guesses {ag_coord:.2f}")
    print(f"  WITHOUT scratchpad : {wr_base*100:5.1f}% solved | avg guesses {ag_base:.2f}")
    print("─" * 60)

    if wr_coord > 0.98 and wr_base > 0.98:
        print("Note: both conditions are already near-ceiling (>98% solved), so there's")
        print("very little headroom left for coordination to show a win-rate benefit.")
        print("For a clearer signal in your report, try a harder setup, e.g.:")
        print(f"  python {os.path.basename(__file__)} --compare --max_turns 4")
        print(f"  python {os.path.basename(__file__)} --compare --num_words 2 --max_turns 8")
        print("and/or a larger --compare-games for a tighter estimate.")

    return {
        "n_games": n_games,
        "with_scratchpad":    {"win_rate": wr_coord, "avg_guesses": ag_coord},
        "without_scratchpad": {"win_rate": wr_base,  "avg_guesses": ag_base},
    }


# ── Save / load ──────────────────────────────────────────────────────────────

def _weights_filename(num_words, max_turns, use_scratchpad):
    tag = "scratchpad" if use_scratchpad else "baseline"
    return f"weights_multiword_{tag}_{num_words}w_{max_turns}t.npz"


def weights_exist(num_words, max_turns, use_scratchpad):
    return os.path.exists(_weights_filename(num_words, max_turns, use_scratchpad))


def save_weights(agents, num_words, max_turns, use_scratchpad, filename=None):
    filename = filename or _weights_filename(num_words, max_turns, use_scratchpad)
    np.savez(filename,
        a_W1=agents[0].W1, a_b1=agents[0].b1, a_W2=agents[0].W2, a_b2=agents[0].b2,
        b_W1=agents[1].W1, b_b1=agents[1].b1, b_W2=agents[1].W2, b_b2=agents[1].b2,
    )
    print(f"Weights saved → {filename}")


def load_weights(num_words, max_turns, use_scratchpad, lr=LR, filename=None):
    filename = filename or _weights_filename(num_words, max_turns, use_scratchpad)
    if not os.path.exists(filename):
        return None
    dim    = agent_state_dim(num_words, use_scratchpad)
    d      = np.load(filename, allow_pickle=True)
    agents = [Policy(dim, 64, N, lr, seed=1), Policy(dim, 64, N, lr, seed=2)]
    for ag, px in zip(agents, ["a", "b"]):
        ag.W1 = d[f"{px}_W1"]; ag.b1 = d[f"{px}_b1"]
        ag.W2 = d[f"{px}_W2"]; ag.b2 = d[f"{px}_b2"]
    print(f"Weights loaded ← {filename}")
    return agents


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                         help="No-op — kept for consistency with the other scenario "
                              "files. This script always demos after training/loading.")
    parser.add_argument("--num_words", type=int, default=DEFAULT_NUM_WORDS)
    parser.add_argument("--max_turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--episodes",  type=int, default=EPISODES)
    parser.add_argument("--games",     type=int, default=3,
                         help="Number of demo games to show (ignored by --compare)")
    parser.add_argument("--words",     nargs="+", default=None,
                         help="Fix the secret word(s), e.g. --words crane")
    parser.add_argument("--no_scratchpad", action="store_true",
                         help="Train/demo the no-coordination baseline instead")
    parser.add_argument("--llm", action="store_true",
                         help="Narrate scratchpad notes with a local Ollama model "
                              "instead of the fixed template (demo-only).")
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--ollama-host", type=str, default=None)
    parser.add_argument("--compare", action="store_true",
                         help="Train/load BOTH conditions and print a head-to-head "
                              "win-rate / avg-guesses comparison (the research question).")
    parser.add_argument("--compare-games", type=int, default=200,
                         help="Games per condition for --compare")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.ollama_model:
        OLLAMA_MODEL = args.ollama_model
    if args.ollama_host:
        OLLAMA_HOST = args.ollama_host

    if args.compare:
        compare_scratchpad(episodes=args.episodes, num_words=args.num_words,
                            max_turns=args.max_turns, n_games=args.compare_games,
                            seed=args.seed)
    else:
        use_scratchpad = not args.no_scratchpad
        agents = load_weights(args.num_words, args.max_turns, use_scratchpad)
        if agents is None:
            agents = train(episodes=args.episodes, num_words=args.num_words,
                            max_turns=args.max_turns, use_scratchpad=use_scratchpad,
                            seed=args.seed)
            save_weights(agents, args.num_words, args.max_turns, use_scratchpad)

        print(f"\n[Multiword/Coordination] Running {args.games} demo game(s)...")
        wins = 0
        for _ in range(args.games):
            solved, _ = demo_game(agents, num_words=args.num_words,
                                   max_turns=args.max_turns, secret_words=args.words,
                                   use_scratchpad=use_scratchpad, use_llm=args.llm)
            wins += int(solved)
        print(f"\nResult: {wins}/{args.games} games won (all words solved)")