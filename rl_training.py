"""
rl_training.py — Reinforcement Learning training utilities

This file contains the REINFORCE algorithm logic that is shared
across all three scenarios. You can import from here instead of
copying training loops into each scenario file.

What is REINFORCE?
─────────────────
REINFORCE is the simplest policy gradient algorithm. Here's the idea:

  1. The agent plays a game and records (state, action, reward) for each step.
  2. After the game ends, we calculate the "return" for each step:
       G_t = reward_t + gamma * reward_{t+1} + gamma^2 * reward_{t+2} + ...
     This tells us how good the situation was from step t onwards.
  3. We update the neural network to make actions with high G_t more probable
     and actions with low G_t less probable.
  4. We subtract a "baseline" from G_t to reduce variance (make training stable).
     The baseline is a running average of recent returns.

That's it. No Q-values, no replay buffer, just: did it work? → do more of that.

Usage:
    from rl_training import Trainer

    trainer = Trainer(agents, moderator, run_episode_fn, config)
    trainer.run()
"""

import numpy as np
import time
import json


# ── Training configuration dataclass ──────────────────────────────────────────

class TrainConfig:
    """
    Holds all training hyperparameters in one place.

    episodes  : total number of games to train on
    lr        : learning rate — how big each weight update is
                (too high → unstable, too low → slow learning)
    gamma     : discount factor — how much future rewards matter
                (0 = only care about now, 1 = care equally about all future)
    beta      : baseline smoothing — how fast the baseline adapts
                (closer to 1 = slower adaptation = more stable)
    grad_clip : maximum gradient size — prevents exploding gradients
    log_every : print progress every N episodes
    window    : average win rate over the last N episodes for logging
    seed      : random seed for reproducibility
    """
    def __init__(
        self,
        episodes  = 15000,
        lr        = 0.003,
        gamma     = 0.97,
        beta      = 0.99,
        grad_clip = 5.0,
        log_every = 1000,
        window    = 500,
        seed      = 42,
        label     = "",
    ):
        self.episodes  = episodes
        self.lr        = lr
        self.gamma     = gamma
        self.beta      = beta
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.window    = window
        self.seed      = seed
        self.label     = label   # e.g. "English" or "Macedonian"


# ── Main trainer class ─────────────────────────────────────────────────────────

class Trainer:
    """
    Runs the REINFORCE training loop for any scenario.

    Parameters:
        agents        : list of 3 Policy networks (one per agent type)
        moderator     : Policy network for the moderator
        episode_fn    : function(agents, moderator, rng) → (solved, n_guesses,
                                                             agent_mem, mod_mem)
                        This is different per scenario.
        config        : TrainConfig object
    """

    def __init__(self, agents, moderator, episode_fn, config=None):
        self.agents    = agents
        self.moderator = moderator
        self.ep_fn     = episode_fn
        self.cfg       = config or TrainConfig()

        self.rng = np.random.default_rng(self.cfg.seed)

        # Running baselines — one per agent + one for moderator
        # These are exponential moving averages of recent returns
        self.base_agent = [0.0, 0.0, 0.0]
        self.base_mod   = 0.0

        # Tracking statistics
        self.wins      = []
        self.guess_log = []

    def _compute_returns(self, rewards):
        """
        Compute discounted returns from a list of rewards.

        If rewards = [r0, r1, r2] and gamma = 0.97:
            G0 = r0 + 0.97*r1 + 0.97^2*r2
            G1 = r1 + 0.97*r2
            G2 = r2

        This is done in reverse for efficiency.
        """
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.cfg.gamma * G
            returns.append(G)
        returns.reverse()
        return returns

    def _train_moderator(self, mod_mem):
        """Update moderator using episodic REINFORCE."""
        rewards = [r for (_, _, r) in mod_mem]
        returns = self._compute_returns(rewards)

        for (s, c, _), Gt in zip(mod_mem, returns):
            # Update baseline: moving average of returns
            self.base_mod = (
                self.cfg.beta * self.base_mod +
                (1 - self.cfg.beta) * Gt
            )
            # Advantage = how much better was this than average?
            advantage = Gt - self.base_mod
            self.moderator.pg_step(s, c, advantage, self.cfg.grad_clip)

    def _train_agents(self, agent_mem):
        """Update each agent using per-turn bandit REINFORCE."""
        for (which, s, a, r) in agent_mem:
            # Update this agent's baseline
            self.base_agent[which] = (
                self.cfg.beta * self.base_agent[which] +
                (1 - self.cfg.beta) * r
            )
            advantage = r - self.base_agent[which]
            self.agents[which].pg_step(s, a, advantage, self.cfg.grad_clip)

    def run(self):
        """
        Run the full training loop.
        Returns final win rate and average guesses when solved.
        """
        cfg = self.cfg
        t0  = time.time()
        lbl = f"[{cfg.label}]" if cfg.label else ""

        print(f"\n{lbl} Starting REINFORCE training")
        print(f"  Episodes: {cfg.episodes}  LR: {cfg.lr}  "
              f"Gamma: {cfg.gamma}  Seed: {cfg.seed}")
        print("─" * 60)

        for ep in range(cfg.episodes):

            solved, n_guesses, agent_mem, mod_mem = self.ep_fn(
                self.agents, self.moderator, self.rng
            )

            self.wins.append(1 if solved else 0)
            self.guess_log.append(n_guesses)

            self._train_moderator(mod_mem)
            self._train_agents(agent_mem)

            if (ep + 1) % cfg.log_every == 0:
                self._log(ep + 1, time.time() - t0)

        print("─" * 60)
        return self._final_metrics()

    def _log(self, ep, elapsed):
        """Print a progress line."""
        cfg = self.cfg
        lbl = f"[{cfg.label}]" if cfg.label else ""

        win_rate  = np.mean(self.wins[-cfg.window:])
        solved_gs = [g for g, w in
                     zip(self.guess_log[-cfg.window:], self.wins[-cfg.window:])
                     if w]
        avg_g = np.mean(solved_gs) if solved_gs else float("nan")

        print(f"  {lbl} ep {ep:6d} | win% {win_rate*100:5.1f} | "
              f"avg guesses (won) {avg_g:.2f} | {elapsed:.1f}s")

    def _final_metrics(self):
        """Return a dict of final training statistics."""
        cfg       = self.cfg
        final_wr  = float(np.mean(self.wins[-cfg.window:]))
        solved_gs = [g for g, w in zip(self.guess_log, self.wins) if w]
        avg_g     = float(np.mean(solved_gs[-cfg.window:])) if solved_gs else float("nan")

        metrics = {
            "label":                   cfg.label,
            "episodes":                cfg.episodes,
            "final_win_rate":          round(final_wr, 4),
            "avg_guesses_when_solved": round(avg_g, 3) if not np.isnan(avg_g) else None,
            "win_curve_per_1000":      [
                round(float(np.mean(self.wins[i:i+1000])), 3)
                for i in range(0, len(self.wins), 1000)
            ],
        }

        lbl = f"[{cfg.label}]" if cfg.label else ""
        print(f"{lbl} Final: win rate {final_wr*100:.1f}%  "
              f"avg guesses {avg_g:.2f}")

        return metrics


# ── Evaluation utilities ───────────────────────────────────────────────────────

def evaluate_agents(agents, moderator, episode_fn, n_eval=1000, seed=99999,
                    label=""):
    """
    Run n_eval games without training and report statistics.

    Returns a dict with win rate, average guesses, and agent usage.
    """
    rng           = np.random.default_rng(seed)
    wins          = []
    guess_log     = []
    agent_choices = [0, 0, 0]   # How often did the moderator pick each agent?

    for _ in range(n_eval):
        solved, n_guesses, _, mod_mem = episode_fn(agents, moderator, rng)
        wins.append(1 if solved else 0)
        guess_log.append(n_guesses if solved else 6)

        # Count agent picks (from moderator memory)
        for (_, choice, _) in mod_mem:
            agent_choices[choice] += 1

    total_picks = sum(agent_choices) or 1
    wr          = float(np.mean(wins))
    solved_gs   = [g for g, w in zip(guess_log, wins) if w]
    avg_g       = float(np.mean(solved_gs)) if solved_gs else float("nan")

    lbl = f"[{label}]" if label else ""
    print(f"\n{lbl} Evaluation over {n_eval} games:")
    print(f"  Win rate:         {wr*100:.1f}%")
    print(f"  Avg guesses:      {avg_g:.2f}")
    print(f"  Agent usage:")
    names = ["Eliminator", "Probabilist", "RiskTaker"]
    for i, name in enumerate(names):
        pct = 100 * agent_choices[i] / total_picks
        print(f"    {name:<14}: {pct:.1f}%")

    return {
        "label":           label,
        "n_eval":          n_eval,
        "win_rate":        round(wr, 4),
        "avg_guesses":     round(avg_g, 3) if not np.isnan(avg_g) else None,
        "agent_usage":     {names[i]: round(agent_choices[i]/total_picks, 3)
                            for i in range(3)},
    }


def save_results(metrics, filename):
    """Save training/eval metrics to a JSON file."""
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved → {filename}")


def load_results(filename):
    """Load metrics from a JSON file."""
    import os
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        return json.load(f)
