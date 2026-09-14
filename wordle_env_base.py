"""
wordle_env_base.py — Shared base environment for all Wordle scenarios.

This file contains:
  - The word list and helper functions used by all 3 scenarios
  - The reward functions (task_reward and agent_reward)
  - The Policy class (the neural network each agent uses to learn)
  - The base WordleEnv class that scenarios can extend

Reward system explanation:
  - task_reward:  how well did the TEAM do this turn?
                  greens (right letter, right place) score highest,
                  yellows (right letter, wrong place) score a little,
                  and we add a raw letter-match score so the agents
                  also learn to get closer to the secret in pure letter
                  distance (not just Wordle feedback).
  - agent_reward: how good was THIS AGENT's individual guess strategy?
                  Each agent type (Eliminator, Probabilist, RiskTaker)
                  has a different way of judging quality.
"""

import numpy as np

# ── Word list (English 5-letter words) ────────────────────────────────────────

WORDS_EN = [
    "crane","slate","trace","stare","arise","raise","audio","about","other","their",
    "there","would","these","thing","world","water","after","where","right","think",
    "three","years","place","sound","great","again","still","every","found","those",
    "while","house","under","never","start","might","money","point","group","often",
    "until","since","power","story","value","level","study","light","night","heart",
    "table","music","field","plant","brain","clean","clear","close","cover","dream",
    "drive","early","earth","enjoy","enter","equal","event","exist","extra","faith",
    "false","fault","floor","fence","fiber","final","first","focus","force","frame",
    "fresh","front","fruit","funny","giant","glass","grace","grade","grand","grant",
    "green","guard","guess","guest","guide","happy","heavy","horse","hotel","human",
    "ideal","image","index","input","issue","joint","judge","knife","known","large",
    "laugh","layer","learn","least","leave","lemon","local","logic","loose","lucky",
    "magic","major","maker","march","match","media","metal","model","month","moral",
    "motor","mount","mouse","mouth","movie","naive","nerve","noise","north","noted",
    "novel","nurse","occur","ocean","offer","olive","order","outer","owned","owner",
    "paint","panel","paper","party","pasta","patch","pause","peace","pearl","penny",
    "phase","phone","photo","piano","piece","pilot","pitch","pixel","pizza","plain",
    "plane","plank","plate","plaza","plead","pluck","plumb","plume","plunk","plush",
    "poach","polar","poppy","porch","pound","press","price","pride","prime","print",
    "prior","prize","probe","prone","proof","prose","proud","prove","prowl","prune",
    "pulse","punch","pupil","purse","queen","query","quest","quick","quiet","quota",
    "quote","radar","radio","ranch","range","rapid","ratio","reach","ready","realm",
    "rebel","refer","reign","relax","reply","reset","rider","ridge","rifle","rigid",
    "risky","rival","river","rocky","rouge","rough","round","route","rowdy","ruler",
    "rural","saint","salad","sauce","scale","scene","scoop","score","scout","screw",
    "sedan","seize","sense","serve","setup","seven","shade","shaft","shake","shame",
    "shape","shark","sharp","shear","sheep","sheer","sheet","shelf","shell","shift",
    "shirt","shock","shore","short","shout","shove","shown","sight","silly","sixth",
    "sixty","skill","skull","slack","slant","slash","sleep","slice","slide","slime",
    "sling","slope","slosh","sloth","slump","slurp","smack","small","smash","smear",
    "smell","smile","smirk","smoke","snack","snail","snake","snare","snarl","sneak",
    "sneer","sniff","snore","snort","snout","snuff","solar","solid","solve","sorry",
    "south","space","spare","spark","speak","spear","speck","speed","spend","spice",
    "spike","spill","spine","spite","splat","split","spoke","spook","spoon","sport",
    "spout","spray","spree","squad","squat","squid","stack","staff","stage","stain",
    "stair","stake","stale","stall","stamp","stand","stank","stark","steam","steel",
    "steep","steer","stern","stick","stiff","sting","stink","stint","stock","stomp",
    "stone","stood","stool","stoop","store","storm","stout","stove","strap","straw",
    "stray","strip","strut","stuck","stuff","stump","stung","stunk","stunt","sugar",
    "suite","sunny","super","surge","swamp","swear","sweat","sweep","sweet","swept",
    "swift","swirl","swoop","sword","swore","sworn","swung","taken","tally","talon",
    "tango","taste","teach","tease","thank","theme","thick","thief","thigh","thorn",
    "threw","throw","thumb","tiger","tight","timer","tired","title","today","token",
    "tooth","topic","total","touch","tough","towel","tower","toxic","track","trail",
    "train","tramp","trash","trawl","tread","treat","trend","trial","trick","troop",
    "trout","trove","truce","truck","truly","trunk","trust","truth","tulip","tumor",
    "tuner","tunic","twang","tweak","tweed","twice","twill","twist","ultra","union",
    "unite","upper","upset","urban","usage","usher","usual","utter","vague","valid",
    "valor","valve","vault","vigor","viral","virus","visor","vista","vivid","vodka",
    "voice","vomit","voter","vouch","wacky","wafer","waltz","watch","weary","weave",
    "wedge","weird","whale","whack","wheat","wheel","whiff","whirl","whisk","white",
    "whole","widen","widow","wield","witty","woman","women","woody","woozy","worse",
    "worst","worth","wound","wrath","wrist","wrote","yacht","yearn","yeast","yield",
    "young","yours","youth","zebra","zonal",
]

# ── Macedonian word list (5-letter Cyrillic words) ─────────────────────────────
# Note: These are real Macedonian 5-letter words written in Cyrillic.
# You can expand this list with more words from a Macedonian dictionary.
WORDS_MK = [
    "мајка","татко","земја","вода","оган","куќа","сонце","месец","ѕвезд","ветер",
    "дрво","цвет","птица","риба","коњ","мачка","куче","овца","крава","свиња",
    "леб","млеко","маса","столч","врата","прозо","книга","молив","школо","учите",
    "море","езеро","река","гора","поле","ливад","камен","песок","небо","облак",
    "снег","дожд","град","село","град","пат","мост","воз","кола","брод",
    "рака","нога","глава","очи","уши","нос","уста","заби","коса","срце",
    "живот","смрт","љубов","омраз","радос","тага","страв","смеа","плач","викот",
    "бело","црно","црвен","жолто","зелен","сино","виоле","порток","розово","сиво",
    "еден","двата","три","четир","пет","шест","седум","осум","девет","десет",
    "голем","мал","висок","низок","брз","бавен","нов","стар","убав","грд",
]

# Filter to only keep proper-length words and remove duplicates
WORDS_EN = list(dict.fromkeys(w for w in WORDS_EN if len(w) == 5))
WORDS_MK = list(dict.fromkeys(w for w in WORDS_MK if len(w) == 5))

# Feedback value constants — used everywhere
GREY   = 0   # Letter not in word at all
YELLOW = 1   # Letter in word but wrong position
GREEN  = 2   # Letter in correct position

MAX_TURNS = 6

# Agent type IDs — used in agent_reward
ELIMINATOR  = 0
PROBABILIST = 1
RISKTAKER   = 2

AGENT_NAMES = ["Eliminator", "Probabilist", "RiskTaker"]

# Per-agent reward weights for the letter-difference score.
# RiskTaker gets a lower weight because it intentionally guesses
# "far" words to gather information — we don't want to punish that.
DIFF_WEIGHTS = {
    ELIMINATOR:  0.2,
    PROBABILIST: 0.35,
    RISKTAKER:   0.1,
}

INFO_GAIN_WEIGHT = 1.0


# ── Core feedback function ─────────────────────────────────────────────────────

def compute_feedback(guess_word, secret_word):
    """
    Compare guess_word to secret_word letter by letter.
    Returns a tuple of 5 values, each GREEN / YELLOW / GREY.

    Example:
        compute_feedback("crane", "trace")
        → (YELLOW, YELLOW, GREEN, YELLOW, GREEN)  i.e. (1,1,2,1,2)
    """
    fb     = [GREY] * 5
    counts = {}

    # Count how many of each letter the secret has
    for ch in secret_word:
        counts[ch] = counts.get(ch, 0) + 1

    # First pass: mark greens
    for i in range(5):
        if guess_word[i] == secret_word[i]:
            fb[i] = GREEN
            counts[guess_word[i]] -= 1

    # Second pass: mark yellows
    for i in range(5):
        if fb[i] == GREY and counts.get(guess_word[i], 0) > 0:
            fb[i] = YELLOW
            counts[guess_word[i]] -= 1

    return tuple(fb)


def fb_to_str(fb):
    """Convert feedback tuple to a readable string. E.g. (2,1,0,0,2) → 'GY..G'"""
    return "".join({GREEN: "G", YELLOW: "Y", GREY: "."}[x] for x in fb)


def filter_candidates(cands, guess_idx, fb, pattern_matrix):
    """
    Keep only the word indices from cands that are consistent with
    this guess getting this feedback.
    """
    return [c for c in cands if pattern_matrix[guess_idx, c] == fb]


def build_pattern_matrix(words):
    """
    Pre-compute feedback for every pair of (guess, secret).
    This is slow to build but makes the game very fast afterwards.
    Returns a 2D numpy array where pattern[g, s] = feedback tuple.
    """
    n = len(words)
    pattern = np.empty((n, n), dtype=object)
    for g in range(n):
        for s in range(n):
            pattern[g, s] = compute_feedback(words[g], words[s])
    return pattern


# ── Letter-difference score (NEW) ─────────────────────────────────────────────

def letter_match_score(guess_word, secret_word):
    """
    How many letter positions match exactly between guess and secret?
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).

    This is DIFFERENT from greens/yellows — it only looks at exact
    position matches, ignoring whether a letter appears elsewhere.

    Example:
        guess  = "crane"
        secret = "trace"
        Matching positions: r(pos1) no, a(pos2) yes → 1/5 = 0.2... 
        actually let's check: c≠t, r=r yes, a=a yes, n≠c, e=e yes → 3/5 = 0.6
    """
    word_len = len(guess_word)
    matches  = sum(1 for g, s in zip(guess_word, secret_word) if g == s)
    return matches / word_len


# ── Partition quality (replaces log-based entropy) ────────────────────────────

def partition_quality(guess_idx, cands, pattern_matrix):
    """
    How well does this guess split the remaining candidates into groups?

    Instead of using log2 (entropy), we use a simpler measure:
        quality = 1 - sum((group_size / total)^2)

    This is called the Gini impurity. It equals 0 when all candidates
    land in the same group (bad — no information gained) and is high
    when the candidates spread evenly across many groups (good).

    It's mathematically similar to entropy but uses only basic arithmetic.
    Range: 0.0 (worst) to ~1.0 (best).
    """
    if len(cands) <= 1:
        return 0.0

    groups = {}
    for c in cands:
        p = pattern_matrix[guess_idx, c]
        groups[p] = groups.get(p, 0) + 1

    total = len(cands)
    # Sum of (fraction)^2 for each group — high means concentrated (bad)
    sum_sq = sum((count / total) ** 2 for count in groups.values())

    # Subtract from 1: high result means well-spread (good)
    return 1.0 - sum_sq


def expected_remaining_frac(guess_idx, cands, pattern_matrix):
    """
    After playing this guess, what fraction of candidates do we expect
    to keep on average? Lower is better (we eliminated more words).
    """
    if not cands:
        return 0.0
    groups = {}
    for c in cands:
        p = pattern_matrix[guess_idx, c]
        groups[p] = groups.get(p, 0) + 1
    total = len(cands)
    return sum((n_p / total) ** 2 for n_p in groups.values())


# ── Reward functions ───────────────────────────────────────────────────────────

def task_reward(fb, solved, last_turn, guess_word=None, secret_word=None, which=None,
                 cands_before=None, cands_after=None, info_gain_weight=INFO_GAIN_WEIGHT):
    """
    Reward signal for the TEAM (used by the moderator).

    Parameters:
        fb              : feedback tuple from compute_feedback, e.g. (2,1,0,0,2)
        solved          : True if the guess matched the secret
        last_turn       : True if this was the last allowed guess
        guess_word      : the actual guess string (e.g. "crane")  — optional
        secret_word     : the actual secret string (e.g. "trace") — optional
        which           : agent type (0/1/2) — used to pick diff_weight
        cands_before    : number of candidates before this guess — optional
        cands_after     : number of candidates after this guess  — optional
        info_gain_weight: weight applied to the info-gain term

    Returns a float reward value.

    Breakdown:
        +0.2 per green  (right letter, right position)
        +0.05 per yellow (right letter, wrong position)
        -0.1  base penalty per turn (encourages solving quickly)
        +letter_match_score * weight  (how close was the guess in raw letters?)
        +info_gain_weight * fraction of candidates eliminated this turn
        +5.0 if solved
        -1.0 if last turn and not solved
    """
    greens  = sum(1 for x in fb if x == GREEN)
    yellows = sum(1 for x in fb if x == YELLOW)

    r = 0.2 * greens + 0.05 * yellows - 0.1

    if guess_word is not None and secret_word is not None:
        diff_weight = DIFF_WEIGHTS.get(which, 0.2)
        r += diff_weight * letter_match_score(guess_word, secret_word)

    if cands_before is not None and cands_after is not None and cands_before > 0:
        r += info_gain_weight * (1.0 - (cands_after / cands_before))

    if solved:
        r += 5.0
    elif last_turn:
        r -= 1.0

    return r


def agent_reward(which, guess_idx, cands, secret_idx, words, pattern_matrix):
    """
    Reward signal for an INDIVIDUAL AGENT based on its strategy type.

    Parameters:
        which         : agent type — ELIMINATOR, PROBABILIST, or RISKTAKER
        guess_idx     : index of the word the agent guessed
        cands         : list of candidate word indices still possible
        secret_idx    : index of the secret word
        words         : the full word list
        pattern_matrix: precomputed feedback matrix

    Returns a float reward value.

    Each agent is rewarded differently:
        Eliminator  → rewarded for eliminating more candidates
        Probabilist → rewarded for picking common/likely words
        RiskTaker   → rewarded for splitting candidates into even groups
    """
    win_bonus = 1.5 if guess_idx == secret_idx else 0.0
    fb        = pattern_matrix[guess_idx, secret_idx]

    guess_word  = words[guess_idx]
    secret_word = words[secret_idx]

    if which == ELIMINATOR:
        # How many candidates did we eliminate?
        kept = len(filter_candidates(cands, guess_idx, fb, pattern_matrix))
        base = (len(cands) - kept) / max(len(cands), 1)

    elif which == PROBABILIST:
        # Is this guess one of the likely candidates, weighted by frequency rank?
        n     = len(words)
        prior = np.array([n - i for i in range(n)], dtype=np.float64)
        prior = prior / prior.max()
        in_cands = 1.0 if guess_idx in cands else 0.0
        base     = 0.6 * in_cands + 0.4 * prior[guess_idx]

    else:  # RISKTAKER
        # How well does this guess partition the remaining candidates?
        # (uses Gini impurity instead of entropy — no log needed)
        base = partition_quality(guess_idx, cands, pattern_matrix)

    # Add letter-match bonus (with agent-specific weight)
    diff_weight = DIFF_WEIGHTS[which]
    base += diff_weight * letter_match_score(guess_word, secret_word)

    return base + win_bonus


# ── Agent state vector ─────────────────────────────────────────────────────────

def build_agent_state(cands, turn, absent_flags, known_green, yellows, n_words):
    """
    Build the state vector that the agent's neural network receives.

    This tells the agent:
      - which words are still possible (one-hot mask)
      - what turn it is
      - which letters have been ruled out (absent)
      - which letters are confirmed in which positions (green)
      - which letters are present somewhere (yellow)

    Returns a 1D numpy array.
    """
    # 1. Candidate mask: 1.0 for each word still in play
    mask = np.zeros(n_words)
    mask[cands] = 1.0

    # 2. Turn one-hot: which turn are we on?
    t = np.zeros(MAX_TURNS)
    if turn < MAX_TURNS:
        t[turn] = 1.0

    # 3. Green encoding: 5 positions × 26 letters
    green_enc = np.zeros(5 * 26)
    for pos, ch in enumerate(known_green):
        if ch:
            green_enc[pos * 26 + (ord(ch) - 97)] = 1.0

    # 4. Yellow encoding: which letters are present but misplaced
    yellow_enc = np.zeros(26)
    for ch in yellows:
        yellow_enc[ord(ch) - 97] = 1.0

    return np.concatenate([mask, t, absent_flags, green_enc, yellow_enc])


# ── Policy network (the "brain" of each agent) ────────────────────────────────

class Policy:
    """
    A simple 2-layer neural network that learns which word to guess.

    Input  → Hidden layer (tanh activation) → Output (softmax probabilities)

    We use REINFORCE (policy gradient) to train it — the network learns
    by receiving reward signals and adjusting weights to make rewarded
    actions more likely in the future.

    You don't need to understand the math deeply — the key idea is:
        - forward()   : given a state, returns probabilities for each word
        - sample()    : pick a word to guess (randomly, weighted by probs)
        - pg_step()   : update weights based on how good the guess was
    """

    def __init__(self, in_dim, hidden_dim, out_dim, learning_rate, seed=0):
        rng = np.random.default_rng(seed)
        # He initialisation — helps the network train stably
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim),     (hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (out_dim, hidden_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = learning_rate

    def forward(self, x):
        """Compute action probabilities for state x."""
        self._x  = x
        self._z1 = self.W1 @ x + self.b1
        self._h  = np.tanh(self._z1)             # Hidden layer activation
        logits   = self.W2 @ self._h + self.b2
        logits  -= logits.max()                   # Numerical stability trick
        exp_l    = np.exp(logits)
        self._p  = exp_l / exp_l.sum()            # Softmax → probabilities
        return self._p

    def sample(self, x, valid_indices=None, rng=None):
        """
        Pick an action (word index) based on probabilities.
        If valid_indices is given, only choose from those words.
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.forward(x)

        if valid_indices is not None:
            mask  = np.zeros(len(p))
            mask[valid_indices] = 1.0
            p     = p * mask
            total = p.sum()
            if total > 0:
                p = p / total
            else:
                # After many training updates the network can become so
                # confident in one word that every OTHER word's probability
                # underflows to exactly 0.0. If that one confident word isn't
                # a valid candidate anymore, masking zeroes out everything —
                # fall back to a uniform pick among the valid words instead
                # of crashing on a 0/0 division.
                p = mask / mask.sum()
        else:
            p = p / p.sum()

        a = rng.choice(len(p), p=p)
        return a, p

    def pg_step(self, x, action, advantage, grad_clip=5.0):
        """
        Update network weights using the REINFORCE policy gradient rule.

        advantage > 0 → make this action more likely next time
        advantage < 0 → make this action less likely next time
        """
        p          = self.forward(x)
        dlogits    = p.copy()
        dlogits[action] -= 1.0
        dlogits   *= advantage

        # Backprop through the two layers
        dW2 = np.outer(dlogits, self._h)
        db2 = dlogits
        dh  = self.W2.T @ dlogits
        dz1 = dh * (1.0 - self._h ** 2)   # tanh derivative
        dW1 = np.outer(dz1, self._x)
        db1 = dz1

        # Gradient clipping — prevents exploding gradients
        for g in (dW1, db1, dW2, db2):
            np.clip(g, -grad_clip, grad_clip, out=g)

        # Gradient descent step
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


# ── Proposal features for the moderator ───────────────────────────────────────

def proposal_features(guess_idx, cands, words, pattern_matrix):
    """
    Summarise a word proposal in 4 numbers the moderator can use
    to decide which agent's word to play.

    Returns [elimination_power, partition_quality_score, in_candidates, commonness]
    """
    n     = len(words)
    prior = np.array([n - i for i in range(n)], dtype=np.float64)
    prior = prior / prior.max()

    elim_power  = 1.0 - expected_remaining_frac(guess_idx, cands, pattern_matrix)
    part_qual   = partition_quality(guess_idx, cands, pattern_matrix)
    in_cands    = 1.0 if guess_idx in cands else 0.0
    commonness  = float(prior[guess_idx])

    return [elim_power, part_qual, in_cands, commonness]


def build_mod_state(proposals, cands, turn, words, pattern_matrix):
    """
    Build the state vector for the moderator (the agent that picks
    which agent's word to actually play).

    proposals : list of 3 word indices (one from each agent)
    """
    feats = []
    for g in proposals:
        feats += proposal_features(g, cands, words, pattern_matrix)
    # Add turn progress and how many candidates remain (normalised 0-1)
    feats += [turn / MAX_TURNS, len(cands) / max(len(words), 1)]
    return np.array(feats)
