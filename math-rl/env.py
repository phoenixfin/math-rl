"""
PropLogicEnv — RL environment for propositional logic theorem discovery.

State  : binary vector — which theorems are currently proven
Action : which theorem to attempt to derive next
Reward : +checkpoint_value for significant theorems, +1 for others,
         −0.3 for invalid attempts, −0.02 per step

The agent does NOT know prerequisites. It must discover the correct
derivation order through exploration and reward signals.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Set, Dict, Tuple, Optional

# ── Theorem Registry ──────────────────────────────────────────────────────────
THEOREMS: List[str] = [
    'A1', 'A2', 'A3',           # Hilbert axioms (always derivable)
    'T1',  'T2',  'T3',  'T4',  # Layer 1-2
    'T5',  'T6',  'T7',  'T8',  # Layer 3-4
    'T9',  'T10', 'T11', 'T12', # Layer 5
    'T13', 'T14',               # Layer 5 (De Morgan)
]

FORMULAS: Dict[str, str] = {
    'A1':  'p → (q → p)',
    'A2':  '(p → (q → r)) → ((p → q) → (p → r))',
    'A3':  '(¬q → ¬p) → (p → q)',
    'T1':  'p → p',
    'T2':  '(p → q) → ((q → r) → (p → r))',
    'T3':  '(q → r) → ((p → q) → (p → r))',
    'T4':  '¬¬p → p',
    'T5':  'p → ¬¬p',
    'T6':  '(p → q) → (¬q → ¬p)',
    'T7':  '¬p → (p → q)',
    'T8':  'p ∨ ¬p',
    'T9':  '¬(p ∧ ¬p)',
    'T10': '((p → q) ∧ (p → ¬q)) → ¬p',
    'T11': '(p ∧ q) → p',
    'T12': '(p → (q → r)) ↔ ((p ∧ q) → r)',
    'T13': '¬(p ∨ q) ↔ (¬p ∧ ¬q)',
    'T14': '¬(p ∧ q) ↔ (¬p ∨ ¬q)',
}

NAMES: Dict[str, str] = {
    'A1': 'Weakening',              'A2': 'Frege / Distribution',
    'A3': 'Contraposition',         'T1': 'Identity',
    'T2': 'Argument Permutation',   'T3': 'Hypothetical Syllogism',
    'T4': 'Double Negation Elim',   'T5': 'Double Negation Intro',
    'T6': 'Contrapositive',         'T7': 'Ex Falso Quodlibet',
    'T8': 'Law of Excluded Middle', 'T9': 'Non-Contradiction',
    'T10': 'Reductio ad Absurdum',  'T11': 'Conjunction Elim',
    'T12': 'Currying / Exportation','T13': "De Morgan I",
    'T14': "De Morgan II",
}

# Prerequisites: what must be proven before this theorem can be derived
PREREQUISITES: Dict[str, Set[str]] = {
    'A1': set(), 'A2': set(), 'A3': set(),  # axioms: always available
    'T1':  {'A1', 'A2'},
    'T2':  {'A1', 'A2'},
    'T3':  {'A2', 'T2'},
    'T4':  {'A3', 'T1'},
    'T5':  {'A3', 'T4'},
    'T6':  {'A3', 'T3', 'T4'},
    'T7':  {'A1', 'A3', 'T4'},
    'T8':  {'A3', 'T4', 'T7'},
    'T9':  {'T6', 'T1'},
    'T10': {'A2', 'T6'},
    'T11': {'A1', 'T3'},
    'T12': {'A1', 'A2', 'T3', 'T11'},
    'T13': {'T6', 'T8', 'T9'},
    'T14': {'T6', 'T8', 'T13'},
}

# Checkpoints and their rewards (higher = more significant)
CHECKPOINT_REWARDS: Dict[str, float] = {
    'T1':  5.0,   # Identity — fundamental
    'T3':  8.0,   # Hypothetical Syllogism — very useful
    'T4':  10.0,  # Double Negation Elim — classical logic hallmark
    'T6':  8.0,   # Contrapositive
    'T7':  6.0,   # Ex Falso
    'T8':  12.0,  # Excluded Middle — pinnacle of classical logic
    'T9':  10.0,  # Non-Contradiction
    'T10': 6.0,   # Reductio ad Absurdum
    'T12': 7.0,   # Currying
    'T13': 9.0,   # De Morgan I
    'T14': 9.0,   # De Morgan II
}
CHECKPOINTS: Set[str] = set(CHECKPOINT_REWARDS.keys())


# ── Environment ───────────────────────────────────────────────────────────────
class PropLogicEnv(gym.Env):
    """
    Gym environment for propositional logic theorem discovery.

    The agent selects which theorem to try to derive at each step.
    It doesn't know the prerequisites — it must discover valid ordering
    through trial, error, and rewards.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps: int = 120):
        super().__init__()
        self.theorem_list  = THEOREMS
        self.n             = len(THEOREMS)
        self.t2i           = {t: i for i, t in enumerate(THEOREMS)}
        self.max_steps     = max_steps

        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.n,), dtype=np.float32)
        self.action_space      = spaces.Discrete(self.n)

        self.proven: Set[str]          = set()
        self.checkpoints_hit: Set[str] = set()
        self.steps: int                = 0
        self.history: List[Tuple]      = []   # (theorem, success, reward)

    # ── Core methods ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.proven          = set()
        self.checkpoints_hit = set()
        self.steps           = 0
        self.history         = []
        return self._obs(), {}

    def step(self, action: int):
        theorem  = self.theorem_list[action]
        reward   = -0.02   # small step cost
        derived  = False
        reason   = ''

        if theorem in self.proven:
            reward -= 0.1
            reason  = 'already proven'

        elif PREREQUISITES[theorem].issubset(self.proven):
            self.proven.add(theorem)
            derived = True

            if theorem in CHECKPOINTS:
                cp_r    = CHECKPOINT_REWARDS[theorem]
                reward += cp_r
                self.checkpoints_hit.add(theorem)
                reason  = f'CHECKPOINT +{cp_r:.0f}'
            else:
                reward += 1.0
                reason  = 'derived'

        else:
            reward -= 0.3
            missing = PREREQUISITES[theorem] - self.proven
            reason  = f'missing prereqs: {missing}'

        self.steps += 1
        self.history.append((theorem, derived, reward))

        done      = self.checkpoints_hit == CHECKPOINTS
        truncated = self.steps >= self.max_steps

        info = {
            'theorem': theorem, 'name': NAMES[theorem],
            'derived': derived, 'reason': reason,
            'checkpoints_hit': len(self.checkpoints_hit),
        }
        return self._obs(), reward, done, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in self.proven:
            s[self.t2i[t]] = 1.0
        return s

    def get_valid_actions(self) -> List[int]:
        """Actions that would currently succeed (unproven + prereqs met)."""
        return [
            self.t2i[t] for t in self.theorem_list
            if t not in self.proven
            and PREREQUISITES[t].issubset(self.proven)
        ]

    def render(self):
        proven   = sorted(self.proven, key=lambda t: self.t2i[t])
        cps      = [t for t in proven if t in CHECKPOINTS]
        pending  = [t for t in self.theorem_list if t not in self.proven]
        print(f"\n{'─'*55}")
        print(f"Step {self.steps}/{self.max_steps}")
        print(f"Proven     : {proven}")
        print(f"Checkpoints: {cps}  ({len(cps)}/{len(CHECKPOINTS)})")
        print(f"Pending    : {pending}")
        print(f"{'─'*55}")
