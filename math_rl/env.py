"""
GenerativeLogicEnv — The agent constructs formulas from scratch using logical rules.

State:
    The current set of proven formulas, encoded as a fixed-size hash vector.

Actions:
    Pool = 3 atomic vars + last POOL_SIZE derived formulas (default 5) = 8 formulas
    ┌─────────────────────────────────────────────────────────────┐
    │ A1(p,q)        for p,q ∈ pool         : pool² actions      │
    │ A2(p,q,r)      for p,q,r ∈ pool       : pool³ actions      │
    │ A3(p,q)        for p,q ∈ pool         : pool² actions      │
    │ MP(major,minor) for pairs in proven   : pool² actions      │
    └─────────────────────────────────────────────────────────────┘
    Total ≈ 4 × pool² + pool³  (with POOL_SIZE=5 → ~640 actions)

Reward:
    +checkpoint_reward if derived formula matches a checkpoint (alpha-equiv)
    +2.0 for any novel interesting formula
    -0.5 for redundant (already proven)
    -0.1 step cost
    Optionally blended with learned reward model.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Set, List, Tuple, Optional, Callable
from itertools import product

from math_rl.formula import (
    VARS, A1, A2, A3,
    split_implication, is_atomic, is_negation,
    find_mp_consequences, match_checkpoint,
    is_interesting, formula_size,
    CHECKPOINTS_DEF,
)


# ── Action types ──────────────────────────────────────────────────────────────
ACT_A1 = 0
ACT_A2 = 1
ACT_A3 = 2
ACT_MP = 3

POOL_SIZE = 5        # number of recent derived formulas kept in substitution pool
HASH_BUCKETS = 256   # for state encoding
MAX_STEPS_DEFAULT = 200


class GenerativeLogicEnv(gym.Env):
    """
    Agent explores propositional logic by applying inference rules,
    generating formulas as lego bricks toward significant theorems.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        max_steps:    int   = MAX_STEPS_DEFAULT,
        pool_size:    int   = POOL_SIZE,
        reward_fn:    Optional[Callable] = None,
        reward_blend: float = 0.0,
    ):
        super().__init__()
        self.max_steps    = max_steps
        self.pool_size    = pool_size
        self.reward_fn    = reward_fn
        self.reward_blend = reward_blend

        # Pool = atomic vars + recent derived formulas
        self.pool_capacity = len(VARS) + pool_size  # 3 + 5 = 8

        # Action space: enumerate all rule+argument combinations
        # A1: pool × pool
        # A2: pool × pool × pool
        # A3: pool × pool
        # MP: pool × pool
        P = self.pool_capacity
        self.n_a1 = P * P
        self.n_a2 = P * P * P
        self.n_a3 = P * P
        self.n_mp = P * P

        self.total_actions = self.n_a1 + self.n_a2 + self.n_a3 + self.n_mp

        # State: hash vector over formula set + checkpoint binary vector
        self.n_checkpoints = len(CHECKPOINTS_DEF)
        self.state_size    = HASH_BUCKETS + self.n_checkpoints
        self.cp_ids        = list(CHECKPOINTS_DEF.keys())

        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.total_actions)

        # Runtime state
        self.proven:          Set[str]   = set()
        self.pool:            List[str]  = []   # substitution pool (atomics + recent)
        self.checkpoints_hit: Set[str]   = set()
        self.steps:           int        = 0
        self.history:         List[dict] = []

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.proven          = set()
        self.pool            = []   # starts empty; VARS always prepended by _get_pool
        self.checkpoints_hit = set()
        self.steps           = 0
        self.history         = []
        return self._obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, action: int):
        rule, args = self._decode_action(action)
        reward     = -0.1   # step cost
        result     = None
        info       = {'rule': rule, 'args': args, 'derived': None, 'cp': None}

        # ── Apply rule ────────────────────────────────────────────────────────
        try:
            pool = self._get_pool()
            if rule == 'A1' and len(args) == 2:
                result = A1(pool[args[0]], pool[args[1]])
            elif rule == 'A2' and len(args) == 3:
                result = A2(pool[args[0]], pool[args[1]], pool[args[2]])
            elif rule == 'A3' and len(args) == 2:
                result = A3(pool[args[0]], pool[args[1]])
            elif rule == 'MP' and len(args) == 2:
                f1 = pool[args[0]]   # antecedent
                f2 = pool[args[1]]   # should be (f1 → something)
                parts = split_implication(f2)
                if parts and parts[0] == f1 and f1 in self.proven and f2 in self.proven:
                    result = parts[1]
                else:
                    reward -= 0.3   # invalid MP
        except (IndexError, AssertionError):
            reward -= 0.3

        # ── Process result ────────────────────────────────────────────────────
        if result is not None:
            if result in self.proven:
                reward -= 0.5   # already known
            elif not is_interesting(result, self.proven):
                reward -= 0.2   # too complex / filtered
            else:
                self.proven.add(result)
                self._update_pool(result)
                info['derived'] = result

                # Check checkpoint
                cp = match_checkpoint(result)
                if cp and cp not in self.checkpoints_hit:
                    self.checkpoints_hit.add(cp)
                    cp_reward = CHECKPOINTS_DEF[cp]['reward']
                    reward += cp_reward
                    info['cp'] = cp
                else:
                    reward += 2.0   # novel formula

                # Blend with learned reward model
                if self.reward_fn is not None:
                    learned = float(self.reward_fn(list(self.proven)))
                    reward  = (1 - self.reward_blend) * reward + self.reward_blend * learned

        self.steps += 1
        self.history.append({'step': self.steps, 'rule': rule,
                             'result': result, 'reward': reward, **info})

        done      = self.checkpoints_hit == set(self.cp_ids)
        truncated = self.steps >= self.max_steps

        info.update({
            'checkpoints_hit':   len(self.checkpoints_hit),
            'total_checkpoints': self.n_checkpoints,
            'proven_count':      len(self.proven),
        })
        return self._obs(), reward, done, truncated, info

    # ── Valid actions ─────────────────────────────────────────────────────────
    def get_valid_actions(self) -> List[int]:
        """
        Returns actions that would produce a NEW formula.
        Axiom actions: always valid (generate novel formulas usually)
        MP actions: only valid if both antecedent and (ant→cons) are proven.
        """
        pool     = self._get_pool()
        P        = len(pool)
        valid    = []
        offset   = 0

        # A1 actions: always valid (might be redundant, but always applicable)
        for i, j in product(range(P), repeat=2):
            result = A1(pool[i], pool[j])
            if result not in self.proven:
                valid.append(offset + i * P + j)
        offset += self.n_a1

        # A2 actions: always valid
        for i, j, k in product(range(P), repeat=3):
            result = A2(pool[i], pool[j], pool[k])
            if result not in self.proven:
                valid.append(offset + i * P * P + j * P + k)
        offset += self.n_a2

        # A3 actions: always valid
        for i, j in product(range(P), repeat=2):
            result = A3(pool[i], pool[j])
            if result not in self.proven:
                valid.append(offset + i * P + j)
        offset += self.n_a3

        # MP actions: valid only when antecedent + implication are both proven
        for i, j in product(range(P), repeat=2):
            f1, f2 = pool[i], pool[j]
            parts  = split_implication(f2)
            if (parts and parts[0] == f1
                    and f1 in self.proven
                    and f2 in self.proven
                    and parts[1] not in self.proven):
                valid.append(offset + i * P + j)

        return valid

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_pool(self) -> List[str]:
        """Current substitution pool: atomics + recent derived."""
        return list(VARS) + self.pool

    def _update_pool(self, new_formula: str):
        """Add new formula to pool, keep only last pool_size."""
        self.pool.append(new_formula)
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)

    def _decode_action(self, action: int) -> Tuple[str, List[int]]:
        """Map flat action index → (rule_name, [arg_indices])."""
        P = self.pool_capacity
        offset = 0

        if action < offset + self.n_a1:
            idx  = action - offset
            return 'A1', [idx // P, idx % P]
        offset += self.n_a1

        if action < offset + self.n_a2:
            idx = action - offset
            return 'A2', [idx // (P*P), (idx // P) % P, idx % P]
        offset += self.n_a2

        if action < offset + self.n_a3:
            idx = action - offset
            return 'A3', [idx // P, idx % P]
        offset += self.n_a3

        idx = action - offset
        return 'MP', [idx // P, idx % P]

    def _obs(self) -> np.ndarray:
        """
        State = hash bag-of-formulas (HASH_BUCKETS) + checkpoint binary (n_checkpoints)
        """
        # Hash each proven formula into a bucket
        hash_vec = np.zeros(HASH_BUCKETS, dtype=np.float32)
        for f in self.proven:
            bucket = hash(f) % HASH_BUCKETS
            hash_vec[bucket] = min(1.0, hash_vec[bucket] + 1.0)
        hash_vec = np.tanh(hash_vec)   # normalize

        # Checkpoint hits
        cp_vec = np.array(
            [1.0 if cp in self.checkpoints_hit else 0.0 for cp in self.cp_ids],
            dtype=np.float32
        )
        return np.concatenate([hash_vec, cp_vec])

    def render(self):
        print(f"\n{'─'*60}")
        print(f"Step {self.steps}/{self.max_steps}  |  Proven: {len(self.proven)}  |  "
              f"Pool: {self._get_pool()}")
        if self.checkpoints_hit:
            cps = [f"{c} ({CHECKPOINTS_DEF[c]['name']})" for c in self.checkpoints_hit]
            print(f"Checkpoints: {cps}")
        if self.proven:
            recent = sorted(self.proven, key=lambda f: formula_size(f))[:8]
            print(f"Proven (smallest): {recent}")
        print(f"{'─'*60}")
