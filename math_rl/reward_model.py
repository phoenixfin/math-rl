"""
reward_model.py — Learns mathematical intuition from human ratings.
Uses PyTorch (TensorFlow has no Python 3.14 wheel).

Features per path (formula-structure based, no predefined vocabulary):
  - Checkpoint coverage ratio
  - Checkpoint / step efficiency
  - Normalized path length
  - Mean formula size
  - Mean formula depth
  - MP step ratio
  - Formula complexity growth (slope of sizes along path)
  - Novel formulas per step

Output: predicted composite score (1-5), scaled to env reward magnitude.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from math_rl.formula import formula_size, formula_depth, CHECKPOINTS_DEF

MIN_RATINGS   = 10
N_FEATURES    = 8
MAX_STEPS     = 300
N_CHECKPOINTS = len(CHECKPOINTS_DEF)


def path_to_features(path_record: Dict) -> np.ndarray:
    """
    Extract fixed-size feature vector from a generative path record.
    Works for both new-style (steps list) and legacy (path list of IDs).
    """
    steps = path_record.get('steps', [])

    if not steps:
        return np.zeros(N_FEATURES, dtype=np.float32)

    formulas = [s['formula'] for s in steps if 'formula' in s]
    cps_hit  = path_record.get('checkpoints_hit', [])
    n_steps  = len(steps)

    cp_coverage   = len(cps_hit) / N_CHECKPOINTS
    cp_efficiency = len(cps_hit) / max(n_steps, 1)
    path_len      = n_steps / MAX_STEPS

    sizes      = [formula_size(f) for f in formulas] if formulas else [0]
    mean_size  = np.mean(sizes) / 15.0
    depths     = [formula_depth(f) for f in formulas] if formulas else [0]
    mean_depth = np.mean(depths) / 6.0

    n_mp     = sum(1 for s in steps if s.get('rule') == 'MP')
    mp_ratio = n_mp / max(n_steps, 1)

    if len(sizes) >= 2:
        growth = (sizes[-1] - sizes[0]) / max(sizes[0], 1)
        growth = float(np.clip(growth, -1, 3) / 3.0)
    else:
        growth = 0.0

    cp_steps      = sum(1 for s in steps if s.get('cp') is not None)
    cp_step_ratio = cp_steps / max(n_steps, 1)

    return np.array([
        cp_coverage, cp_efficiency, path_len,
        mean_size, mean_depth, mp_ratio,
        growth, cp_step_ratio,
    ], dtype=np.float32)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RewardModel:
    def __init__(self, lr: float = 1e-3):
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net     = _Net().to(self.device)
        self.optim   = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.trained = False

    def train(self, ratings_file: str, epochs: int = 100) -> Optional[float]:
        rp = Path(ratings_file)
        if not rp.exists():
            return None
        ratings = json.loads(rp.read_text())
        if len(ratings) < MIN_RATINGS:
            print(f"[RewardModel] Need {MIN_RATINGS} ratings, have {len(ratings)}.")
            return None

        X = np.array([path_to_features(r) for r in ratings], dtype=np.float32)
        y = np.array([(float(r.get('composite', 3.0)) - 1) / 4.0
                      for r in ratings], dtype=np.float32)

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).to(self.device)

        self.net.train()
        last_loss = None
        for _ in range(epochs):
            self.optim.zero_grad()
            pred = self.net(X_t)
            loss = self.loss_fn(pred, y_t)
            loss.backward()
            self.optim.step()
            last_loss = loss.item()

        self.trained = True
        print(f"[RewardModel] Trained on {len(X)} ratings. Loss={last_loss:.4f}")
        return last_loss

    def predict(self, path_record: Dict) -> float:
        """Returns reward signal scaled to match env reward magnitude (~0-15)."""
        if not self.trained:
            return 0.0
        feats = torch.FloatTensor(path_to_features(path_record)).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            raw = float(self.net(feats).item())
        return raw * 15.0

    def save(self, path: str):
        torch.save({'net': self.net.state_dict()}, str(path) + '.pt')

    def load(self, path: str):
        pt_path = str(path) + '.pt'
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt['net'])
        self.trained = True
