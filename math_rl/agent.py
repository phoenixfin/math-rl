"""
DQN Agent — Double DQN with action masking and experience replay.
Uses PyTorch (TensorFlow has no Python 3.14 wheel).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Optional
import random


def _build_net(state_size: int, action_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(state_size, 128), nn.ReLU(),
        nn.Linear(128, 256),        nn.ReLU(),
        nn.Linear(256, 128),        nn.ReLU(),
        nn.Linear(128, action_size),
    )


class DQNAgent:
    def __init__(
        self,
        state_size:         int,
        action_size:        int,
        gamma:              float = 0.99,
        epsilon:            float = 1.0,
        epsilon_min:        float = 0.05,
        epsilon_decay:      float = 0.997,
        lr:                 float = 1e-3,
        batch_size:         int   = 64,
        memory_size:        int   = 20_000,
        target_update_freq: int   = 150,
    ):
        self.state_size         = int(state_size)
        self.action_size        = int(action_size)
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.memory             = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq
        self.train_step         = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model        = _build_net(self.state_size, self.action_size).to(self.device)
        self.target_model = _build_net(self.state_size, self.action_size).to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn      = nn.HuberLoss()
        self._sync_target()

    def _sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray,
            valid_actions: Optional[List[int]] = None) -> int:
        if np.random.random() < self.epsilon:
            pool = valid_actions if (valid_actions and np.random.random() < 0.8) \
                   else list(range(self.action_size))
            return random.choice(pool)

        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.model(t).cpu().numpy()[0]

        if valid_actions is not None and len(valid_actions) > 0:
            mask = np.full(self.action_size, -1e9, dtype=np.float32)
            mask[valid_actions] = q[valid_actions]
            return int(np.argmax(mask))
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, float(reward), next_state, bool(done)))

    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch       = random.sample(self.memory, self.batch_size)
        states      = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions     = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards     = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones       = torch.FloatTensor([float(e[4]) for e in batch]).to(self.device)

        # Double DQN: online net picks action, target net evaluates it
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            td_target    = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._sync_target()

        return float(loss.item())

    def save(self, path: str):
        torch.save({
            'model':        self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'epsilon':      self.epsilon,
            'train_step':   self.train_step,
        }, str(path) + '.pt')

    def load(self, path: str):
        pt_path = str(path) + '.pt'
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model'])
        self.target_model.load_state_dict(ckpt['target_model'])
        self.epsilon    = ckpt.get('epsilon', self.epsilon)
        self.train_step = ckpt.get('train_step', 0)
