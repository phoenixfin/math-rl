"""
DQN Agent — Double DQN with action masking and experience replay.
"""

import numpy as np
import tensorflow as tf
keras = tf.keras
from collections import deque
from typing import List, Optional
import random


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

        self.model        = self._build(lr)
        self.target_model = self._build(lr)
        self._sync_target()

    def _build(self, lr: float):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='huber'
        )
        return model

    def _sync_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray,
            valid_actions: Optional[List[int]] = None) -> int:
        if np.random.random() < self.epsilon:
            pool = valid_actions if (valid_actions and np.random.random() < 0.8) \
                   else list(range(self.action_size))
            return random.choice(pool)

        q = self.model(state[np.newaxis], training=False).numpy()[0]

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
        states      = np.array([e[0] for e in batch], dtype=np.float32)
        actions     = np.array([e[1] for e in batch], dtype=np.int32)
        rewards     = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones       = np.array([e[4] for e in batch], dtype=np.float32)

        next_q_online = self.model(next_states, training=False).numpy()
        next_q_target = self.target_model(next_states, training=False).numpy()
        best_next     = np.argmax(next_q_online, axis=1)

        td_targets = self.model(states, training=False).numpy()
        for i in range(self.batch_size):
            td = rewards[i]
            if not dones[i]:
                td += self.gamma * next_q_target[i, best_next[i]]
            td_targets[i, actions[i]] = td

        history = self.model.fit(states, td_targets,
                                 epochs=1, verbose=0,
                                 batch_size=self.batch_size)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self._sync_target()

        return float(loss)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = keras.models.load_model(path)
        self._sync_target()
