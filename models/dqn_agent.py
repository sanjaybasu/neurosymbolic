"""
Simple Deep Q-Network agent for Waymark tabular/state-vector data.

This is a minimal, standard DQN implementation:
- MLP Q-network
- Target network updates
- ε-greedy exploration
- Replay buffer

Intended as a drop-in baseline to train on observations_* parquet files.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Feedforward MLP for Q-value estimation."""

    def __init__(self, state_dim: int, n_actions: int, hidden: List[int] = None):
        super().__init__()
        hidden = hidden or [256, 128]
        layers: List[nn.Module] = []
        dims = [state_dim] + hidden
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer: int = 1_000
    target_update: int = 1_000  # steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    hidden: Optional[List[int]] = None
    device: str = "cpu"


class ReplayBuffer:
    """Fixed-size replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Standard DQN with target network and epsilon-greedy policy."""

    def __init__(self, config: DQNConfig):
        self.cfg = config
        self.q_net = QNetwork(config.state_dim, config.n_actions, config.hidden).to(config.device)
        self.target_net = QNetwork(config.state_dim, config.n_actions, config.hidden).to(config.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.steps = 0
        self.epsilon = config.epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        self.steps += 1
        self._decay_epsilon()
        if random.random() < self.epsilon:
            return random.randrange(self.cfg.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.cfg.device)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values).item())

    def _decay_epsilon(self):
        frac = min(1.0, self.steps / self.cfg.epsilon_decay_steps)
        self.epsilon = self.cfg.epsilon_start - frac * (self.cfg.epsilon_start - self.cfg.epsilon_end)

    def push(self, transition):
        self.buffer.push(transition)

    def update(self) -> Optional[float]:
        """One gradient step from replay buffer."""
        if len(self.buffer) < self.cfg.min_buffer:
            return None
        state, action, reward, next_state, done = self.buffer.sample(self.cfg.batch_size)

        device = self.cfg.device
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        action_t = torch.tensor(action, dtype=torch.int64, device=device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
        done_t = torch.tensor(done, dtype=torch.float32, device=device)

        # Q(s,a)
        q_values = self.q_net(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_state_t).max(1)[0]
            target = reward_t + self.cfg.gamma * next_q * (1 - done_t)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target net
        if self.steps % self.cfg.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.q_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["model_state_dict"]))
        if "config" in ckpt:
            self.cfg = DQNConfig(**ckpt["config"])
