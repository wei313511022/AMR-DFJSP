from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class ReplayTransition:
    state: np.ndarray
    action_index: int
    action_feat: np.ndarray
    action_feats: np.ndarray
    reward: float
    next_state: Optional[np.ndarray]
    next_action_feats: np.ndarray
    done: bool
    discount: float


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.buffer: deque[ReplayTransition] = deque()

    def append(self, transition: ReplayTransition) -> List[ReplayTransition]:
        self.buffer.append(transition)
        ready: List[ReplayTransition] = []

        if transition.done:
            ready.extend(self.flush())
            return ready

        if len(self.buffer) >= self.n_step:
            ready.append(self._build_transition())
            self.buffer.popleft()

        return ready

    def flush(self) -> List[ReplayTransition]:
        ready: List[ReplayTransition] = []
        while self.buffer:
            ready.append(self._build_transition())
            self.buffer.popleft()
        return ready

    def _build_transition(self) -> ReplayTransition:
        reward = 0.0
        next_state: Optional[np.ndarray] = None
        next_action_feats = np.zeros((0, 4), dtype=np.float32)
        done = False
        steps = 0

        for i, item in enumerate(self.buffer):
            reward += (self.gamma**i) * float(item.reward)
            next_state = item.next_state
            next_action_feats = item.next_action_feats
            done = bool(item.done)
            steps = i + 1
            if done or steps >= self.n_step:
                break

        first = self.buffer[0]
        return ReplayTransition(
            state=first.state,
            action_index=int(first.action_index),
            action_feat=first.action_feat,
            action_feats=first.action_feats,
            reward=float(reward),
            next_state=next_state,
            next_action_feats=next_action_feats,
            done=bool(done),
            discount=float(self.gamma**steps),
        )


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        priority_eps: float = 1e-5,
        sample_with_replacement: bool = True,
    ):
        self.capacity = max(1, int(capacity))
        self.alpha = max(0.0, float(alpha))
        self.priority_eps = float(priority_eps)
        self.sample_with_replacement = bool(sample_with_replacement)
        self.storage: List[Optional[ReplayTransition]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: ReplayTransition, priority: Optional[float] = None) -> None:
        if priority is None:
            priority = self.max_priority
        priority = max(self.priority_eps, float(priority))

        self.storage[self.pos] = transition
        self.priorities[self.pos] = priority
        self.max_priority = max(self.max_priority, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[np.ndarray, List[ReplayTransition], np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = max(1, int(batch_size))
        if not self.sample_with_replacement:
            batch_size = min(batch_size, self.size)
        probs = self._probabilities()
        indices = np.random.choice(
            self.size,
            size=batch_size,
            replace=self.sample_with_replacement,
            p=probs,
        )
        batch = [self.storage[idx] for idx in indices]
        assert all(item is not None for item in batch)

        beta = float(beta)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= max(1e-12, float(weights.max()))

        return indices, [item for item in batch if item is not None], weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        dedup_updates = {}
        for idx, prio in zip(indices, priorities):
            idx_i = int(idx)
            value = max(self.priority_eps, float(prio))
            prev = dedup_updates.get(idx_i)
            if prev is None or value > prev:
                dedup_updates[idx_i] = value

        for idx_i, value in dedup_updates.items():
            self.priorities[idx_i] = value
            self.max_priority = max(self.max_priority, value)

    def _probabilities(self) -> np.ndarray:
        active = self.priorities[: self.size]
        if self.alpha <= 0.0:
            return np.full(self.size, 1.0 / float(self.size), dtype=np.float32)

        scaled = np.power(np.maximum(active, self.priority_eps), self.alpha, dtype=np.float32)
        total = float(np.sum(scaled))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(self.size, 1.0 / float(self.size), dtype=np.float32)
        return (scaled / total).astype(np.float32)
