from collections import deque
import numpy as np


class UCBBandit:
    def __init__(self, n_operators: int, c: float = 1.414, window: int = 200):
        self.c = c
        self.window = window
        # Per-operator history: each entry is (reward,)
        self._history = [deque(maxlen=window) for _ in range(n_operators)]
        self.N = 0  # global pull count, never reset

    def select(self) -> int:
        untried = [i for i, h in enumerate(self._history) if len(h) == 0]
        if untried:
            return untried[0]

        means = np.array([np.mean(h) for h in self._history])
        counts = np.array([len(h) for h in self._history])
        bonus = self.c * np.sqrt(np.log(self.N) / counts)
        return int(np.argmax(means + bonus))

    def update(self, op_idx: int, reward: float):
        self._history[op_idx].append(reward)
        self.N += 1