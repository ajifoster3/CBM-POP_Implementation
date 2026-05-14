import random
from collections import deque
import numpy as np


class UCBBandit:
    def __init__(self, n_operators: int, c: float = 1.414, window: int = 200):
        self.c = c
        self.window = window
        # Per-operator history: each entry is (reward,)
        self._history = [deque(maxlen=window) for _ in range(n_operators)]
        self.N = 0  # global pull count, never reset

    def select(self, admissible_indices=None) -> int:
        if admissible_indices is None:
            admissible = list(range(len(self._history)))
        else:
            admissible = [int(i) for i in admissible_indices]

        if not admissible:
            raise ValueError("UCBBandit.select() requires at least one admissible operator")
        if min(admissible) < 0 or max(admissible) >= len(self._history):
            raise IndexError("admissible operator index out of range")

        untried = [i for i in admissible if len(self._history[i]) == 0]
        if untried:
            return random.choice(untried)

        means = np.array([np.mean(self._history[i]) for i in admissible])
        counts = np.array([len(self._history[i]) for i in admissible])
        bonus = self.c * np.sqrt(np.log(self.N) / counts)
        return admissible[int(np.argmax(means + bonus))]

    def update(self, op_idx: int, reward: float):
        self._history[op_idx].append(reward)
        self.N += 1
