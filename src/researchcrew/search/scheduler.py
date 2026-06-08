from __future__ import annotations


class ExplorationScheduler:
    """Entropy-inspired piecewise schedule shifting exploration → exploitation.

    Phases:
    1. ``iterations <= explore_until``:              C = c_max
    2. ``explore_until < iterations < decay_until``: C decays linearly
    3. ``iterations >= decay_until``:                C = c_min

    Stagnation (no improvement in best Q for ``stagnation_window`` steps)
    temporarily resets C to c_max to trigger BranchFusion exploration.
    """

    def __init__(
        self,
        c_max: float = 2.0,
        c_min: float = 0.5,
        explore_until: int = 10,
        decay_until: int = 30,
        stagnation_window: int = 5,
    ) -> None:
        self.c_max = c_max
        self.c_min = c_min
        self.explore_until = explore_until
        self.decay_until = decay_until
        self.stagnation_window = stagnation_window
        self._iteration = 0
        self._best_q_history: list[float] = []
        self._stagnating = False

    def step(self, best_q: float) -> float:
        """Advance one iteration, record best Q, return current C."""
        self._iteration += 1
        self._best_q_history.append(best_q)
        if len(self._best_q_history) >= self.stagnation_window:
            window = self._best_q_history[-self.stagnation_window :]
            self._stagnating = (max(window) - min(window)) < 1e-4
        return self.c(self._iteration)

    def c(self, iteration: int) -> float:
        if self._stagnating:
            return self.c_max
        if iteration <= self.explore_until:
            return self.c_max
        if iteration >= self.decay_until:
            return self.c_min
        progress = (iteration - self.explore_until) / (
            self.decay_until - self.explore_until
        )
        return self.c_max - progress * (self.c_max - self.c_min)

    @property
    def is_stagnating(self) -> bool:
        return self._stagnating

    @property
    def iteration(self) -> int:
        return self._iteration
