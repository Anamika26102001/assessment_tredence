"""SGD with (heavy-ball) momentum; velocity is frozen on masked entries and reset on
prune / revive for the same reasons as in Adam."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nn.module import Parameter
from optim.base import Optimizer


class SGD(Optimizer):
    def __init__(
        self, params: Sequence[Parameter], lr: float = 0.01, momentum: float = 0.9
    ) -> None:
        super().__init__(params)
        self.lr, self.momentum = lr, momentum
        self.vel = {id(p): np.zeros_like(p.data) for p in self.params}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            vel = self.vel[id(p)]
            if p.mask is None:
                vel *= self.momentum
                vel += p.grad
                p.data -= self.lr * vel
            else:
                vel[...] = np.where(p.mask, self.momentum * vel + p.grad, vel)
                p.data -= np.where(p.mask, self.lr * vel, 0.0).astype(p.data.dtype)

    def reset_entries(self, param: Parameter, entries: np.ndarray) -> None:
        self.vel[id(param)][entries] = 0.0
