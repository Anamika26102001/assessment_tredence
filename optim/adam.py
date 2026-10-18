"""Adam with per-element state, correct under a changing mask.

Plain Adam keeps one global step counter ``t`` and updates ``m``/``v`` everywhere. Under
pruning-and-regrowth that is wrong in three ways:

1. Stale moments: a pruned-then-revived weight would inherit ``m``/``v`` from before it
   was pruned and be pushed in an old direction.
2. Moment decay on masked entries (gradient is 0, so ``m`` decays and the weight would
   keep drifting if it were not frozen).
3. Bias correction: a revived entry has fresh (zero) moments but a large global ``t``, so
   the bias-correction terms are ~1 and its first step is
   ``(1-b1) g / sqrt((1-b2) g^2)`` (about 3.16 * lr once ``t`` is in the thousands)
   instead of ``lr``.

Here ``t`` is an integer array with one counter per weight. Masked entries are frozen
(``m``, ``v``, ``t`` and the weight do not move) and ``reset_entries`` zeroes all three
whenever an entry is pruned or revived, so a revived weight behaves exactly like a fresh
parameter: its first step is ``-lr * g / (|g| + eps)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nn.module import Parameter
from optim.base import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params)
        self.lr, self.betas, self.eps = lr, betas, eps
        self.m = {id(p): np.zeros_like(p.data) for p in self.params}
        self.v = {id(p): np.zeros_like(p.data) for p in self.params}
        self.t = {id(p): np.zeros(p.shape, dtype=np.int64) for p in self.params}

    def step(self) -> None:
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            k = id(p)
            alive = p.mask  # None means every entry is alive
            g = p.grad
            m, v, t = self.m[k], self.v[k], self.t[k]
            if alive is None:
                t += 1
                m *= b1
                m += (1 - b1) * g
                v *= b2
                v += (1 - b2) * g * g
                t_f = t.astype(p.data.dtype)
                upd = (m / (1 - b1**t_f)) / (np.sqrt(v / (1 - b2**t_f)) + self.eps)
                p.data -= self.lr * upd
            else:
                t += alive
                m[...] = np.where(alive, b1 * m + (1 - b1) * g, m)
                v[...] = np.where(alive, b2 * v + (1 - b2) * g * g, v)
                t_f = np.maximum(t, 1).astype(p.data.dtype)  # t=0 only where masked
                upd = (m / (1 - b1**t_f)) / (np.sqrt(v / (1 - b2**t_f)) + self.eps)
                p.data -= np.where(alive, self.lr * upd, 0.0).astype(p.data.dtype)

    def reset_entries(self, param: Parameter, entries: np.ndarray) -> None:
        k = id(param)
        self.m[k][entries] = 0.0
        self.v[k][entries] = 0.0
        self.t[k][entries] = 0
