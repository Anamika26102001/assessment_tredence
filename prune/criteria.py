"""Importance criteria: a score per connection, higher = more important to keep.

Why a Taylor criterion approximates "loss increase if removed"
---------------------------------------------------------------
Removing weight w_i means moving it by delta_i = -w_i. Second-order expansion:

    dL_i = L(w - w_i e_i) - L(w) ~= -g_i w_i + 1/2 H_ii w_i^2

* Magnitude pruning uses |w_i|: it ignores g and H, so it is only right if every weight
  has the same curvature, which is false (layers have different scales and fan-in).
* ``taylor1`` = |g_i w_i| is the first-order term. Near a minimum E[g] ~ 0, so on a single
  mini-batch it is dominated by gradient *noise* and says little about the true effect.
* ``fisher_taylor`` keeps the second-order term. With the empirical Fisher as a
  Hessian-diagonal proxy, H_ii ~ F_ii = E[g_i^2] (exact when the model is well-specified
  at a minimum), so  dL_i ~= 1/2 F_ii w_i^2.  This is Optimal Brain Damage with the Fisher
  diagonal. We rank by s_i = w_i^2 * EMA[g_i^2] (the constant 1/2 does not change ranks).
  Averaging g^2 over many batches (EMA) is exactly what makes it a stable estimate of
  F_ii, and it costs nothing extra: it reuses gradients the optimizer already computes.

Scores are in units of loss, so they are comparable *across layers*, which is what lets
us rank globally instead of fixing a per-layer sparsity by hand.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nn.module import Parameter


class Criterion:
    name = "base"

    def update(self, params: Sequence[Parameter]) -> None:
        """Called after every backward pass (gradients populated)."""

    def scores(self, params: Sequence[Parameter]) -> list[np.ndarray]:
        raise NotImplementedError

    def reset_entries(self, param: Parameter, entries: np.ndarray) -> None:
        """Forget statistics of entries that were just (re)grown."""


class Magnitude(Criterion):
    name = "magnitude"

    def scores(self, params: Sequence[Parameter]) -> list[np.ndarray]:
        return [np.abs(p.data).astype(np.float64) for p in params]


class Taylor1(Criterion):
    """|w * g| from the most recent mini-batch gradient (first-order Taylor term)."""

    name = "taylor1"

    def scores(self, params: Sequence[Parameter]) -> list[np.ndarray]:
        out = []
        for p in params:
            g = p.grad if p.grad is not None else np.zeros_like(p.data)
            out.append(np.abs(p.data.astype(np.float64) * g))
        return out


class FisherTaylor(Criterion):
    """w^2 * EMA[g^2]: second-order Taylor term with the empirical Fisher diagonal."""

    name = "fisher_taylor"

    def __init__(self, beta: float = 0.99) -> None:
        self.beta = beta
        self.g2: dict[int, np.ndarray] = {}

    def update(self, params: Sequence[Parameter]) -> None:
        b = self.beta
        for p in params:
            if p.grad is None:
                continue
            ema = self.g2.setdefault(id(p), np.zeros_like(p.data))
            ema *= b
            ema += (1 - b) * p.grad * p.grad

    def scores(self, params: Sequence[Parameter]) -> list[np.ndarray]:
        return [
            (p.data.astype(np.float64) ** 2) * self.g2.get(id(p), np.zeros_like(p.data))
            for p in params
        ]

    def reset_entries(self, param: Parameter, entries: np.ndarray) -> None:
        if id(param) in self.g2:
            self.g2[id(param)][entries] = 0.0


def make_criterion(name: str, beta: float = 0.99) -> Criterion:
    if name == "magnitude":
        return Magnitude()
    if name == "taylor1":
        return Taylor1()
    if name == "fisher_taylor":
        return FisherTaylor(beta)
    raise ValueError(f"unknown criterion {name!r}")
