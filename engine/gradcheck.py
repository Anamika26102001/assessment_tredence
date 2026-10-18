"""Finite-difference gradient checking (central differences, float64)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from engine import ops
from engine.tensor import Tensor, no_grad


def _scalarize(out: Tensor, proj: np.ndarray) -> Tensor:
    # Random projection turns any output into a scalar while exercising every element.
    return ops.sum(ops.mul(out, Tensor(proj)))


def numerical_grad(
    f: Callable[..., Tensor], arrays: Sequence[np.ndarray], idx: int, proj: np.ndarray, eps: float
) -> np.ndarray:
    x = arrays[idx]
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        with no_grad():
            hi = _scalarize(f(*[Tensor(a) for a in arrays]), proj).item()
        x[i] = old - eps
        with no_grad():
            lo = _scalarize(f(*[Tensor(a) for a in arrays]), proj).item()
        x[i] = old
        num[i] = (hi - lo) / (2 * eps)
    return num


def gradcheck(
    f: Callable[..., Tensor],
    inputs: Sequence[np.ndarray],
    eps: float = 1e-6,
    tol: float = 1e-6,
    seed: int = 0,
) -> float:
    """Compare analytic and numeric gradients of ``f`` w.r.t. every input.

    Returns the worst relative error; raises AssertionError above ``tol``.
    """
    arrays = [np.array(a, dtype=np.float64) for a in inputs]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = f(*tensors)
    proj = np.random.default_rng(seed).standard_normal(out.shape)
    _scalarize(out, proj).backward()

    worst = 0.0
    for k, t in enumerate(tensors):
        analytic = t.grad if t.grad is not None else np.zeros_like(t.data)
        numeric = numerical_grad(f, arrays, k, proj, eps)
        err = np.abs(analytic - numeric).max() / max(
            1.0, np.abs(analytic).max(), np.abs(numeric).max()
        )
        worst = max(worst, float(err))
    if worst > tol:
        raise AssertionError(f"gradcheck failed: max relative error {worst:.3e} > {tol:.0e}")
    return worst
