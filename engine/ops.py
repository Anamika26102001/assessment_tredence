"""Differentiable primitives. Each op builds a node and a closure mapping the upstream
gradient to a gradient contribution for every parent (via ``_accum``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from engine.tensor import Tensor, is_grad_enabled, unbroadcast

Axis = int | tuple[int, ...] | None


def _t(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _node(
    data: np.ndarray,
    parents: tuple[Tensor, ...],
    op: str,
    backward: Callable[[np.ndarray], None],
) -> Tensor:
    """Attach graph info only if grad is enabled and some parent needs grad."""
    if not is_grad_enabled() or not any(p.requires_grad for p in parents):
        return Tensor(data, op=op)
    out = Tensor(data, requires_grad=True, _parents=parents, op=op)
    out._backward = backward
    return out


def _push(p: Tensor, g: np.ndarray) -> None:
    if p.requires_grad:
        p._accum(unbroadcast(g, p.shape))


# ------------------------------------------------------------------ elementwise
def add(a: Any, b: Any) -> Tensor:
    a, b = _t(a), _t(b)

    def bw(g: np.ndarray) -> None:
        _push(a, g)
        _push(b, g)

    return _node(a.data + b.data, (a, b), "add", bw)


def sub(a: Any, b: Any) -> Tensor:
    a, b = _t(a), _t(b)

    def bw(g: np.ndarray) -> None:
        _push(a, g)
        _push(b, -g)

    return _node(a.data - b.data, (a, b), "sub", bw)


def mul(a: Any, b: Any) -> Tensor:
    a, b = _t(a), _t(b)

    def bw(g: np.ndarray) -> None:
        _push(a, g * b.data)
        _push(b, g * a.data)

    return _node(a.data * b.data, (a, b), "mul", bw)


def div(a: Any, b: Any) -> Tensor:
    a, b = _t(a), _t(b)

    def bw(g: np.ndarray) -> None:
        _push(a, g / b.data)
        _push(b, -g * a.data / (b.data * b.data))

    return _node(a.data / b.data, (a, b), "div", bw)


def neg(a: Any) -> Tensor:
    a = _t(a)
    return _node(-a.data, (a,), "neg", lambda g: _push(a, -g))


def pow(a: Any, p: float) -> Tensor:  # noqa: A001
    """a ** p for a constant scalar exponent."""
    a = _t(a)
    return _node(a.data**p, (a,), "pow", lambda g: _push(a, g * p * a.data ** (p - 1)))


def exp(a: Any) -> Tensor:
    a = _t(a)
    out = np.exp(a.data)
    return _node(out, (a,), "exp", lambda g: _push(a, g * out))


def log(a: Any) -> Tensor:
    a = _t(a)
    return _node(np.log(a.data), (a,), "log", lambda g: _push(a, g / a.data))


# --------------------------------------------------------------- non-linearities
def relu(a: Any) -> Tensor:
    a = _t(a)
    pos = a.data > 0  # subgradient at 0 is 0 (the common convention)
    return _node(a.data * pos, (a,), "relu", lambda g: _push(a, g * pos))


def tanh(a: Any) -> Tensor:
    a = _t(a)
    out = np.tanh(a.data)
    return _node(out, (a,), "tanh", lambda g: _push(a, g * (1.0 - out * out)))


def sigmoid(a: Any) -> Tensor:
    a = _t(a)
    x = a.data
    # Branch on sign so exp() never sees a large positive argument.
    e = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(x.dtype)
    return _node(out, (a,), "sigmoid", lambda g: _push(a, g * out * (1.0 - out)))


# ---------------------------------------------------------------------- matmul
def matmul(a: Any, b: Any) -> Tensor:
    """2-D @ 2-D. dA = G B^T, dB = A^T G."""
    a, b = _t(a), _t(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul expects 2-D operands, got {a.shape} @ {b.shape}")

    def bw(g: np.ndarray) -> None:
        if a.requires_grad:
            a._accum(g @ b.data.T)
        if b.requires_grad:
            b._accum(a.data.T @ g)

    return _node(a.data @ b.data, (a, b), "matmul", bw)


# ------------------------------------------------------------------ reductions
def _expand(g: np.ndarray, shape: tuple[int, ...], axis: Axis, keepdims: bool) -> np.ndarray:
    """Re-insert reduced axes so ``g`` broadcasts back to ``shape``."""
    if axis is None or keepdims:
        return g if keepdims else np.reshape(g, (1,) * len(shape))
    axes = (axis,) if isinstance(axis, int) else axis
    axes = tuple(ax % len(shape) for ax in axes)
    for ax in sorted(axes):
        g = np.expand_dims(g, ax)
    return g


def sum(a: Any, axis: Axis = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    a = _t(a)

    def bw(g: np.ndarray) -> None:
        _push(a, np.broadcast_to(_expand(g, a.shape, axis, keepdims), a.shape))

    return _node(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum", bw)


def mean(a: Any, axis: Axis = None, keepdims: bool = False) -> Tensor:
    a = _t(a)
    if axis is None:
        n = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        n = int(np.prod([a.shape[ax] for ax in axes]))

    def bw(g: np.ndarray) -> None:
        _push(a, np.broadcast_to(_expand(g, a.shape, axis, keepdims), a.shape) / n)

    return _node(a.data.mean(axis=axis, keepdims=keepdims), (a,), "mean", bw)
