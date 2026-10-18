"""Reverse-mode autodiff tensor.

A Tensor wraps an ndarray and, when produced by an op, remembers its parents and a
closure that pushes an upstream gradient into them. ``backward()`` walks the graph in
reverse topological order. Gradients are *accumulated* (``+=``) so a node used twice
(``x * x``, diamonds) receives the sum of both paths.

Every node keeps its ``.grad`` after backward (not only leaves): the pruner reads the
gradient of the effective (masked) weight node as a "what if this connection existed"
signal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (inference / evaluation)."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``: the adjoint of NumPy broadcasting.

    Broadcasting copies a value along an axis, so its gradient is the sum over that axis.
    Two cases: leading axes that were prepended, and axes where the operand had size 1.
    """
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad


class Tensor:
    __array_priority__ = 1000  # make ndarray defer to Tensor's reflected operators

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        _parents: Sequence[Tensor] = (),
        op: str = "",
    ) -> None:
        arr = np.asarray(data)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents: tuple[Tensor, ...] = tuple(_parents)
        self._backward: Callable[[np.ndarray], None] | None = None
        self.op = op

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self.op or 'leaf'}, grad={self.requires_grad})"

    # ---------------------------------------------------------------- backward
    def _accum(self, g: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype, copy=True)
        else:
            self.grad += g

    def _topo(self) -> list[Tensor]:
        """Post-order DFS, iterative (deep graphs must not hit the recursion limit)."""
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        return order

    def backward(self, grad: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar output")
            grad = np.ones_like(self.data)
        self._accum(np.asarray(grad))
        for node in reversed(self._topo()):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    # --------------------------------------------------------------- operators
    # ops imports Tensor, so it is imported lazily inside each method.
    def __add__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.add(self, other)

    def __radd__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.add(other, self)

    def __sub__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.mul(self, other)

    def __rmul__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.mul(other, self)

    def __truediv__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.div(other, self)

    def __neg__(self) -> Tensor:
        from engine import ops

        return ops.neg(self)

    def __pow__(self, p: float) -> Tensor:
        from engine import ops

        return ops.pow(self, p)

    def __matmul__(self, other: Any) -> Tensor:
        from engine import ops

        return ops.matmul(self, other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        from engine import ops

        return ops.sum(self, axis, keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        from engine import ops

        return ops.mean(self, axis, keepdims)
