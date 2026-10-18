"""Parameters with an optional connectivity mask, and a minimal Module base.

Masking semantics (the core correctness point):

* The forward pass uses ``W_eff = W * M`` where ``M`` is a constant 0/1 array.
* ``dL/dW = dL/dW_eff * M`` -- the true derivative of ``L(W * M)`` w.r.t. ``W``. A masked
  entry therefore gets an *exact* zero gradient, which agrees with finite differences
  (perturbing a masked weight cannot change the loss).
* The unmasked gradient ``dL/dW_eff`` is kept on the ``W_eff`` node (``_eff.grad``) so
  the pruner can ask "how much would the loss drop if this connection existed?". It is
  never given to an optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from engine import Tensor, ops

if TYPE_CHECKING:
    from optim.base import Optimizer


class Parameter(Tensor):
    def __init__(self, data: Any, prunable: bool = False) -> None:
        super().__init__(data, requires_grad=True)
        self.prunable = prunable
        self.mask: np.ndarray | None = None  # bool, True = connection alive
        self._mask_f: Tensor | None = None  # cached 0/1 constant in param dtype
        self._eff: Tensor | None = None  # W_eff node of the last forward pass

    def effective(self) -> Tensor:
        """The tensor the forward pass should use: ``W`` or ``W * M``."""
        if self.mask is None or self._mask_f is None:
            self._eff = None
            return self
        self._eff = ops.mul(self, self._mask_f)
        return self._eff

    def dense_grad(self) -> np.ndarray | None:
        """dL/dW_eff from the last backward pass (nonzero even at masked entries)."""
        if self._eff is None:
            return self.grad
        return self._eff.grad

    def alive(self) -> np.ndarray:
        return self.mask if self.mask is not None else np.ones(self.shape, dtype=bool)

    def set_mask(self, new_mask: np.ndarray, optimizer: Optimizer | None = None) -> None:
        """Install a new mask. Pruned *and* revived entries are zeroed and their optimizer
        state is reset, so a revived connection starts exactly like a fresh weight."""
        new_mask = np.asarray(new_mask, dtype=bool)
        if new_mask.shape != self.shape:
            raise ValueError(f"mask shape {new_mask.shape} != param shape {self.shape}")
        changed = self.alive() != new_mask
        self.data[~new_mask] = 0.0  # a real zero, not a small number
        self.data[new_mask & changed] = 0.0  # revived: starts from zero
        self.mask = new_mask.copy()
        self._mask_f = Tensor(self.mask.astype(self.data.dtype))
        if optimizer is not None:
            optimizer.reset_entries(self, changed)


class Module:
    def parameters(self) -> list[Parameter]:
        out: list[Parameter] = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, list | tuple):
                for item in v:
                    if isinstance(item, Module):
                        out.extend(item.parameters())
        return out

    def prunable_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if p.prunable]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
