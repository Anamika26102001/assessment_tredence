"""Sparse-aware inference paths, written from scratch (no scipy).

Three ways to run a pruned MLP, only two of which are real savings:

* dense x zeros   -- ``x @ W`` with zeros left in W. Same FLOPs and (up to cache effects)
                     same time as the unpruned model. NOT a speedup; benchmarked only as
                     the honest reference point.
* CSR             -- work proportional to the number of alive weights (``2 * nnz`` FLOPs
                     per sample); no multiply by zero is ever executed.
* compaction      -- delete whole neurons / input features that the mask disconnected and
                     run smaller *dense* matmuls on BLAS. Exact: a unit with no inputs
                     outputs the constant ``act(b)``, folded into the next layer's bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nn.layers import Linear, ReLU, Tanh
from nn.module import Module


def _act(name: str | None, x: np.ndarray) -> np.ndarray:
    if name is None:
        return x
    if name == "relu":
        return np.maximum(x, 0)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(name)


def _layers_of(model: Module) -> list[tuple[Linear, str | None]]:
    """(Linear, activation-after-it) pairs of a Sequential MLP."""
    seq = getattr(model, "layers")  # noqa: B009
    out: list[tuple[Linear, str | None]] = []
    for layer in seq:
        if isinstance(layer, Linear):
            out.append((layer, None))
        elif isinstance(layer, ReLU):
            out[-1] = (out[-1][0], "relu")
        elif isinstance(layer, Tanh):
            out[-1] = (out[-1][0], "tanh")
        else:
            raise TypeError(f"unsupported layer {type(layer).__name__}")
    return out


@dataclass
class CSR:
    """Compressed sparse rows of W^T: row j lists the inputs feeding output unit j."""

    n_in: int
    n_out: int
    indptr: np.ndarray  # (n_out + 1,)
    indices: np.ndarray  # (nnz,) input index of each stored weight
    data: np.ndarray  # (nnz,)

    @classmethod
    def from_dense(cls, w: np.ndarray, alive: np.ndarray | None = None) -> CSR:
        alive = (w != 0) if alive is None else alive
        rows, cols = np.nonzero(alive.T)  # row-major over (out, in): sorted by output unit
        counts = np.bincount(rows, minlength=w.shape[1])
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(w.shape[0], w.shape[1], indptr, cols.astype(np.int64), w.T[rows, cols].copy())

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """x @ W for x of shape (B, n_in); only alive weights are multiplied."""
        out = np.zeros((x.shape[0], self.n_out), dtype=x.dtype)
        if self.nnz == 0:
            return out
        nonempty = np.flatnonzero(np.diff(self.indptr) > 0)
        gathered = x[:, self.indices] * self.data  # (B, nnz): one multiply per alive weight
        # reduceat sums each contiguous segment; empty rows would alias a neighbour, so
        # only non-empty rows are reduced and the rest stay exactly 0.
        out[:, nonempty] = np.add.reduceat(gathered, self.indptr[nonempty], axis=1)
        return out


class SparseMLP:
    """MLP whose layers are CSR matrices."""

    def __init__(self, model: Module) -> None:
        self.layers: list[tuple[CSR, np.ndarray, str | None]] = []
        for lin, act in _layers_of(model):
            w = lin.weight
            self.layers.append((CSR.from_dense(w.data, w.alive()), lin.bias.data.copy(), act))

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for csr, b, act in self.layers:
            h = _act(act, csr.matmul(h) + b)
        return h

    @property
    def n_active_params(self) -> int:
        return sum(c.nnz + b.size for c, b, _ in self.layers)

    @property
    def flops_per_sample(self) -> int:
        """2 FLOPs per alive weight (multiply + add) plus bias adds."""
        return sum(2 * c.nnz + b.size for c, b, _ in self.layers)


class CompactMLP:
    """Dense MLP with disconnected neurons / input features physically removed."""

    def __init__(
        self,
        input_idx: np.ndarray,
        layers: list[tuple[np.ndarray, np.ndarray, str | None]],
        n_in: int,
    ) -> None:
        self.input_idx, self.layers, self.n_in = input_idx, layers, n_in

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x if len(self.input_idx) == self.n_in else x[:, self.input_idx]
        for w, b, act in self.layers:
            h = _act(act, h @ w + b)
        return h

    @property
    def shapes(self) -> list[tuple[int, int]]:
        return [w.shape for w, _, _ in self.layers]

    @property
    def flops_per_sample(self) -> int:
        """Dense matmul FLOPs of the *compacted* matrices (zeros inside them still count)."""
        return sum(2 * w.size + b.size for w, b, _ in self.layers)


def compact(model: Module) -> CompactMLP:
    """Remove neurons the mask has disconnected, exactly (no change to the function).

    * input feature with no outgoing weight        -> never read: drop it
    * hidden unit with no outgoing weight          -> output unused: drop it
    * hidden unit with no incoming weight          -> constant act(b): fold into next
      layer's bias, then drop it
    Dropping units can disconnect others, so repeat until nothing changes. Output (logit)
    units are never removed.
    """
    pairs = _layers_of(model)
    ws = [(lin.weight.data * lin.weight.alive()).copy() for lin, _ in pairs]
    bs = [lin.bias.data.copy() for lin, _ in pairs]
    acts = [a for _, a in pairs]
    n_in = ws[0].shape[0]
    input_idx = np.arange(n_in)
    changed = True
    while changed:
        changed = False
        used = (ws[0] != 0).any(axis=1)
        if not used.all():
            ws[0], input_idx, changed = ws[0][used], input_idx[used], True
        for layer in range(len(ws) - 1):
            no_in = ~(ws[layer] != 0).any(axis=0)
            no_out = ~(ws[layer + 1] != 0).any(axis=1)
            dead = no_in | no_out
            if not dead.any():
                continue
            fold = no_in & ~no_out
            if fold.any():
                const = _act(acts[layer], bs[layer][fold])
                bs[layer + 1] = bs[layer + 1] + const @ ws[layer + 1][fold, :]
            keep = ~dead
            ws[layer], bs[layer] = ws[layer][:, keep], bs[layer][keep]
            ws[layer + 1] = ws[layer + 1][keep, :]
            changed = True
    return CompactMLP(input_idx, list(zip(ws, bs, acts, strict=True)), n_in)


def dense_flops_per_sample(model: Module) -> int:
    return sum(2 * lin.weight.data.size + lin.bias.data.size for lin, _ in _layers_of(model))
