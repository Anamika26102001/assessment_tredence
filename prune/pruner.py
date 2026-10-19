"""Self-pruning during training: schedule-driven global pruning with optional regrowth.

Plugs into the training loop as a callback. At each scheduled event:

1. PRUNE to the scheduled sparsity: keep the ``n_keep`` highest-scoring *alive* weights
   across all prunable layers (global ranking; stable sort so ties break by index and runs
   are reproducible). The achieved count is asserted to equal the target exactly.
2. REGROW (optional, RigL-style): swap out the ``k`` lowest-scoring alive weights for the
   ``k`` dead weights with the largest RMS gradient of the *dense* (unmasked) loss, as read
   from the W_eff node. Sparsity is unchanged. ``k`` decays with a cosine to 0 by the end of
   the ramp so the mask converges and the final fine-tune phase is stable.

Safety rails (applied identically to every criterion):
* per-layer floor: each layer keeps at least ``floor_frac`` of its weights, so global
  ranking cannot disconnect the network;
* grace period: a just-regrown weight starts at exactly 0 and would score ~0 at the next
  event, so it is protected from being dropped for ``grace_events`` events;
* weights pruned or swapped out in an event cannot be regrown in the same event.

Pruned and regrown entries go through ``Parameter.set_mask(..., optimizer)``: weight set
to exact 0 and Adam m / v / t reset, so a revived connection trains like a fresh one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from nn.module import Module, Parameter
from optim.base import Optimizer
from prune.criteria import Criterion
from prune.schedule import CubicSchedule, OneShotSchedule


class Pruner:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: Criterion,
        schedule: CubicSchedule | OneShotSchedule,
        regrow_frac: float = 0.0,
        grace_events: int = 2,
        floor_frac: float = 0.01,
        rms_beta: float = 0.9,
    ) -> None:
        self.params: list[Parameter] = model.prunable_parameters()
        self.opt = optimizer
        self.criterion = criterion
        self.schedule = schedule
        self.regrow_frac = regrow_frac
        self.grace_events = grace_events
        self.floor_frac = floor_frac
        self.rms_beta = rms_beta
        self.total = sum(p.data.size for p in self.params)
        self.events_at = set(schedule.events)
        self.interval = getattr(schedule, "interval", 1)
        self.born = [np.full(p.shape, -(10**12), dtype=np.int64) for p in self.params]
        self.dense_g2 = [np.zeros_like(p.data) for p in self.params]
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ state
    def n_alive(self) -> int:
        return int(sum(p.alive().sum() for p in self.params))

    def sparsity(self) -> float:
        return 1.0 - self.n_alive() / self.total

    def layer_sparsity(self) -> list[float]:
        return [1.0 - float(p.alive().mean()) for p in self.params]

    # -------------------------------------------------------------- callbacks
    def after_backward(self, step: int) -> None:
        self.criterion.update(self.params)
        if self.regrow_frac > 0:
            b = self.rms_beta
            for p, acc in zip(self.params, self.dense_g2, strict=True):
                g = p.dense_grad()
                if g is not None:
                    acc *= b
                    acc += (1 - b) * g * g

    def after_step(self, step: int) -> None:
        n_done = step + 1
        if n_done in self.events_at:
            self.event(n_done)

    def on_epoch_end(self, epoch: int, record: dict) -> None:
        record["sparsity"] = self.sparsity()
        record["layer_sparsity"] = self.layer_sparsity()

    # ----------------------------------------------------------------- events
    def _flat(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        return np.concatenate([a.ravel() for a in arrays])

    def _forced(self, scores: list[np.ndarray], n_done: int, protect: bool) -> np.ndarray:
        """Alive entries that must survive: per-layer floor (+ grace-protected regrown)."""
        forced = []
        for p, s, born in zip(self.params, scores, self.born, strict=True):
            alive = p.alive()
            f = np.zeros(p.shape, dtype=bool)
            floor_k = min(math.ceil(self.floor_frac * p.data.size), int(alive.sum()))
            if floor_k > 0:
                key = np.where(alive, s, -np.inf).ravel()
                top = np.argsort(-key, kind="stable")[:floor_k]
                f.ravel()[top] = True
            if protect:
                f |= alive & (n_done - born <= self.grace_events * self.interval)
            forced.append(f)
        return self._flat(forced)

    def event(self, n_done: int) -> None:
        target = self.schedule.sparsity_at(n_done)
        scores = self.criterion.scores(self.params)
        before = self.n_alive()
        alive_before = self._flat([p.alive() for p in self.params])
        n_dropped = self._prune(target, scores, n_done)
        just_pruned = alive_before & ~self._flat([p.alive() for p in self.params])
        n_grown = self._regrow(n_done, scores, just_pruned) if self.regrow_frac > 0 else 0
        achieved = self.sparsity()
        n_keep = self.total - round(self.total * target)
        if self.n_alive() > before or self.n_alive() != min(before, n_keep):
            raise AssertionError(
                f"budget violated: alive {self.n_alive()}, expected {min(before, n_keep)}"
            )
        self.log.append(
            {
                "step": n_done,
                "target_sparsity": target,
                "achieved_sparsity": achieved,
                "pruned": n_dropped,
                "regrown": n_grown,
                "layer_sparsity": self.layer_sparsity(),
            }
        )

    def _prune(self, target: float, scores: list[np.ndarray], n_done: int) -> int:
        n_alive = self.n_alive()
        n_keep = self.total - round(self.total * target)
        if n_keep >= n_alive:
            return 0
        alive = self._flat([p.alive() for p in self.params])
        key = np.where(alive, self._flat(scores), -np.inf)
        forced = self._forced(scores, n_done, protect=True)
        if int(forced.sum()) > n_keep:  # protection would break the budget: floors only
            forced = self._forced(scores, n_done, protect=False)
        if int(forced.sum()) > n_keep:
            raise ValueError("per-layer floors exceed the sparsity budget; lower floor_frac")
        key = np.where(forced, np.inf, key)
        order = np.argsort(-key, kind="stable")
        keep = np.zeros(self.total, dtype=bool)
        keep[order[:n_keep]] = True
        assert not (keep & ~alive).any()
        self._install(keep, n_done, grown=None)
        return n_alive - n_keep

    def _regrow(
        self, n_done: int, scores: list[np.ndarray], just_pruned: np.ndarray | None = None
    ) -> int:
        frac = self.regrow_frac * 0.5 * (1 + math.cos(math.pi * self.schedule.progress(n_done)))
        k = round(frac * self.n_alive())
        alive = self._flat([p.alive() for p in self.params])
        candidates = ~alive if just_pruned is None else ~alive & ~just_pruned
        k = min(k, int(candidates.sum()))
        if k <= 0:
            return 0
        forced = self._forced(scores, n_done, protect=True)
        droppable = alive & ~forced
        k = min(k, int(droppable.sum()))
        if k <= 0:
            return 0
        drop_key = np.where(droppable, self._flat(scores), np.inf)
        drop = np.argsort(drop_key, kind="stable")[:k]
        grow_key = np.where(candidates, np.sqrt(self._flat(self.dense_g2)), -np.inf)
        grow = np.argsort(-grow_key, kind="stable")[:k]
        keep = alive.copy()
        keep[drop] = False
        grown = np.zeros(self.total, dtype=bool)
        grown[grow] = True
        keep[grow] = True
        self._install(keep, n_done, grown=grown)
        return k

    def _install(self, keep: np.ndarray, n_done: int, grown: np.ndarray | None) -> None:
        off = 0
        for i, p in enumerate(self.params):
            sl = slice(off, off + p.data.size)
            off += p.data.size
            new = keep[sl].reshape(p.shape)
            p.set_mask(new, self.opt)
            if grown is not None:
                g = grown[sl].reshape(p.shape)
                self.born[i][g] = n_done
                self.criterion.reset_entries(p, g)
                self.dense_g2[i][g] = 0.0
