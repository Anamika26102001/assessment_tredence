"""Sparsity schedules, in units of completed optimizer steps.

Cubic ramp (Zhu & Gupta 2017):  s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (n * dt))**3

The derivative is largest at t0 and goes to zero at the end. Early on the network is
heavily over-parameterised, so large cuts are cheap; near the target every cut removes
something useful and the net needs many steps to recover, so the ramp flattens out. One
shot at the end (the weaker alternative) forces the whole accuracy hit in a single
step with no chance to recover or to re-rank weights as the others adapt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CubicSchedule:
    final_sparsity: float
    start_step: int
    end_step: int
    interval: int
    initial_sparsity: float = 0.0

    @property
    def n_events(self) -> int:
        return max(1, (self.end_step - self.start_step) // self.interval)

    @property
    def events(self) -> list[int]:
        return [self.start_step + j * self.interval for j in range(self.n_events + 1)]

    def progress(self, step: int) -> float:
        span = self.n_events * self.interval
        return min(1.0, max(0.0, (step - self.start_step) / span))

    def sparsity_at(self, step: int) -> float:
        f = self.progress(step)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1 - f) ** 3


@dataclass(frozen=True)
class OneShotSchedule:
    final_sparsity: float
    step: int

    @property
    def events(self) -> list[int]:
        return [self.step]

    def progress(self, step: int) -> float:
        return 1.0

    def sparsity_at(self, step: int) -> float:
        return self.final_sparsity if step >= self.step else 0.0
