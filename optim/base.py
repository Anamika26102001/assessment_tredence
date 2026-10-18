from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nn.module import Parameter


class Optimizer:
    def __init__(self, params: Sequence[Parameter]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def reset_entries(self, param: Parameter, entries: np.ndarray) -> None:
        """Forget all optimizer history for ``entries`` of ``param`` (prune / revive)."""
        raise NotImplementedError
