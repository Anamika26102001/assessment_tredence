from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from engine import Tensor, ops
from nn.init import glorot_normal, he_normal
from nn.module import Module, Parameter


class Linear(Module):
    """y = x @ W_eff + b. Only the weight matrix is prunable; biases are never masked."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init: str = "he",
        dtype: type = np.float32,
    ) -> None:
        init_fn = he_normal if init == "he" else glorot_normal
        self.weight = Parameter(init_fn(rng, in_features, out_features, dtype), prunable=True)
        self.bias = Parameter(np.zeros(out_features, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.effective() + self.bias


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def mlp(
    sizes: Sequence[int],
    rng: np.random.Generator,
    activation: Callable[[], Module] = ReLU,
    dtype: type = np.float32,
) -> Sequential:
    """MLP with He init before each activation and Glorot init on the logits layer."""
    layers: list[Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        last = i == len(sizes) - 2
        layers.append(Linear(a, b, rng, init="glorot" if last else "he", dtype=dtype))
        if not last:
            layers.append(activation())
    return Sequential(*layers)
