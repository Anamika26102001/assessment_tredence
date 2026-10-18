"""Weight initialisers.

He (Kaiming) normal, std = sqrt(2 / fan_in), for layers followed by ReLU: ReLU zeroes half
of a symmetric pre-activation, halving E[x^2], so doubling the weight variance keeps the
second moment of activations constant with depth (no vanishing/exploding signal).

Glorot (Xavier) normal, std = sqrt(2 / (fan_in + fan_out)), for the logits layer, which has
no ReLU after it: it balances forward and backward variance.
"""

from __future__ import annotations

import numpy as np


def he_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype: type) -> np.ndarray:
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(dtype)


def glorot_normal(rng: np.random.Generator, fan_in: int, fan_out: int, dtype: type) -> np.ndarray:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return (rng.standard_normal((fan_in, fan_out)) * std).astype(dtype)
