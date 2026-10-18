"""Fused numerically-stable losses."""

from __future__ import annotations

import numpy as np

from engine.ops import _node
from engine.tensor import Tensor


def log_softmax_np(z: np.ndarray) -> np.ndarray:
    """log-sum-exp with the row max subtracted: exp() only sees values <= 0."""
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean cross-entropy of softmax(logits) against integer class ``labels``.

    Fused because the combined gradient is simply (softmax - onehot) / N, which avoids
    differentiating through log(softmax) where softmax can underflow to 0.
    """
    n = logits.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    logp = log_softmax_np(logits.data)
    loss = -logp[np.arange(n), labels].mean()

    def bw(g: np.ndarray) -> None:
        if logits.requires_grad:
            d = np.exp(logp)
            d[np.arange(n), labels] -= 1.0
            logits._accum(d * (g / n))

    return _node(np.asarray(loss, dtype=logits.data.dtype), (logits,), "softmax_ce", bw)
