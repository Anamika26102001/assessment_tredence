"""Mini-batch training loop with hooks, so pruning can plug in without touching it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from engine import Tensor, no_grad, softmax_cross_entropy
from nn.module import Module
from optim.base import Optimizer


class Callback(Protocol):
    def after_backward(self, step: int) -> None: ...  # grads populated, before opt.step()
    def after_step(self, step: int) -> None: ...  # after the parameter update
    def on_epoch_end(self, epoch: int, record: dict) -> None: ...


class NoCallback:
    def after_backward(self, step: int) -> None:
        pass

    def after_step(self, step: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, record: dict) -> None:
        pass


@dataclass
class History:
    epochs: list[dict] = field(default_factory=list)

    def append(self, rec: dict) -> None:
        self.epochs.append(rec)


def evaluate(
    model: Module, x: np.ndarray, y: np.ndarray, batch_size: int = 1000
) -> tuple[float, float]:
    """Mean loss and accuracy, no graph built."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for i in range(0, len(x), batch_size):
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            logits = model(Tensor(xb))
            total_loss += softmax_cross_entropy(logits, yb).item() * len(xb)
            correct += int((logits.data.argmax(axis=1) == yb).sum())
    return total_loss / len(x), correct / len(x)


def _grad_norm(model: Module) -> float:
    sq = sum(
        float((p.grad.astype(np.float64) ** 2).sum())
        for p in model.parameters()
        if p.grad is not None
    )
    return sq**0.5


def train(
    model: Module,
    opt: Optimizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int = 128,
    seed: int = 0,
    callback: Callback | None = None,
    start_epoch: int = 0,
    start_step: int = 0,
    verbose: bool = True,
) -> History:
    """Shuffle each epoch with a seeded generator, run mini-batches, log per epoch.

    Raises FloatingPointError the moment a loss is non-finite, so a diverged run can never
    be mistaken for a trained one."""
    cb = callback or NoCallback()
    rng = np.random.default_rng(seed)
    hist = History()
    step = start_step
    n = len(x_train)
    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        loss_sum, correct, gnorm_sum, nb = 0.0, 0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            logits = model(Tensor(xb))
            loss = softmax_cross_entropy(logits, yb)
            if not np.isfinite(loss.data):
                raise FloatingPointError(
                    f"non-finite loss {loss.item()} at epoch {epoch} step {step}"
                )
            loss.backward()
            cb.after_backward(step)
            gnorm_sum += _grad_norm(model)
            opt.step()
            cb.after_step(step)
            loss_sum += loss.item() * len(idx)
            correct += int((logits.data.argmax(axis=1) == yb).sum())
            nb += 1
            step += 1
        val_loss, val_acc = evaluate(model, x_val, y_val)
        rec = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": loss_sum / n,
            "train_acc": correct / n,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad_norm": gnorm_sum / nb,
            "seconds": time.perf_counter() - t0,
        }
        cb.on_epoch_end(epoch, rec)
        hist.append(rec)
        if verbose:
            print(
                f"epoch {rec['epoch']:3d}  train {rec['train_loss']:.4f}/{rec['train_acc']:.4f}"
                f"  val {val_loss:.4f}/{val_acc:.4f}  |g| {rec['grad_norm']:.3f}"
                f"  {rec['seconds']:.1f}s"
            )
    return hist
