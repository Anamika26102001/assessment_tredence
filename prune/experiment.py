"""One self-pruning run, end to end. Shared by scripts/train_prune.py and the sweep so
every reported number comes from exactly one code path."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from nn import mlp
from optim import Adam
from prune.criteria import make_criterion
from prune.pruner import Pruner
from prune.schedule import CubicSchedule, OneShotSchedule
from train import evaluate, train
from train.data import Dataset


@dataclass
class PruneConfig:
    criterion: str = "fisher_taylor"  # magnitude | taylor1 | fisher_taylor
    sparsity: float = 0.9  # target fraction of prunable weights removed
    seed: int = 0
    epochs: int = 12
    warmup_epochs: int = 1  # dense training before the ramp starts
    ramp_end_epoch: int = 8  # last pruning event; remaining epochs fine-tune at fixed mask
    interval: int = 100  # steps between pruning events
    schedule: str = "cubic"  # cubic | oneshot
    regrow: float = 0.0  # initial swap fraction (0 = no regrowth)
    grace_events: int = 2
    floor_frac: float = 0.01
    ema_beta: float = 0.99
    batch_size: int = 128
    lr: float = 1e-3
    sizes: list[int] = field(default_factory=lambda: [784, 300, 100, 10])


def run_pruning(cfg: PruneConfig, data: Dataset, verbose: bool = False) -> dict[str, Any]:
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)  # same seed -> same init for every criterion
    model = mlp(cfg.sizes, rng)
    opt = Adam(model.parameters(), lr=cfg.lr)
    spe = math.ceil(len(data.x_train) / cfg.batch_size)

    pruner: Pruner | None = None
    if cfg.sparsity > 0:
        sched: CubicSchedule | OneShotSchedule
        if cfg.schedule == "cubic":
            sched = CubicSchedule(
                cfg.sparsity, cfg.warmup_epochs * spe, cfg.ramp_end_epoch * spe, cfg.interval
            )
        else:
            sched = OneShotSchedule(cfg.sparsity, cfg.ramp_end_epoch * spe)
        pruner = Pruner(
            model,
            opt,
            make_criterion(cfg.criterion, cfg.ema_beta),
            sched,
            regrow_frac=cfg.regrow,
            grace_events=cfg.grace_events,
            floor_frac=cfg.floor_frac,
        )

    hist = train(
        model, opt, data.x_train, data.y_train, data.x_val, data.y_val,
        epochs=cfg.epochs, batch_size=cfg.batch_size, seed=cfg.seed,
        callback=pruner, verbose=verbose,
    )  # fmt: skip
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)

    weights = model.prunable_parameters()
    total = sum(p.data.size for p in weights)
    active = int(sum(p.alive().sum() for p in weights))
    # A "pruned" weight must be a real zero, and the model must be finite.
    assert all(not p.data[~p.alive()].any() for p in weights)
    assert all(np.isfinite(p.data).all() for p in model.parameters())
    return {
        "config": asdict(cfg),
        "test_acc": test_acc,
        "test_loss": test_loss,
        "val_acc": hist.epochs[-1]["val_acc"],
        "n_weights": total,
        "n_active": active,
        "achieved_sparsity": 1.0 - active / total,
        "layer_sparsity": [1.0 - float(p.alive().mean()) for p in weights],
        "matmul_flops_dense": int(2 * total),  # per sample: 2 FLOPs per multiply-accumulate
        "matmul_flops_active": int(2 * active),
        "history": hist.epochs,
        "events": pruner.log if pruner else [],
        "seconds": time.perf_counter() - t_start,
    }
