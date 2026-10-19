"""Shared helpers for the explanatory ablations (scripts/ablate_*.py).

These ablations only explain the headline comparison; none of them selects anything on the
test set. Anything that chooses a setting uses validation accuracy.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
from typing import Any

import numpy as np

from nn import mlp
from nn.module import Module
from optim import Adam
from prune import CubicSchedule, FisherTaylor, Pruner
from train import evaluate, train
from train.data import Dataset


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values get the mean of the ranks they span."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(np.asarray(a)), average_ranks(np.asarray(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_concordance(
    score: np.ndarray, delta: np.ndarray, pairs: np.ndarray | None = None
) -> tuple[float, int]:
    """Fraction of pairs (i, j) with delta_i != delta_j whose score order matches the
    delta order (a higher true loss increase should get a higher score). Score ties count
    half. ``pairs`` is an optional (n, n) boolean matrix restricting which pairs count."""
    ds = delta[:, None] - delta[None, :]
    ss = score[:, None] - score[None, :]
    valid = np.triu(np.ones_like(ds, dtype=bool), k=1) & (ds != 0)
    if pairs is not None:
        valid &= pairs | pairs.T
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    agree = (np.sign(ss) == np.sign(ds))[valid].sum() + 0.5 * (ss == 0)[valid].sum()
    return float(agree / n), n


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else float("nan")


def flat(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([a.ravel() for a in arrays])


def train_dense_with_fisher(
    seed: int, data: Dataset, epochs: int = 12, batch_size: int = 128, lr: float = 1e-3
) -> tuple[Module, FisherTaylor, float]:
    """The sweep's dense run, plus a FisherTaylor EMA accumulated on the side.

    The pruner never fires (its schedule starts at step 1e9), so the weights are bit-identical
    to the committed dense run for this seed; it only records EMA[g^2]."""
    model = mlp([784, 300, 100, 10], np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=lr)
    crit = FisherTaylor(0.99)
    pruner = Pruner(model, opt, crit, CubicSchedule(0.95, 10**9, 10**9 + 1, 1))
    train(model, opt, data.x_train, data.y_train, data.x_val, data.y_val,
          epochs=epochs, batch_size=batch_size, seed=seed, callback=pruner, verbose=False)  # fmt: skip
    return model, crit, evaluate(model, data.x_test, data.y_test)[1]


def cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def blas_name() -> str:
    try:
        cfg: Any = np.show_config(mode="dicts")
        return str(cfg["Build Dependencies"]["blas"].get("name", "unknown"))
    except Exception:
        return "unknown"


def env_info() -> dict[str, Any]:
    threads = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    }
    return {
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": blas_name(),
        "blas_threads_env": threads,
    }


def mean_std(x: list[float] | np.ndarray) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else math.nan
