"""MNIST loading. scikit-learn is used here ONLY to download the dataset (via OpenML);
it plays no part in models, training or gradients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "mnist.npz"


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _download() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml  # dataset loading only

    x, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff")
    return x.astype(np.uint8), y.astype(np.int64)


def load_raw_mnist(path: Path = CACHE) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        with np.load(path) as f:
            return f["x"], f["y"]
    x, y = _download()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, y=y)
    return x, y


def split_and_normalize(
    x: np.ndarray, y: np.ndarray, n_train: int = 55_000, n_val: int = 5_000, n_test: int = 10_000
) -> Dataset:
    """Fixed split, independent of any run seed: the last ``n_test`` rows are the standard
    MNIST test set; the rest is permuted with a fixed seed into train / val. Val is for
    every decision; test is only ever reported. Pixels are scaled to [0, 1] and
    standardised with a single global mean/std computed on the training split only."""
    if n_train + n_val + n_test > len(x):
        raise ValueError("split sizes exceed dataset size")
    pool = len(x) - n_test
    perm = np.random.default_rng(0).permutation(pool)
    tr, va = perm[:n_train], perm[n_train : n_train + n_val]
    xf = x.astype(np.float32) / 255.0
    mean, std = xf[tr].mean(), xf[tr].std()
    xf = (xf - mean) / std
    return Dataset(xf[tr], y[tr], xf[va], y[va], xf[pool:][:n_test], y[pool:][:n_test])


def load_mnist() -> Dataset:
    x, y = load_raw_mnist()
    return split_and_normalize(x, y)
