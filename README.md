# Self-Pruning Network

A from-scratch NumPy reverse-mode autodiff engine, an MLP trained with a mask-aware Adam,
and a training loop that prunes its own connections to a hard sparsity budget.

## Dataset

MNIST (OpenML `mnist_784`, version 1), fetched with scikit-learn's `fetch_openml`. Scikit-learn is
used **only** to download the dataset (`train/data.py`); it plays no part in training, models or
gradients. The first run downloads it and caches it at `data/mnist.npz` (gitignored); later runs
read the cache.

Work in progress — full instructions land with the final docs.
