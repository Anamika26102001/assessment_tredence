# Self-Pruning Network

A neural-network library written from scratch in NumPy: a reverse-mode autodiff engine, an MLP
with Adam / SGD, and a training loop that prunes its own connections during training to a hard
sparsity budget, with the evidence for what that bought.

**Scikit-learn is used only to download MNIST** (`fetch_openml` in `train/data.py`). It plays no
part in training, models, gradients or pruning. No PyTorch, TensorFlow, JAX, Keras or other
autodiff / pruning / neural-network library is used; the engine, backward pass, pruning
mechanism, optimizers and training loop are in this repository.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Dataset: MNIST (OpenML `mnist_784`, version 1). **The first run downloads it (about 15 MB) and
caches it at `data/mnist.npz`** (git-ignored); later runs read the cache. The split is fixed:
55,000 train / 5,000 validation / 10,000 test (the standard test set), normalised with training
statistics. All choices are made on validation; test accuracy is only reported.

## Run

All commands run from the repository root. `make` pins BLAS to a single thread, which makes
reductions deterministic and wall-clock comparisons fair.

| What | Command | Time (Apple M-series laptop) |
|---|---|---|
| Gradient checks (every op, broadcasting, masked weights) | `make gradcheck` | seconds |
| All tests, lint and type check | `make check` | seconds |
| Part 2: dense MLP + Adam, learning curve | `make train` | ~15 s |
| Part 3: one self-pruning run (fisher_taylor, 90%, seed 0) | `make prune` | ~20 s |
| Part 4: full Pareto sweep, 160 runs, statistics, plots | `make sweep` | ~15 min with 4 workers |
| Part 4: inference cost benchmark (one model per sparsity) | `make bench` | ~3 min |
| Inference cost over 8 seeds with variance | `make bench-variance` | ~5 min |
| Explanatory ablations (see `results/RESULTS.md`) | `make ablate` | ~10 min |
| Interactive digit demo (`demo/index.html`) | `make demo` | seconds |

The sweep is resumable: each run is one JSON file in `results/runs/` and existing files are
skipped, so `make sweep` on a finished checkout only regenerates the statistics and plots.

## Layout

```
engine/   Tensor, differentiable ops, fused softmax cross-entropy, gradcheck, no_grad
nn/       Parameter (with connectivity mask), Linear / ReLU / Tanh / Sequential, initialisers
optim/    Adam and SGD+momentum with per-entry state that is correct under masking
prune/    importance criteria, sparsity schedules, the Pruner (prune + regrow),
          CSR / compacted sparse inference
train/    MNIST loading and the mini-batch training loop
scripts/  entry points: training, sweep, statistics, plots, benchmarks, ablations
tests/    gradient checks, masked-weight correctness, optimizer, pruning, sparse inference, stats
results/  committed raw numbers (runs/, *.csv, *.json), plots, and RESULTS.md
```

## What it does

- **Masked weights.** The forward pass uses `W * M`, so a pruned weight is a true zero. The
  gradient of a masked weight is exactly 0 (the true derivative of `L(W*M)`), matching finite
  differences. The unmasked gradient `dL/dW_eff` is kept for regrowth scoring and never reaches an
  optimizer. Adam keeps a per-entry step count and moments, freezes masked entries, and resets
  them when a connection is pruned or revived (`tests/test_masked.py` demonstrates this, including
  a deliberately naive Adam that fails it).
- **Pruning.** Importance criteria `magnitude`, `taylor1` (|w g|) and `fisher_taylor`
  (`w^2 * EMA[g^2]`, Optimal Brain Damage with the empirical Fisher), a global ranking with a
  per-layer floor, a cubic sparsity ramp (one-shot as an ablation), an exact weight budget checked
  after every event, and optional RigL-style regrowth.
- **Evidence.** Sparsity-accuracy sweep over 8 seeds with paired statistics, counted FLOPs and
  parameters, and measured wall-clock for a sparse-aware CSR forward and a compacted dense forward.

## Results in one paragraph

On MNIST (784-300-100-10), magnitude pruning is **better** than the gradient-based criterion at
95% sparsity: 98.15% vs 97.97% test accuracy over 8 seeds, paired difference +0.18 pp, 95%
bootstrap CI [+0.07, +0.29], magnitude better in 6 of 8 seeds. Both stay within about half a point
of the dense model (and above it, against an unregularised baseline). Counted FLOPs fall by
sparsity (19.7x at 95%), but wall-clock does not follow: the NumPy CSR path is slower than dense
BLAS, and structural compaction speeds up only some models. Seven ablations probe why magnitude
wins. **Everything, with the caveats and limitations, is in [`results/RESULTS.md`](results/RESULTS.md).**

## Reproducibility

- Every random choice is seeded (`np.random.default_rng(seed)` for initialisation and batch
  order). The same config gives bit-identical numbers; the sweep's runs depend only on their
  config, so `--jobs` changes wall-clock time, never results.
- Dependencies are locked in `uv.lock`. Timings depend on the machine; counts, accuracies and
  statistics do not. `results/bench.json` and `results/bench_variance.json` record the CPU,
  NumPy version, BLAS and thread settings they were measured with.
- `make sweep` regenerates `results/pareto.csv`, `paired_stats.csv`, `RESULTS.md` and the plots
  from `results/runs/*.json`.
