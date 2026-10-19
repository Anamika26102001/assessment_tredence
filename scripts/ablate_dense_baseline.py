"""How much does 'pruned beats dense' depend on which dense checkpoint is the baseline?

Trains the sweep's dense model for seeds 0-7 (identical to the committed dense runs) and
records validation and test accuracy after every epoch. Reports the final epoch (what the
sweep compares against) and the epoch with the best VALIDATION accuracy (first one on ties).
Test accuracy is read at that epoch but never used to choose it. Nothing is retuned."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nn import mlp
from nn.module import Module
from optim import Adam
from scripts.ablate_lib import env_info, mean_std
from scripts.stats import group, load_runs, paired_summary
from train import evaluate, train
from train.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(8))


class TestAccPerEpoch:
    def __init__(self, model: Module, data: Dataset) -> None:
        self.model, self.data = model, data
        self.test_acc: list[float] = []

    def after_backward(self, step: int) -> None:
        pass

    def after_step(self, step: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, record: dict) -> None:
        self.test_acc.append(evaluate(self.model, self.data.x_test, self.data.y_test)[1])


def one_seed(seed: int) -> dict:
    from train.data import load_mnist

    data = load_mnist()
    model = mlp([784, 300, 100, 10], np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=1e-3)
    cb = TestAccPerEpoch(model, data)
    hist = train(model, opt, data.x_train, data.y_train, data.x_val, data.y_val,
                 epochs=12, batch_size=128, seed=seed, callback=cb, verbose=False)  # fmt: skip
    val = [h["val_acc"] for h in hist.epochs]
    best = int(np.argmax(val))
    return {
        "seed": seed,
        "val_acc": val,
        "test_acc": cb.test_acc,
        "train_acc_final": hist.epochs[-1]["train_acc"],
        "final": {"epoch": 12, "val": val[-1], "test": cb.test_acc[-1]},
        "best_val": {"epoch": best + 1, "val": val[best], "test": cb.test_acc[best]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results" / "ablations" / "dense_baseline.json"
    )
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        runs = sorted(ex.map(one_seed, SEEDS), key=lambda r: r["seed"])

    g = group(load_runs(ROOT / "results" / "runs"))
    committed_dense = np.array([g[("dense", 0.0)][s]["test_acc"] for s in SEEDS])
    assert np.array_equal(committed_dense, [r["final"]["test"] for r in runs]), "dense runs drifted"

    summary: dict = {}
    for which in ("final", "best_val"):
        for field in ("val", "test"):
            mu, sd = mean_std([r[which][field] for r in runs])
            summary[f"{which}_{field}"] = {"mean": mu, "std": sd}
        summary[f"{which}_epochs"] = [r[which]["epoch"] for r in runs]
    pruned: dict[str, dict] = {}
    for method in ("magnitude", "fisher_taylor"):
        acc = np.array([g[(method, 0.95)][s]["test_acc"] for s in SEEDS])
        pruned[method] = {
            "test_mean": float(acc.mean()),
            "minus_dense_final": paired_summary(acc, np.array([r["final"]["test"] for r in runs])),
            "minus_dense_best_val": paired_summary(
                acc, np.array([r["best_val"]["test"] for r in runs])
            ),
        }
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"dense {k:15s} {100 * v['mean']:.2f} +/- {100 * v['std']:.2f}")
        else:
            print(f"dense {k:15s} {v}")
    for method, d in pruned.items():
        for ref in ("minus_dense_final", "minus_dense_best_val"):
            p = d[ref]
            print(
                f"95% {method:13s} {ref:21s} {100 * p['mean_diff']:+.2f} pp  "
                f"CI [{100 * p['ci_lo']:+.2f}, {100 * p['ci_hi']:+.2f}]  p={p['p_signflip']:.4f}"
            )
    out = {"per_seed": runs, "summary": summary, "pruned_95_vs_dense": pruned, "env": env_info()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
