"""Inference cost with variance: the wall-clock benchmark over 8 seeds' models.

The single-model benchmark (bench_inference.py) gives one number per sparsity. This one
trains the magnitude and fisher_taylor models for seeds 0-7 at 90/95/98% (cached under
data/bench_models/, deterministic), then times three forward paths on each model:

  dense_x_zeros  dense BLAS on the masked weights (the "multiply by zeros" shortcut)
  csr            our own CSR forward
  compact        dead neurons / input features removed, dense BLAS on the rest

Timing is round-robin over the paths (so drift hits all of them alike) with 50 samples per
path after warm-up; each sample is the mean of several calls when one call is microseconds.
Per model the speedup is median(dense_x_zeros) / median(path); across the 8 models we report
the median and inter-quartile range of that speedup. Training runs in parallel, timing runs
alone in the main process."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from engine import Tensor, no_grad
from nn import mlp
from prune.experiment import PruneConfig, run_pruning_with_model
from prune.sparse import SparseMLP, compact
from scripts.ablate_lib import env_info

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "bench_models"
METHODS = ["magnitude", "fisher_taylor"]
SPARSITIES = [0.9, 0.95, 0.98]
SEEDS = list(range(8))
BATCHES = {1: 20, 256: 1}  # batch size -> calls averaged per timing sample
_DATA = None


def model_path(method: str, sparsity: float, seed: int) -> Path:
    return CACHE / f"{method}_s{sparsity}_seed{seed}.npz"


def _train(job: tuple[str, float, int]) -> str:
    global _DATA
    method, sparsity, seed = job
    path = model_path(*job)
    if path.exists():
        return str(path)
    if _DATA is None:
        from train.data import load_mnist

        _DATA = load_mnist()
    res, model = run_pruning_with_model(
        PruneConfig(criterion=method, sparsity=sparsity, seed=seed, ramp_end_epoch=8), _DATA
    )
    arrays: dict[str, np.ndarray] = {"test_acc": np.array(res["test_acc"])}
    weights = model.prunable_parameters()
    for i, p in enumerate(weights):
        arrays[f"w{i}"], arrays[f"m{i}"] = p.data, p.alive()
    for i, p in enumerate(q for q in model.parameters() if not q.prunable):
        arrays[f"b{i}"] = p.data
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]
    return str(path)


def load_model(path: Path):
    z = np.load(path)
    model = mlp([784, 300, 100, 10], np.random.default_rng(0))
    for i, p in enumerate(model.prunable_parameters()):
        p.set_mask(z[f"m{i}"])
        p.data[...] = z[f"w{i}"]
    for i, p in enumerate(q for q in model.parameters() if not q.prunable):
        p.data[...] = z[f"b{i}"]
    return model, float(z["test_acc"])


def time_paths(
    fns: dict[str, Callable[[np.ndarray], np.ndarray]], x: np.ndarray, inner: int, reps: int = 50
) -> dict[str, list[float]]:
    for fn in fns.values():
        for _ in range(20):
            fn(x)
    samples: dict[str, list[float]] = {k: [] for k in fns}
    for _ in range(reps):
        for name, fn in fns.items():
            t = time.perf_counter()
            for _ in range(inner):
                fn(x)
            samples[name].append((time.perf_counter() - t) / inner * 1e3)
    return samples


def q(a: list[float] | np.ndarray, p: float) -> float:
    return float(np.percentile(np.asarray(a), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bench_variance.json")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    from train.data import load_mnist

    jobs = [(m, s, seed) for m, s, seed in product(METHODS, SPARSITIES, SEEDS)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(_train, jobs))
    x_test = load_mnist().x_test

    models = []
    for method, sparsity, seed in jobs:
        model, test_acc = load_model(model_path(method, sparsity, seed))
        weights = [p.data for p in model.prunable_parameters()]
        biases = [p.data for p in model.parameters() if not p.prunable]
        sparse, comp = SparseMLP(model), compact(model)

        def dense_zeros(h, ws=weights, bs=biases):
            for i, (w, b) in enumerate(zip(ws, bs, strict=True)):
                h = h @ w + b
                if i < len(ws) - 1:
                    h = np.maximum(h, 0)
            return h

        with no_grad():
            ref = model(Tensor(x_test[:512])).data
        agree = bool(
            (sparse.forward(x_test[:512]).argmax(1) == ref.argmax(1)).all()
            and (comp.forward(x_test[:512]).argmax(1) == ref.argmax(1)).all()
        )
        row = {
            "method": method, "sparsity": sparsity, "seed": seed, "test_acc": test_acc,
            "paths_agree_with_dense": agree,
            "flops_dense": 2 * sum(w.size for w in weights) + sum(b.size for b in biases),
            "flops_csr": sparse.flops_per_sample,
            "flops_compact": comp.flops_per_sample,
            "compact_shapes": [list(s) for s in comp.shapes],
        }  # fmt: skip
        fns: dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dense_x_zeros": dense_zeros,
            "csr": sparse.forward,
            "compact": comp.forward,
        }
        for batch, inner in BATCHES.items():
            s = time_paths(fns, x_test[:batch], inner, args.reps)
            for name, v in s.items():
                row[f"ms_b{batch}_{name}"] = {
                    "median": q(v, 50), "q25": q(v, 25), "q75": q(v, 75),
                }  # fmt: skip
        models.append(row)
        print(
            f"{method:13s} s={sparsity} seed={seed} agree={agree} "
            f"b256 dense/csr/compact ms {row['ms_b256_dense_x_zeros']['median']:.3f}/"
            f"{row['ms_b256_csr']['median']:.2f}/{row['ms_b256_compact']['median']:.3f}",
            flush=True,
        )

    summary = []
    for method, sparsity in product(METHODS, SPARSITIES):
        rows = [r for r in models if r["method"] == method and r["sparsity"] == sparsity]
        entry: dict = {
            "method": method, "sparsity": sparsity, "n_models": len(rows),
            "flops_reduction_csr": float(np.median([r["flops_dense"] / r["flops_csr"] for r in rows])),
            "flops_reduction_compact": float(np.median([r["flops_dense"] / r["flops_compact"] for r in rows])),
        }  # fmt: skip
        for batch in BATCHES:
            for path in ("csr", "compact"):
                sp = [
                    r[f"ms_b{batch}_dense_x_zeros"]["median"] / r[f"ms_b{batch}_{path}"]["median"]
                    for r in rows
                ]
                entry[f"speedup_b{batch}_{path}"] = {
                    "median": q(sp, 50), "q25": q(sp, 25), "q75": q(sp, 75),
                    "min": min(sp), "max": max(sp),
                }  # fmt: skip
        summary.append(entry)
        c1, c256 = entry["speedup_b1_compact"], entry["speedup_b256_compact"]
        print(
            f"{method:13s} {sparsity}: compact speedup b1 {c1['median']:.2f}x [{c1['q25']:.2f},{c1['q75']:.2f}]"
            f"  b256 {c256['median']:.2f}x [{c256['q25']:.2f},{c256['q75']:.2f}]"
            f"  csr b1 {entry['speedup_b1_csr']['median']:.2f}x b256 {entry['speedup_b256_csr']['median']:.3f}x"
        )
    out = {
        "env": env_info(),
        "protocol": {
            "samples_per_path": args.reps,
            "calls_per_sample": BATCHES,
            "order": "round-robin over paths",
            "speedup": "median(dense_x_zeros) / median(path), per model; then median and IQR over 8 models",
        },
        "models": models,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    flat_rows = []
    for e in summary:
        r = {
            k: e[k]
            for k in (
                "method",
                "sparsity",
                "n_models",
                "flops_reduction_csr",
                "flops_reduction_compact",
            )
        }
        for b, p in product(BATCHES, ("csr", "compact")):
            for k in ("median", "q25", "q75"):
                r[f"speedup_b{b}_{p}_{k}"] = e[f"speedup_b{b}_{p}"][k]
        flat_rows.append(r)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        w.writeheader()
        w.writerows(flat_rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
