"""Part 4 cost evidence: does pruning actually make inference cheaper?

For each target sparsity we train a self-pruned model, then measure, on the same weights:

  dense_x_zeros  x @ W with zeros left in W (BLAS). The "dense multiply by zeros shortcut".
                 NOT a real saving; included only as the honest reference.
  csr            our own CSR forward: work proportional to alive weights.
  compact        dead neurons / input features physically removed, dense BLAS on the rest.

Reports counted FLOPs, active parameters and median wall-clock latency (single BLAS thread,
set by the Makefile). Whatever the numbers are, they are what gets committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import Tensor, no_grad
from prune.experiment import PruneConfig, run_pruning_with_model
from prune.sparse import SparseMLP, compact, dense_flops_per_sample
from train.data import load_mnist

ROOT = Path(__file__).resolve().parent.parent


def median_ms(fn, x: np.ndarray, reps: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1e3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sparsities", type=float, nargs="+", default=[0.0, 0.5, 0.75, 0.9, 0.95, 0.98]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    data = load_mnist()
    rows = []
    for s in args.sparsities:
        cfg = PruneConfig(sparsity=s, seed=args.seed, epochs=args.epochs,
                          ramp_end_epoch=min(8, args.epochs - 2))  # fmt: skip
        res, model = run_pruning_with_model(cfg, data)
        weights = [p.data for p in model.prunable_parameters()]
        biases = [p.data for p in model.parameters() if not p.prunable]
        sparse, comp = SparseMLP(model), compact(model)
        xt = data.x_test

        # Correctness first: all three paths must agree with the masked dense forward.
        with no_grad():
            ref = model(Tensor(xt)).data
        agree_csr = bool((sparse.forward(xt).argmax(1) == ref.argmax(1)).all())
        agree_cmp = bool((comp.forward(xt).argmax(1) == ref.argmax(1)).all())
        max_diff = float(
            max(np.abs(sparse.forward(xt) - ref).max(), np.abs(comp.forward(xt) - ref).max())
        )

        def dense_zeros(x, ws=weights, bs=biases):
            h = x
            for i, (w, b) in enumerate(zip(ws, bs, strict=True)):
                h = h @ w + b
                if i < len(ws) - 1:
                    h = np.maximum(h, 0)
            return h

        row = {
            "target_sparsity": s,
            "achieved_sparsity": res["achieved_sparsity"],
            "test_acc": res["test_acc"],
            "active_params": sparse.n_active_params,
            "dense_params": int(sum(w.size for w in weights) + sum(b.size for b in biases)),
            "flops_dense": dense_flops_per_sample(model),
            "flops_csr": sparse.flops_per_sample,
            "flops_compact_dense": comp.flops_per_sample,
            "compact_shapes": [list(sh) for sh in comp.shapes],
            "compact_inputs": int(len(comp.input_idx)),
            "paths_agree_with_dense": agree_csr and agree_cmp,
            "max_abs_logit_diff": max_diff,
        }
        for batch, reps in ((1, 1500), (256, 60)):
            xb = xt[:batch]
            row[f"ms_b{batch}_dense_x_zeros"] = median_ms(dense_zeros, xb, reps)
            row[f"ms_b{batch}_csr"] = median_ms(sparse.forward, xb, reps)
            row[f"ms_b{batch}_compact"] = median_ms(comp.forward, xb, reps)
        rows.append(row)
        print(
            f"s={s:.2f} acc={row['test_acc']:.4f} active={row['active_params']:,} "
            f"flops {row['flops_dense']:,}->{row['flops_csr']:,} | "
            f"b1 ms dense/csr/compact {row['ms_b1_dense_x_zeros']:.3f}/{row['ms_b1_csr']:.3f}/"
            f"{row['ms_b1_compact']:.3f} | b256 {row['ms_b256_dense_x_zeros']:.2f}/"
            f"{row['ms_b256_csr']:.2f}/{row['ms_b256_compact']:.2f} | "
            f"agree={row['paths_agree_with_dense']}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "machine": platform.platform(),
        "numpy": np.__version__,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "note": "median of repeated runs after warmup; single BLAS thread",
    }
    (args.out / "bench.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    with open(args.out / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "compact_shapes"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "compact_shapes"})

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    sp = [r["achieved_sparsity"] * 100 for r in rows]
    axes[0].plot(
        sp, [r["flops_dense"] / r["flops_csr"] for r in rows], "o-", label="counted FLOPs (CSR)"
    )
    axes[0].plot(
        sp, [r["dense_params"] / r["active_params"] for r in rows], "s--", label="active params"
    )
    axes[0].set(xlabel="sparsity %", ylabel="reduction factor (x)", title="Work removed")
    for ax, b in ((axes[1], 1), (axes[2], 256)):
        base = [r[f"ms_b{b}_dense_x_zeros"] for r in rows]
        ax.plot(
            sp, [x / r[f"ms_b{b}_csr"] for x, r in zip(base, rows, strict=True)], "o-", label="CSR"
        )
        ax.plot(
            sp,
            [x / r[f"ms_b{b}_compact"] for x, r in zip(base, rows, strict=True)],
            "s-",
            label="compacted dense",
        )
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set(
            xlabel="sparsity %",
            ylabel="speedup vs dense x zeros (x)",
            title=f"Wall-clock, batch {b}",
        )
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "bench.png", dpi=130)


if __name__ == "__main__":
    main()
