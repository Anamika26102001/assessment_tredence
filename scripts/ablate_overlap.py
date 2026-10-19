"""Kept-set overlap between magnitude and fisher_taylor after pruning to 95%.

For each seed the two methods start from the same init and data order; the Jaccard overlap
of the sets of surviving weights says how differently they prune, per layer and overall."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from prune.experiment import PruneConfig, run_pruning_with_model
from scripts.ablate_lib import env_info, jaccard, mean_std

ROOT = Path(__file__).resolve().parent.parent
SPARSITY = 0.95


def one_seed(seed: int) -> dict:
    from train.data import load_mnist

    data = load_mnist()
    masks, acc = {}, {}
    for crit in ("magnitude", "fisher_taylor"):
        res, model = run_pruning_with_model(
            PruneConfig(criterion=crit, sparsity=SPARSITY, seed=seed, ramp_end_epoch=8), data
        )
        masks[crit] = [p.alive().copy() for p in model.prunable_parameters()]
        acc[crit] = res["test_acc"]
    m, f = masks["magnitude"], masks["fisher_taylor"]
    layers = {f"layer{i + 1}": jaccard(a, b) for i, (a, b) in enumerate(zip(m, f, strict=True))}
    layers["all"] = jaccard(
        np.concatenate([a.ravel() for a in m]), np.concatenate([b.ravel() for b in f])
    )
    return {
        "seed": seed,
        "jaccard": layers,
        "kept_per_layer": {c: [int(a.sum()) for a in masks[c]] for c in masks},
        "test_acc": acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations" / "overlap.json")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        per_seed = sorted(ex.map(one_seed, args.seeds), key=lambda r: r["seed"])
    summary = {}
    for name in per_seed[0]["jaccard"]:
        mu, sd = mean_std([r["jaccard"][name] for r in per_seed])
        summary[name] = {"mean": mu, "std": sd}
        print(f"{name:7s} Jaccard {mu:.3f} +/- {sd:.3f}")
    out = {"sparsity": SPARSITY, "per_seed": per_seed, "summary": summary, "env": env_info()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
