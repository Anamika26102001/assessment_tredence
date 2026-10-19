"""Sensitivity of fisher_taylor at 95% sparsity to the EMA decay beta.

beta sets the window of the squared-gradient average (about 1/(1-beta) steps) relative to the
100-step pruning interval. Only VALIDATION accuracy is recorded or reported here: test
accuracy is deliberately not stored, so this sweep cannot be used to pick beta on test."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from prune.experiment import PruneConfig, run_pruning
from scripts.ablate_lib import env_info, mean_std
from scripts.stats import load_runs, paired_summary

ROOT = Path(__file__).resolve().parent.parent
BETAS = [0.9, 0.99, 0.999, 0.9999]
SEEDS = list(range(8))
SPARSITY = 0.95
_DATA = None


def _worker(cfg: PruneConfig) -> dict:
    global _DATA
    if _DATA is None:
        from train.data import load_mnist

        _DATA = load_mnist()
    res = run_pruning(cfg, _DATA)
    return {"beta": cfg.ema_beta, "seed": cfg.seed, "val_acc": res["val_acc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations" / "beta.json")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    base = PruneConfig(criterion="fisher_taylor", sparsity=SPARSITY, ramp_end_epoch=8)
    cfgs = [replace(base, ema_beta=b, seed=s) for b in BETAS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        runs = list(ex.map(_worker, cfgs))
    val = {b: np.array([r["val_acc"] for r in runs if r["beta"] == b]) for b in BETAS}

    committed = {
        r["config"]["seed"]: r["val_acc"]
        for r in load_runs(ROOT / "results" / "runs")
        if r["config"]["criterion"] == "magnitude" and r["config"]["sparsity"] == SPARSITY
    }
    magnitude_val = np.array([committed[s] for s in SEEDS])
    out: dict = {
        "sparsity": SPARSITY,
        "metric": "validation accuracy (test accuracy intentionally not recorded)",
        "magnitude_val_mean": float(magnitude_val.mean()),
        "betas": {},
        "env": env_info(),
    }
    for b in BETAS:
        mu, sd = mean_std(val[b])
        vs_mag = paired_summary(val[b], magnitude_val)
        vs_default = paired_summary(val[b], val[0.99])
        out["betas"][str(b)] = {
            "window_steps": round(1 / (1 - b)),
            "val_acc": val[b].tolist(),
            "val_mean": mu,
            "val_std": sd,
            "minus_magnitude": vs_mag,
            "minus_beta_0.99": vs_default,
        }
        print(
            f"beta {b:<7} window {round(1 / (1 - b)):>6} steps  val {100 * mu:.2f} +/- {100 * sd:.2f}"
            f"   vs magnitude {100 * vs_mag['mean_diff']:+.2f} pp"
            f" [{100 * vs_mag['ci_lo']:+.2f}, {100 * vs_mag['ci_hi']:+.2f}]"
        )
    print(f"magnitude val {100 * magnitude_val.mean():.2f} (committed runs)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
