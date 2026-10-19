"""Does the magnitude-vs-fisher_taylor gap depend on the optimizer?

Adam rescales every weight's step by its own gradient scale, which could make |w| already
gradient-aware. Here both criteria are run with SGD + momentum instead. The learning rate is
picked on VALIDATION accuracy of the dense model only (seeds 100-102, disjoint from the
evaluation seeds), then held fixed for every pruned run; test accuracy is reported only for
the final paired comparison."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from prune.experiment import PruneConfig, run_pruning
from scripts.ablate_lib import env_info, mean_std
from scripts.stats import paired_summary

ROOT = Path(__file__).resolve().parent.parent
LRS = [0.003, 0.01, 0.03, 0.1]
SELECTION_SEEDS = [100, 101, 102]
EVAL_SEEDS = list(range(8))
SPARSITY = 0.95
_DATA = None


def _worker(cfg: PruneConfig) -> dict:
    global _DATA
    if _DATA is None:
        from train.data import load_mnist

        _DATA = load_mnist()
    try:
        res = run_pruning(cfg, _DATA)
    except FloatingPointError as e:  # a diverged run counts as the worst outcome
        return {"config": asdict(cfg), "diverged": str(e), "val_acc": 0.0, "test_acc": 0.0}
    return {
        "config": asdict(cfg),
        "val_acc": res["val_acc"],
        "test_acc": res["test_acc"],
        "val_curve": [h["val_acc"] for h in res["history"]],
        "achieved_sparsity": res["achieved_sparsity"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations" / "sgd.json")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    base = PruneConfig(optimizer="sgd", momentum=0.9, ramp_end_epoch=8)

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        sel_cfgs = [
            replace(base, sparsity=0.0, lr=lr, seed=s) for lr in LRS for s in SELECTION_SEEDS
        ]
        sel = list(ex.map(_worker, sel_cfgs))
        table = {}
        mean_val: dict[float, float] = {}
        for lr in LRS:
            accs = [r["val_acc"] for r in sel if r["config"]["lr"] == lr]
            mean_val[lr] = float(np.mean(accs))
            table[str(lr)] = {"mean_val_acc": mean_val[lr], "val_acc": accs}
            print(f"lr {lr:<6} dense val acc {mean_val[lr]:.4f}")
        best_lr = max(LRS, key=mean_val.__getitem__)
        print(f"selected lr = {best_lr} (validation only)")

        cfgs = {"dense": replace(base, sparsity=0.0, lr=best_lr)}
        for crit in ("magnitude", "fisher_taylor"):
            cfgs[crit] = replace(base, criterion=crit, sparsity=SPARSITY, lr=best_lr)
        jobs = [(k, replace(c, seed=s)) for k, c in cfgs.items() for s in EVAL_SEEDS]
        results = list(ex.map(_worker, [c for _, c in jobs]))
    by: dict[str, dict[int, dict]] = {k: {} for k in cfgs}
    for (k, c), r in zip(jobs, results, strict=True):
        by[k][c.seed] = r

    def arr(k: str, field: str) -> np.ndarray:
        return np.array([by[k][s][field] for s in EVAL_SEEDS])

    summary = {}
    for k in cfgs:
        mu, sd = mean_std(arr(k, "test_acc"))
        vmu, vsd = mean_std(arr(k, "val_acc"))
        summary[k] = {"test_mean": mu, "test_std": sd, "val_mean": vmu, "val_std": vsd}
        print(
            f"{k:14s} test {100 * mu:.2f} +/- {100 * sd:.2f}   val {100 * vmu:.2f} +/- {100 * vsd:.2f}"
        )
    paired = {
        "fisher_taylor_minus_magnitude": paired_summary(
            arr("fisher_taylor", "test_acc"), arr("magnitude", "test_acc")
        ),
        "magnitude_minus_dense": paired_summary(
            arr("magnitude", "test_acc"), arr("dense", "test_acc")
        ),
    }
    for name, p in paired.items():
        print(
            f"{name}: {100 * p['mean_diff']:+.2f} pp  CI [{100 * p['ci_lo']:+.2f}, "
            f"{100 * p['ci_hi']:+.2f}]  p={p['p_signflip']:.4f}  A wins {p['a_wins']}/8"
        )
    out = {
        "optimizer": "sgd+momentum 0.9",
        "lr_search": table,
        "selected_lr": best_lr,
        "sparsity": SPARSITY,
        "runs": {k: {str(s): v for s, v in d.items()} for k, d in by.items()},
        "summary": summary,
        "paired": paired,
        "env": env_info(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
