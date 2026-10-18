"""Part 4 sweep: every (method, sparsity, seed) is one resumable run on disk.

One JSON per run in --out; existing files are skipped, so an interrupted sweep resumes and
a finished one is a no-op. Runs are independent and deterministic (fixed seed, single BLAS
thread), so --jobs only changes wall-clock time, never the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from prune.experiment import PruneConfig, run_pruning

ROOT = Path(__file__).resolve().parent.parent
SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98]
HIGH = [0.9, 0.95, 0.98]  # where the ablations are run
SEEDS = list(range(8))

_DATA = None


def variant_of(cfg: dict) -> str:
    """Human-readable method name from a run's config dict."""
    if cfg["sparsity"] == 0:
        return "dense"
    name = cfg["criterion"]
    if cfg["schedule"] == "oneshot":
        name += "+oneshot"
    if cfg["regrow"] > 0:
        name += "+regrow"
    return name


def plan(
    seeds: list[int] = SEEDS,
    sparsities: list[float] = SPARSITIES,
    high: list[float] = HIGH,
    epochs: int = 12,
) -> list[PruneConfig]:
    base = PruneConfig(epochs=epochs, ramp_end_epoch=min(8, epochs - 2))
    cfgs: list[PruneConfig] = []
    for seed in seeds:
        cfgs.append(replace(base, sparsity=0.0, seed=seed))  # dense reference (0% sparsity)
        for s in sparsities:
            for crit in ("magnitude", "fisher_taylor"):
                cfgs.append(replace(base, criterion=crit, sparsity=s, seed=seed))
        for s in high:  # ablations: gradient order, regrowth, schedule
            cfgs.append(replace(base, criterion="taylor1", sparsity=s, seed=seed))
            cfgs.append(replace(base, criterion="fisher_taylor", sparsity=s, seed=seed, regrow=0.1))
            cfgs.append(
                replace(base, criterion="fisher_taylor", sparsity=s, seed=seed, schedule="oneshot")
            )
    return cfgs


def run_key(cfg: PruneConfig) -> str:
    from dataclasses import asdict

    return f"{variant_of(asdict(cfg))}__s{cfg.sparsity}__seed{cfg.seed}"


def _worker(cfg: PruneConfig) -> dict:
    global _DATA
    if _DATA is None:
        from train.data import load_mnist

        _DATA = load_mnist()
    return run_pruning(cfg, _DATA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--quick", action="store_true", help="2 seeds, 3 sparsities, 4 epochs (smoke test)"
    )
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfgs = plan(seeds=[0, 1], sparsities=[0.5, 0.9], high=[0.9], epochs=4) if args.quick else plan()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (args.out / f"{run_key(c)}.json").exists()]
    print(f"{len(cfgs)} runs planned, {len(cfgs) - len(todo)} already done, {len(todo)} to run")
    if args.quick or args.jobs <= 1:
        from train.data import load_mnist

        data = load_mnist()
        results = ((c, run_pruning(c, data)) for c in todo)
        for i, (c, res) in enumerate(results, 1):
            (args.out / f"{run_key(c)}.json").write_text(json.dumps(res))
            print(f"[{i}/{len(todo)}] {run_key(c)} acc={res['test_acc']:.4f}", flush=True)
        return
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_worker, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            res = fut.result()
            (args.out / f"{run_key(c)}.json").write_text(json.dumps(res))
            eta = (time.perf_counter() - t0) / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] {run_key(c)} acc={res['test_acc']:.4f}  eta {eta / 60:.1f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()
