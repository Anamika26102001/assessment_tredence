"""Oracle test: does each importance score predict the TRUE loss increase from removing a weight?

On the trained dense seed-0 model, zero single weights one at a time and measure the exact
validation-loss increase (full 5k validation set, float64, deterministic). Compare it with
|w|, w^2 * EMA[g^2] (bias-corrected) and |w * g| (last mini-batch gradient, as the taylor1
criterion sees it). Three weight sets:

  random      500 weights drawn uniformly
  boundary_m  the 500 weights ranked closest to the 95% keep threshold under |w|
  boundary_f  the same under the Fisher-Taylor score

Boundary sets are where a pruning decision is actually contested. Note the selection effect:
inside boundary_m the |w| values are nearly equal by construction (likewise for _f), so the
selecting criterion has little spread to work with there; the cross-threshold pairs are the
decisions that criterion really makes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engine.functional import log_softmax_np
from nn.module import Module
from prune import Magnitude, Taylor1
from scripts.ablate_lib import (
    env_info,
    flat,
    pair_concordance,
    spearman,
    train_dense_with_fisher,
)
from train.data import load_mnist

ROOT = Path(__file__).resolve().parent.parent
SPARSITY = 0.95
N_PER_SET = 500


class Oracle:
    """Exact single-weight ablation on a ReLU MLP, float64, caching the activations below
    the layer being edited."""

    def __init__(self, model: Module, x: np.ndarray, y: np.ndarray) -> None:
        self.ws = [p.data.astype(np.float64) for p in model.parameters() if p.prunable]
        self.bs = [p.data.astype(np.float64) for p in model.parameters() if not p.prunable]
        self.y = np.asarray(y)
        self.acts = [x.astype(np.float64)]  # input to each layer
        for i in range(len(self.ws) - 1):
            self.acts.append(np.maximum(self.acts[-1] @ self.ws[i] + self.bs[i], 0))
        self.base = self.loss_from(0, self.ws[0], self.acts[0])

    def loss_from(self, layer: int, w: np.ndarray, h: np.ndarray) -> float:
        n = len(self.ws)
        for i in range(layer, n):
            h = h @ (w if i == layer else self.ws[i]) + self.bs[i]
            if i < n - 1:
                h = np.maximum(h, 0)
        return float(-log_softmax_np(h)[np.arange(len(self.y)), self.y].mean())

    def delta(self, layer: int, index: tuple[Any, ...]) -> float:
        w = self.ws[layer]
        old = w[index]
        w[index] = 0.0
        loss = self.loss_from(layer, w, self.acts[layer])
        w[index] = old
        return loss - self.base


def locate(flat_idx: np.ndarray, sizes: list[int]) -> list[tuple[int, int]]:
    edges = np.cumsum([0, *sizes])
    layer = np.searchsorted(edges, flat_idx, side="right") - 1
    return [(int(layer[k]), int(flat_idx[k] - edges[layer[k]])) for k in range(len(flat_idx))]


def band_around_threshold(score: np.ndarray, n_keep: int, n: int) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    return order[n_keep - n // 2 : n_keep + n // 2]


def analyse(
    name: str, idx: np.ndarray, oracle: Oracle, sizes: list[int], shapes: list[tuple[int, ...]],
    scores: dict[str, np.ndarray], thresholds: dict[str, float],
) -> dict:  # fmt: skip
    t0 = time.perf_counter()
    dl = np.empty(len(idx))
    for k, (layer, off) in enumerate(locate(idx, sizes)):
        dl[k] = oracle.delta(layer, np.unravel_index(off, shapes[layer]))
    out: dict = {
        "indices": idx.tolist(),
        "delta_loss": dl.tolist(),
        "n_negative_delta": int((dl < 0).sum()),
        "median_delta": float(np.median(dl)),
        "median_abs_delta": float(np.median(np.abs(dl))),
        "seconds": time.perf_counter() - t0,
        "criteria": {},
    }
    # "signed": the true loss change. Removing a weight can lower validation loss, so signed
    # deltas carry noise-sized sign flips; "abs" asks the criterion's real question, how much
    # does removal move the loss.
    targets = {"signed": dl, "abs": np.abs(dl)}
    for cname, s in scores.items():
        sc = s[idx]
        above = sc >= thresholds[cname]
        cross = above[:, None] ^ above[None, :]
        entry: dict = {"score": sc.tolist()}
        for tname, target in targets.items():
            conc_all, n_all = pair_concordance(sc, target)
            conc_cross, n_cross = pair_concordance(sc, target, cross)
            entry[tname] = {
                "spearman": spearman(target, sc),
                "pair_concordance_all": conc_all,
                "n_pairs_all": n_all,
                "pair_concordance_cross_threshold": conc_cross,
                "n_pairs_cross": n_cross,
            }
        out["criteria"][cname] = entry
    print(f"[{name}] {time.perf_counter() - t0:.0f}s  median |dL| {out['median_abs_delta']:.2e}")
    for cname, e in out["criteria"].items():
        for tname in targets:
            t = e[tname]
            print(
                f"  {cname:9s} vs {tname:6s} spearman {t['spearman']:+.3f}"
                f"  pairs ok {t['pair_concordance_all']:.3f} (all, n={t['n_pairs_all']})"
                f"  {t['pair_concordance_cross_threshold']:.3f}"
                f" (cross-threshold, n={t['n_pairs_cross']})"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations" / "oracle.json")
    args = ap.parse_args()
    data = load_mnist()
    model, crit, test_acc = train_dense_with_fisher(args.seed, data)
    params = model.prunable_parameters()
    scores = {
        "magnitude": flat(Magnitude().scores(params)),
        "fisher": flat(crit.scores(params)),
        "taylor1": flat(Taylor1().scores(params)),
    }
    total = scores["magnitude"].size
    n_keep = total - round(total * SPARSITY)
    thresholds = {}
    for k, s in scores.items():
        thresholds[k] = float(np.sort(s)[::-1][n_keep - 1])
    sizes = [p.data.size for p in params]
    shapes = [p.shape for p in params]
    oracle = Oracle(model, data.x_val, data.y_val)

    rng = np.random.default_rng(1234)
    sets = {
        "random": rng.choice(total, N_PER_SET, replace=False),
        "boundary_m": band_around_threshold(scores["magnitude"], n_keep, N_PER_SET),
        "boundary_f": band_around_threshold(scores["fisher"], n_keep, N_PER_SET),
    }
    result = {
        "seed": args.seed,
        "dense_test_acc": test_acc,
        "val_loss_base": oracle.base,
        "sparsity": SPARSITY,
        "n_keep": n_keep,
        "thresholds": thresholds,
        "sets": {
            name: analyse(name, idx, oracle, sizes, shapes, scores, thresholds)
            for name, idx in sets.items()
        },
        "env": env_info(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
