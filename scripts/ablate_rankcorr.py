"""Rank agreement between the importance scores on the trained dense seed-0 model.

If |w| and the Fisher score rank weights almost identically, the two criteria can only
disagree on a few borderline weights; this measures how much room there is to differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prune import Magnitude, Taylor1
from scripts.ablate_lib import env_info, jaccard, spearman, train_dense_with_fisher
from train.data import load_mnist

ROOT = Path(__file__).resolve().parent.parent
SPARSITY = 0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ablations" / "rank_corr.json")
    args = ap.parse_args()
    model, crit, test_acc = train_dense_with_fisher(args.seed, load_mnist())
    params = model.prunable_parameters()
    per = {
        "magnitude": Magnitude().scores(params),
        "fisher": crit.scores(params),
        "taylor1": Taylor1().scores(params),
    }
    pairs = [("magnitude", "fisher"), ("magnitude", "taylor1"), ("fisher", "taylor1")]
    names = [f"layer{i + 1}" for i in range(len(params))] + ["all"]
    out: dict = {
        "seed": args.seed,
        "dense_test_acc": test_acc,
        "spearman": {},
        "top5pct_jaccard": {},
    }
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        out["spearman"][key], out["top5pct_jaccard"][key] = {}, {}
        groups = [(per[a][i].ravel(), per[b][i].ravel()) for i in range(len(params))]
        groups.append(
            (np.concatenate([g[0] for g in groups]), np.concatenate([g[1] for g in groups]))
        )
        for name, (sa, sb) in zip(names, groups, strict=True):
            k = len(sa) - round(len(sa) * SPARSITY)
            top = lambda s, k=k: np.isin(np.arange(len(s)), np.argsort(-s, kind="stable")[:k])  # noqa: E731
            out["spearman"][key][name] = spearman(sa, sb)
            out["top5pct_jaccard"][key][name] = jaccard(top(sa), top(sb))
    out["env"] = env_info()
    print(f"dense seed {args.seed} test acc {test_acc:.4f}")
    for kind in ("spearman", "top5pct_jaccard"):
        print(kind)
        for key, v in out[kind].items():
            print(f"  {key:22s} " + "  ".join(f"{n} {x:.3f}" for n, x in v.items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
