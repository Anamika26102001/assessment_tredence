"""Part 3: train the MLP so it prunes itself to a target sparsity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prune.experiment import PruneConfig, run_pruning
from train.data import load_mnist

ROOT = Path(__file__).resolve().parent.parent


def plot_run(res: dict, path: Path) -> None:
    hist = res["history"]
    ep = [r["epoch"] for r in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(ep, [r["val_acc"] for r in hist], label="val acc", color="C0")
    a.plot(ep, [r["train_acc"] for r in hist], label="train acc", color="C1")
    a.set(xlabel="epoch", ylabel="accuracy", title="Accuracy while pruning")
    a.legend(loc="lower left")
    a.grid(alpha=0.3)
    b.plot(ep, [r["sparsity"] for r in hist], color="C3", marker="o")
    b.set(xlabel="epoch", ylabel="sparsity (fraction of weights removed)", title="Sparsity ramp")
    b.grid(alpha=0.3)
    c = res["config"]
    fig.suptitle(
        f"{c['criterion']} -> {c['sparsity']:.0%}, {c['schedule']}, regrow={c['regrow']}, "
        f"seed {c['seed']}: test acc {res['test_acc']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", default="fisher_taylor",
                    choices=["magnitude", "taylor1", "fisher_taylor"])  # fmt: skip
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", default="cubic", choices=["cubic", "oneshot"])
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    cfg = PruneConfig(
        criterion=args.criterion, sparsity=args.sparsity, seed=args.seed,
        schedule=args.schedule, regrow=args.regrow, epochs=args.epochs,
        ramp_end_epoch=min(8, args.epochs - 2),
    )  # fmt: skip
    res = run_pruning(cfg, load_mnist(), verbose=True)
    print(
        f"\nTEST acc {res['test_acc']:.4f}  achieved sparsity {res['achieved_sparsity']:.4%} "
        f"({res['n_active']:,}/{res['n_weights']:,} weights active)  "
        f"layer sparsity {[round(s, 3) for s in res['layer_sparsity']]}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"prune_{cfg.criterion}_{cfg.sparsity}_{cfg.schedule}_r{cfg.regrow}_s{cfg.seed}"
    (args.out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    plot_run(res, args.out / f"{stem}.png")


if __name__ == "__main__":
    main()
