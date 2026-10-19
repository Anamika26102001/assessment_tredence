"""Part 2: train the dense 784-300-100-10 MLP with Adam; save learning curve + plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import mlp
from optim import Adam
from train import evaluate, train
from train.data import load_mnist

ROOT = Path(__file__).resolve().parent.parent


def plot_curve(hist: list[dict], path: Path, title: str) -> None:
    ep = [r["epoch"] for r in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.plot(ep, [r["train_loss"] for r in hist], label="train")
    a.plot(ep, [r["val_loss"] for r in hist], label="val")
    a.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    b.plot(ep, [r["train_acc"] for r in hist], label="train")
    b.plot(ep, [r["val_acc"] for r in hist], label="val")
    b.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_mnist()
    rng = np.random.default_rng(args.seed)
    model = mlp([784, 300, 100, 10], rng)
    opt = Adam(model.parameters(), lr=args.lr)
    hist = train(
        model, opt, data.x_train, data.y_train, data.x_val, data.y_val,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
    )  # fmt: skip
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)
    finite = all(np.isfinite(p.data).all() for p in model.parameters())
    print(f"TEST loss {test_loss:.4f} acc {test_acc:.4f}  all-weights-finite={finite}")

    result = {
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "n_weights": int(sum(p.data.size for p in model.prunable_parameters())),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": hist.epochs,
    }
    (args.out / "dense_curve.json").write_text(json.dumps(result, indent=2))
    plot_curve(
        hist.epochs, args.out / "dense_curve.png", f"Dense MLP, Adam lr={args.lr}, seed {args.seed}"
    )


if __name__ == "__main__":
    main()
