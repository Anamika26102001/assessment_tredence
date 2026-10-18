"""Pareto curve and paired-difference plots from the committed CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STYLE = {
    "magnitude": ("C1", "o", "magnitude"),
    "fisher_taylor": ("C0", "s", "fisher_taylor (ours)"),
    "fisher_taylor+regrow": ("C2", "^", "fisher_taylor + regrowth"),
    "taylor1": ("C4", "v", "taylor1 |w*g| (ablation)"),
    "fisher_taylor+oneshot": ("C3", "x", "fisher_taylor one-shot (ablation)"),
}


def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    rows = read(args.results / "pareto.csv")
    dense = next(r for r in rows if r["method"] == "dense")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, zoom in zip(axes, (False, True), strict=True):
        for method, (color, marker, label) in STYLE.items():
            pts = [r for r in rows if r["method"] == method]
            if not pts:
                continue
            if method in ("magnitude", "fisher_taylor"):
                pts = [dense, *pts]  # every curve starts from the 0% dense reference
            pts.sort(key=lambda r: float(r["achieved_sparsity"]))
            x = [float(r["achieved_sparsity"]) * 100 for r in pts]
            y = [float(r["test_acc_mean"]) * 100 for r in pts]
            e = [float(r["test_acc_std"]) * 100 for r in pts]
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, capsize=3, label=label,
                        ls="-" if method in ("magnitude", "fisher_taylor") else "--", alpha=0.9)  # fmt: skip
        ax.axhline(
            float(dense["test_acc_mean"]) * 100, color="gray", ls=":", label="dense reference"
        )
        ax.set(xlabel="achieved sparsity (% of weights removed)", ylabel="test accuracy (%)")
        ax.grid(alpha=0.3)
        if zoom:
            ax.set_xlim(88, 99)
            ax.set_title("Zoom: high sparsity")
        else:
            ax.set_title("Sparsity - accuracy Pareto curve (mean +/- std over seeds)")
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.results / "pareto.png", dpi=140)
    plt.close(fig)

    paired = [r for r in read(args.results / "paired_stats.csv")
              if r["a"] == "fisher_taylor" and r["b"] == "magnitude"]  # fmt: skip
    if paired:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = [f"{float(r['sparsity']):.0%}" for r in paired]
        d = [float(r["mean_diff"]) * 100 for r in paired]
        lo = [d[i] - float(r["ci_lo"]) * 100 for i, r in enumerate(paired)]
        hi = [float(r["ci_hi"]) * 100 - d[i] for i, r in enumerate(paired)]
        ax.bar(labels, d, yerr=[lo, hi], capsize=4, color="C0", alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="target sparsity", ylabel="fisher_taylor - magnitude (pp test acc)",
               title="Paired difference, 95% bootstrap CI")  # fmt: skip
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(args.results / "paired_diff.png", dpi=140)


if __name__ == "__main__":
    main()
