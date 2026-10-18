"""Train the demo models and bake them into a single self-contained HTML page.

`make demo` writes demo/index.html: draw a digit and watch the dense net and self-pruned
nets (CSR weights, evaluated by plain JavaScript) classify it side by side. The page checks
its own JavaScript forward pass against the Python logits exported here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from engine import Tensor, no_grad
from nn.module import Module
from prune.experiment import PruneConfig, run_pruning_with_model
from prune.sparse import CSR, _layers_of, dense_flops_per_sample
from train.data import Dataset, load_mnist, load_raw_mnist

ROOT = Path(__file__).resolve().parent.parent
SPARSITIES = [0.9, 0.95, 0.98]
METHODS = ["magnitude", "fisher_taylor"]


def export_model(model: Module, meta: dict[str, Any], x_samples: np.ndarray) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    flops = 0
    for lin, act in _layers_of(model):
        w, b = lin.weight, lin.bias
        entry: dict[str, Any] = {
            "n_in": int(w.shape[0]),
            "n_out": int(w.shape[1]),
            "act": act,
            "bias": np.round(b.data.astype(np.float64), 6).tolist(),
        }
        if w.mask is None:
            entry["w"] = np.round(w.data.astype(np.float64), 6).ravel().tolist()  # (in, out)
            flops += 2 * w.data.size + b.data.size
        else:
            csr = CSR.from_dense(w.data, w.alive())  # rows = output units, indices = inputs
            entry["indptr"] = csr.indptr.tolist()
            entry["indices"] = csr.indices.tolist()
            entry["data"] = np.round(csr.data.astype(np.float64), 6).tolist()
            flops += 2 * csr.nnz + b.data.size
        layers.append(entry)
    first = next(iter(model.prunable_parameters()))
    with no_grad():
        logits = model(Tensor(x_samples)).data
    return {
        **meta,
        "flops": int(flops),
        "flops_dense": dense_flops_per_sample(model),
        "active_params": int(sum(p.alive().sum() for p in model.prunable_parameters())),
        "total_params": int(sum(p.data.size for p in model.prunable_parameters())),
        "layers": layers,
        "pixel_survival": np.round(first.alive().mean(axis=1), 3).tolist(),
        "expected": logits.argmax(axis=1).tolist(),
        "expected_logits": np.round(logits.astype(np.float64), 4).tolist(),
    }


def forward_exported(blob: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """NumPy replay of the exported format; mirrors the JavaScript in demo/template.html."""
    h = x.astype(np.float64)
    for layer in blob["layers"]:
        n_in, n_out = layer["n_in"], layer["n_out"]
        out = np.tile(np.array(layer["bias"]), (h.shape[0], 1))
        if "w" in layer:
            out = out + h @ np.array(layer["w"]).reshape(n_in, n_out)
        else:
            ip, idx, dat = layer["indptr"], layer["indices"], layer["data"]
            for j in range(n_out):
                lo, hi = ip[j], ip[j + 1]
                out[:, j] += h[:, idx[lo:hi]] @ np.array(dat[lo:hi])
        h = np.maximum(out, 0) if layer["act"] == "relu" else out
    return h


def pick_samples(data: Dataset, per_class: int = 2) -> np.ndarray:
    idx = []
    for c in range(10):
        idx += list(np.flatnonzero(data.y_test == c)[:per_class])
    return np.array(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "index.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_mnist()
    raw_x, _ = load_raw_mnist()
    idx = pick_samples(data)
    x_norm = data.x_test[idx]
    samples = [
        {"label": int(data.y_test[i]), "pixels": raw_x[-len(data.x_test) :][i].tolist()}
        for i in idx
    ]

    runs_dir = ROOT / "results" / "runs"
    models = []
    plan = [("dense", 0.0)] + [(m, s) for m in METHODS for s in SPARSITIES]
    for method, s in plan:
        crit = "magnitude" if method == "dense" else method
        cfg = PruneConfig(criterion=crit, sparsity=s, seed=args.seed, ramp_end_epoch=8)
        res, model = run_pruning_with_model(cfg, data)
        from scripts.sweep import run_key

        ref = runs_dir / f"{run_key(cfg)}.json"
        note = ""
        if ref.exists():
            same = json.loads(ref.read_text())["test_acc"] == res["test_acc"]
            note = (
                "  (matches committed sweep run)"
                if same
                else "  (DIFFERS from committed sweep run!)"
            )
        print(f"{method:14s} {s:.2f} test acc {res['test_acc']:.4f}{note}", flush=True)
        meta = {
            "id": f"{method}-{s}",
            "method": method,
            "sparsity": res["achieved_sparsity"],
            "test_acc": res["test_acc"],
            "layer_sparsity": res["layer_sparsity"],
        }
        models.append(export_model(model, meta, x_norm))

    blob = {"norm": {"mean": data.mean, "std": data.std}, "samples": samples, "models": models}
    template = (ROOT / "demo" / "template.html").read_text()
    args.out.write_text(template.replace("__DATA__", json.dumps(blob, separators=(",", ":"))))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) - open it in a browser")


if __name__ == "__main__":
    main()
