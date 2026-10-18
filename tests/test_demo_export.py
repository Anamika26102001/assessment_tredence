from __future__ import annotations

import json

import numpy as np
import pytest

from engine import Tensor, no_grad
from nn import mlp
from scripts.export_demo import export_model, forward_exported
from train.data import split_and_normalize

rng = np.random.default_rng(0)


def pruned_model(density: float):
    model = mlp([12, 10, 6, 3], np.random.default_rng(1), dtype=np.float64)
    for p in model.prunable_parameters():
        if density < 1.0:
            p.set_mask(rng.random(p.shape) < density)
    for p in model.parameters():
        if not p.prunable:
            p.data[...] = rng.standard_normal(p.shape)
    return model


@pytest.mark.parametrize("density", [1.0, 0.3, 0.05])
def test_exported_weights_reproduce_the_model(density: float) -> None:
    model = pruned_model(density)
    x = rng.standard_normal((9, 12))
    blob = export_model(model, {"id": "t"}, x)
    with no_grad():
        ref = model(Tensor(x)).data
    np.testing.assert_allclose(forward_exported(blob, x), ref, atol=1e-5)  # 6-decimal rounding
    assert blob["expected"] == ref.argmax(axis=1).tolist()


def test_export_format_is_json_serialisable_and_counts_are_right() -> None:
    model = pruned_model(0.3)
    blob = export_model(model, {"id": "t", "method": "m"}, rng.standard_normal((2, 12)))
    json.dumps(blob)
    alive = sum(int(p.alive().sum()) for p in model.prunable_parameters())
    assert blob["active_params"] == alive and blob["total_params"] == 12 * 10 + 10 * 6 + 6 * 3
    assert blob["flops"] < blob["flops_dense"]
    assert all("w" not in layer for layer in blob["layers"])  # pruned layers are CSR only
    assert len(blob["pixel_survival"]) == 12 and 0 <= min(blob["pixel_survival"]) <= 1


def test_dense_model_is_exported_dense() -> None:
    blob = export_model(pruned_model(1.0), {"id": "d"}, rng.standard_normal((2, 12)))
    assert all("w" in layer and "indptr" not in layer for layer in blob["layers"])
    assert blob["active_params"] == blob["total_params"]


def test_dataset_exposes_raw_train_normalisation() -> None:
    x = np.random.default_rng(0).integers(0, 256, (300, 4)).astype(np.uint8)
    y = np.random.default_rng(1).integers(0, 10, 300)
    d = split_and_normalize(x, y, 150, 50, 100)
    raw = x.astype(np.float32) / 255.0
    # applying the exported constants to raw pixels must give exactly the normalised data
    np.testing.assert_allclose((raw[-100:] - d.mean) / d.std, d.x_test, atol=1e-5)
