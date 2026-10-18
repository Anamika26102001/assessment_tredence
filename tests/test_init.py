import numpy as np

from engine import Tensor, no_grad
from nn import Linear, ReLU, Sequential, mlp


def second_moments(model: Sequential, x: np.ndarray) -> list[float]:
    out, h = [], Tensor(x)
    with no_grad():
        for layer in model.layers:
            h = layer(h)
            if isinstance(layer, ReLU):
                out.append(float((h.data**2).mean()))
    return out


def test_he_init_preserves_activation_scale_with_depth() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 512))
    model = mlp([512] * 7 + [10], rng, dtype=np.float64)
    for sm in second_moments(model, x):
        assert 0.7 < sm < 1.4  # E[relu(z)^2] stays ~ E[x^2] = 1 at every depth


def test_naive_init_vanishes_or_explodes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((512, 256))
    for scale, check in ((0.01, lambda s: s < 1e-6), (1.0, lambda s: s > 1e6)):
        layers: list = []
        for _ in range(6):
            lin = Linear(256, 256, rng, dtype=np.float64)
            lin.weight.data[...] = rng.standard_normal((256, 256)) * scale
            layers += [lin, ReLU()]
        assert check(second_moments(Sequential(*layers), x)[-1])


def test_logits_layer_is_glorot_scaled_and_biases_zero() -> None:
    rng = np.random.default_rng(0)
    model = mlp([784, 300, 100, 10], rng)
    last = [layer for layer in model.layers if isinstance(layer, Linear)][-1]
    assert abs(last.weight.data.std() - np.sqrt(2 / 110)) < 0.01
    assert all(not p.data.any() for p in model.parameters() if not p.prunable)
