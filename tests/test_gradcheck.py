"""Finite-difference check for every differentiable op."""

import numpy as np
import pytest

from engine import Tensor, gradcheck, ops, softmax_cross_entropy

rng = np.random.default_rng(0)


def r(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape)


def pos(*shape: int) -> np.ndarray:
    return rng.uniform(0.5, 2.0, shape)


def away_from_zero(*shape: int) -> np.ndarray:
    x = rng.standard_normal(shape)
    return np.where(np.abs(x) < 0.1, 0.5, x)  # ReLU kink: finite differences invalid at 0


BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


@pytest.mark.parametrize("name", BINARY)
def test_binary(name: str) -> None:
    b = pos(3, 4) if name == "div" else r(3, 4)
    gradcheck(BINARY[name], [r(3, 4), b])


def test_neg_and_reflected() -> None:
    gradcheck(lambda a: -a, [r(3, 4)])
    gradcheck(lambda a: 2.0 - a, [r(3)])
    gradcheck(lambda a: 3.0 / a, [pos(3)])


def test_pow() -> None:
    gradcheck(lambda a: a**3, [r(3, 2)])
    gradcheck(lambda a: a**0.5, [pos(3, 2)])


def test_matmul() -> None:
    gradcheck(lambda a, b: a @ b, [r(5, 3), r(3, 4)])


def test_sum_mean_all_axes() -> None:
    for axis in (None, 0, 1, (0, 1)):
        for keepdims in (False, True):
            gradcheck(lambda a, ax=axis, kd=keepdims: ops.sum(a, ax, kd), [r(3, 4)])
            gradcheck(lambda a, ax=axis, kd=keepdims: ops.mean(a, ax, kd), [r(3, 4)])


def test_sum_mean_3d_negative_axis() -> None:
    gradcheck(lambda a: ops.sum(a, -1), [r(2, 3, 4)])
    gradcheck(lambda a: ops.mean(a, (0, 2)), [r(2, 3, 4)])


def test_exp_log() -> None:
    gradcheck(ops.exp, [r(3, 4)])
    gradcheck(ops.log, [pos(3, 4)])


def test_relu() -> None:
    gradcheck(ops.relu, [away_from_zero(4, 5)])


def test_tanh() -> None:
    gradcheck(ops.tanh, [r(4, 5)])


def test_sigmoid() -> None:
    gradcheck(ops.sigmoid, [r(4, 5) * 3])


def test_sigmoid_extreme_inputs_finite() -> None:
    out = ops.sigmoid(Tensor(np.array([-1e4, -50.0, 0.0, 50.0, 1e4])))
    assert np.isfinite(out.data).all()
    np.testing.assert_allclose(out.data, [0, 0, 0.5, 1, 1], atol=1e-12)


def test_softmax_cross_entropy() -> None:
    labels = rng.integers(0, 5, size=6)
    gradcheck(lambda z: softmax_cross_entropy(z, labels), [r(6, 5)])


def test_softmax_cross_entropy_is_stable_for_huge_logits() -> None:
    z = Tensor(np.array([[1000.0, -1000.0, 0.0], [-1e4, 1e4, 0.0]]), requires_grad=True)
    loss = softmax_cross_entropy(z, np.array([0, 1]))
    loss.backward()
    assert np.isfinite(loss.data) and np.isfinite(z.grad).all()
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_softmax_cross_entropy_matches_naive() -> None:
    z = r(7, 4)
    y = rng.integers(0, 4, size=7)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    naive = -np.log(p[np.arange(7), y]).mean()
    assert softmax_cross_entropy(Tensor(z), y).item() == pytest.approx(naive, rel=1e-12)


def test_composed_mlp_layer() -> None:
    labels = rng.integers(0, 3, size=5)

    def f(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return softmax_cross_entropy(ops.tanh(x @ w + b) @ Tensor(np.eye(4)[:, :3]), labels)

    gradcheck(f, [r(5, 6), r(6, 4), r(4)])
