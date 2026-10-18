import numpy as np
import pytest

from nn.module import Parameter
from optim import SGD, Adam


def quad_grad(p: Parameter) -> None:
    p.grad = 2.0 * (p.data - 3.0)


def test_adam_matches_reference_implementation() -> None:
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    p = Parameter(np.array([0.0, 10.0, -4.0]))
    opt = Adam([p], lr=lr)
    w, m, v = p.data.copy(), np.zeros(3), np.zeros(3)
    for t in range(1, 11):
        quad_grad(p)
        opt.step()
        g = 2.0 * (w - 3.0)
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
        np.testing.assert_allclose(p.data, w, rtol=1e-12)


def test_adam_first_step_is_lr_times_sign() -> None:
    p = Parameter(np.array([1.0, -2.0]))
    p.grad = np.array([0.37, -42.0])
    Adam([p], lr=0.05).step()
    np.testing.assert_allclose(p.data, [1.0 - 0.05, -2.0 + 0.05], rtol=1e-6)


def test_adam_converges_on_quadratic() -> None:
    p = Parameter(np.array([0.0, 10.0]))
    opt = Adam([p], lr=0.1)
    for _ in range(500):
        quad_grad(p)
        opt.step()
    np.testing.assert_allclose(p.data, [3.0, 3.0], atol=1e-2)


def test_sgd_momentum_matches_hand_computation() -> None:
    p = Parameter(np.array([1.0]))
    opt = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([1.0])
    opt.step()  # v=1, w=1-0.1
    p.grad = np.array([1.0])
    opt.step()  # v=1.9, w=0.9-0.19
    assert p.data[0] == pytest.approx(0.71)


def test_adam_step_counter_is_per_element() -> None:
    p = Parameter(np.ones((2, 2)))
    opt = Adam([p], lr=0.01)
    p.set_mask(np.array([[True, True], [True, False]]), opt)
    for _ in range(3):
        p.grad = np.ones((2, 2))
        opt.step()
    np.testing.assert_array_equal(opt.t[id(p)], [[3, 3], [3, 0]])
