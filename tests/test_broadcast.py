"""Broadcasting, gradient accumulation and graph-shape edge cases."""

import numpy as np
import pytest

from engine import Tensor, gradcheck, ops, unbroadcast

rng = np.random.default_rng(1)


def r(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape)


def test_bias_add_ND_plus_D() -> None:
    gradcheck(lambda x, b: x + b, [r(5, 3), r(3)])


def test_bias_grad_is_column_sum() -> None:
    x = Tensor(r(5, 3), requires_grad=True)
    b = Tensor(r(3), requires_grad=True)
    (x + b).sum().backward()
    np.testing.assert_allclose(b.grad, np.full(3, 5.0))
    np.testing.assert_allclose(x.grad, np.ones((5, 3)))


@pytest.mark.parametrize(
    "sa,sb",
    [
        ((4, 1), (1, 3)),
        ((4, 3), (3,)),
        ((4, 3), (1,)),
        ((4, 3), ()),
        ((2, 4, 3), (4, 1)),
        ((3,), (4, 3)),
    ],
)
@pytest.mark.parametrize("op", [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b])
def test_broadcast_shapes(sa: tuple[int, ...], sb: tuple[int, ...], op) -> None:  # type: ignore[no-untyped-def]
    gradcheck(op, [r(*sa), r(*sb)])


def test_div_broadcast() -> None:
    gradcheck(lambda a, b: a / b, [r(4, 3), rng.uniform(1, 2, (3,))])


def test_grad_shapes_match_inputs() -> None:
    a, b = Tensor(r(4, 1), requires_grad=True), Tensor(r(1, 3), requires_grad=True)
    (a * b).sum().backward()
    assert a.grad is not None and a.grad.shape == (4, 1)
    assert b.grad is not None and b.grad.shape == (1, 3)


def test_unbroadcast_direct() -> None:
    g = np.ones((2, 4, 3))
    assert unbroadcast(g, (4, 3)).shape == (4, 3)
    assert unbroadcast(g, (4, 1)).shape == (4, 1)
    assert unbroadcast(g, ()).shape == ()
    np.testing.assert_allclose(unbroadcast(g, (3,)), np.full(3, 8.0))


def test_reused_node_accumulates() -> None:
    x = Tensor(np.array([3.0]), requires_grad=True)
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad, [6.0])


def test_diamond_graph() -> None:
    x = Tensor(np.array([2.0]), requires_grad=True)
    a = x * 3.0
    b = x * 4.0
    (a * b).sum().backward()  # 12 x^2 -> 24 x
    np.testing.assert_allclose(x.grad, [48.0])


def test_gradient_is_not_aliased_between_parents() -> None:
    a = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(np.zeros(3), requires_grad=True)
    c = a + b
    d = c + a  # a receives two contributions; in-place += must not corrupt b
    d.sum().backward()
    np.testing.assert_allclose(a.grad, np.full(3, 2.0))
    np.testing.assert_allclose(b.grad, np.full(3, 1.0))


def test_deep_chain_does_not_hit_recursion_limit() -> None:
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0])


def test_no_grad_builds_no_graph() -> None:
    from engine import no_grad

    x = Tensor(r(3), requires_grad=True)
    with no_grad():
        y = ops.exp(x)
    assert not y.requires_grad and y._parents == ()


def test_constant_inputs_get_no_grad() -> None:
    w = Tensor(r(3, 2), requires_grad=True)
    x = Tensor(r(4, 3))
    (x @ w).sum().backward()
    assert x.grad is None and w.grad is not None


def test_backward_non_scalar_requires_seed() -> None:
    x = Tensor(r(3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
