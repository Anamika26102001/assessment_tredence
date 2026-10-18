from __future__ import annotations

import numpy as np
import pytest

from engine import Tensor, no_grad
from nn import Sequential, Tanh, mlp
from nn.layers import Linear, ReLU
from prune.sparse import CSR, SparseMLP, compact, dense_flops_per_sample

rng = np.random.default_rng(0)


def dense_forward(model, x: np.ndarray) -> np.ndarray:
    with no_grad():
        return model(Tensor(x)).data


def masked_model(sizes, density: float, seed: int = 0, act=ReLU):
    r = np.random.default_rng(seed)
    model = mlp(sizes, r, activation=act, dtype=np.float64)
    for lin in (m for m in model.layers if isinstance(m, Linear)):
        lin.bias.data[...] = r.standard_normal(lin.bias.shape)  # nonzero biases
        lin.weight.set_mask(r.random(lin.weight.shape) < density)
    return model


@pytest.mark.parametrize("density", [1.0, 0.5, 0.05, 0.0])
def test_csr_matmul_matches_dense(density: float) -> None:
    w = rng.standard_normal((20, 15))
    alive = rng.random(w.shape) < density
    x = rng.standard_normal((7, 20))
    np.testing.assert_allclose(
        CSR.from_dense(w * alive, alive).matmul(x), x @ (w * alive), atol=1e-12
    )


def test_csr_handles_empty_rows_and_columns() -> None:
    w = rng.standard_normal((10, 8))
    alive = np.ones_like(w, dtype=bool)
    alive[:, [0, 3, 7]] = False  # output units with no inputs (empty CSR rows, incl. last)
    alive[[2, 5], :] = False  # inputs that feed nothing
    x = rng.standard_normal((4, 10))
    out = CSR.from_dense(w * alive, alive).matmul(x)
    np.testing.assert_allclose(out, x @ (w * alive), atol=1e-12)
    assert not out[:, [0, 3, 7]].any()


def test_csr_stores_only_alive_weights() -> None:
    w = rng.standard_normal((30, 20))
    alive = rng.random(w.shape) < 0.1
    csr = CSR.from_dense(w * alive, alive)
    assert csr.nnz == alive.sum() and csr.indptr[-1] == alive.sum()


@pytest.mark.parametrize("act", [ReLU, Tanh])
@pytest.mark.parametrize("density", [1.0, 0.3, 0.05])
def test_sparse_forward_equals_dense_masked_forward(act, density: float) -> None:
    model = masked_model([12, 16, 10, 4], density, act=act)
    x = rng.standard_normal((9, 12))
    np.testing.assert_allclose(SparseMLP(model).forward(x), dense_forward(model, x), atol=1e-10)


@pytest.mark.parametrize("act", [ReLU, Tanh])
def test_compaction_is_exact_and_removes_dead_units(act) -> None:
    model = masked_model([12, 16, 10, 4], 0.1, act=act)
    # Force specific structure: hidden unit 3 of layer 0 has no inputs (constant act(b),
    # must be folded into the next bias), unit 5 has no outgoing weights, input 1 unused.
    l0, l1 = model.layers[0], model.layers[2]
    m0, m1 = l0.weight.alive().copy(), l1.weight.alive().copy()
    m0[:, 3] = False
    m1[3, 0] = True  # unit 3 *does* feed the next layer
    m1[5, :] = False
    m0[1, :] = False
    l0.weight.set_mask(m0)
    l1.weight.set_mask(m1)
    l1.weight.data[3, 0] = 0.7
    l0.bias.data[3] = 1.3
    x = rng.standard_normal((11, 12))
    c = compact(model)
    np.testing.assert_allclose(c.forward(x), dense_forward(model, x), atol=1e-10)
    assert c.shapes[0][0] < 12 and c.shapes[0][1] < 16  # fewer inputs and hidden units
    assert 1 not in c.input_idx


def test_compaction_cascades_to_fixpoint() -> None:
    model = masked_model([6, 5, 5, 3], 1.0)
    l0, l1, l2 = model.layers[0], model.layers[2], model.layers[4]
    # Kill every outgoing weight of layer-1 unit 0 -> unit 0 unused -> layer-0 units that
    # only fed unit 0 become unused too.
    m0 = np.zeros((6, 5), dtype=bool)
    m0[:, 0] = True
    m0[:, 1] = True  # layer-0 unit 0 and 1 are kept alive only by feeding l1 unit 0 / 1
    m1 = np.zeros((5, 5), dtype=bool)
    m1[0, 0] = True  # l0 unit 0 -> l1 unit 0
    m1[1, 1] = True  # l0 unit 1 -> l1 unit 1
    m2 = np.zeros((5, 3), dtype=bool)
    m2[1, :] = True  # only l1 unit 1 reaches the output
    l0.weight.set_mask(m0)
    l1.weight.set_mask(m1)
    l2.weight.set_mask(m2)
    x = rng.standard_normal((5, 6))
    c = compact(model)
    np.testing.assert_allclose(c.forward(x), dense_forward(model, x), atol=1e-10)
    assert [s[1] for s in c.shapes[:2]] == [1, 1]  # only the unit-1 chain survives
    assert c.shapes[2][1] == 3  # logit units are never removed


def test_all_weights_removed_still_runs() -> None:
    model = masked_model([4, 3, 2], 0.0)
    x = rng.standard_normal((3, 4))
    np.testing.assert_allclose(compact(model).forward(x), dense_forward(model, x), atol=1e-10)
    np.testing.assert_allclose(SparseMLP(model).forward(x), dense_forward(model, x), atol=1e-10)


def test_flop_and_param_counts() -> None:
    model = masked_model([20, 10, 5], 0.2)
    sp = SparseMLP(model)
    nnz = sum(int(m.weight.alive().sum()) for m in model.layers if isinstance(m, Linear))
    n_bias = 10 + 5
    assert sp.flops_per_sample == 2 * nnz + n_bias
    assert sp.n_active_params == nnz + n_bias
    assert dense_flops_per_sample(model) == 2 * (20 * 10 + 10 * 5) + n_bias
    assert sp.flops_per_sample < dense_flops_per_sample(model)


def test_unsupported_layer_rejected() -> None:
    class Weird(ReLU):
        pass

    # subclasses of supported layers are fine; a bare Module is not
    from nn.module import Module

    class Bare(Module):
        def forward(self, x):
            return x

    with pytest.raises(TypeError):
        SparseMLP(Sequential(Linear(2, 2, rng), Bare()))
