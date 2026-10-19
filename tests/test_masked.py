"""Part 1 correctness requirement: the engine and optimizers stay correct when
connections are masked out, pruned and revived."""

from __future__ import annotations

import numpy as np
import pytest

from engine import Tensor, no_grad, softmax_cross_entropy
from nn import Linear, ReLU, Sequential
from nn.module import Parameter
from optim import SGD, Adam
from optim.base import Optimizer


def make(seed: int = 0, density: float = 0.5):
    rng = np.random.default_rng(seed)
    lin = Linear(6, 4, rng, dtype=np.float64)
    mask = rng.random((6, 4)) < density
    lin.weight.set_mask(mask)
    x = rng.standard_normal((8, 6))
    y = rng.integers(0, 4, 8)
    return lin, mask, x, y


def loss_of(model, x, y) -> Tensor:
    return softmax_cross_entropy(model(Tensor(x)), y)


def numeric_grad(model, param: Parameter, x, y, eps: float = 1e-6) -> np.ndarray:
    num = np.zeros_like(param.data)
    for i in np.ndindex(*param.shape):
        old = param.data[i]
        param.data[i] = old + eps
        with no_grad():
            hi = loss_of(model, x, y).item()
        param.data[i] = old - eps
        with no_grad():
            lo = loss_of(model, x, y).item()
        param.data[i] = old
        num[i] = (hi - lo) / (2 * eps)
    return num


# --------------------------------------------------------------- gradient semantics
def test_masked_weights_get_exactly_zero_gradient() -> None:
    lin, mask, x, y = make()
    loss_of(lin, x, y).backward()
    g = lin.weight.grad
    assert g is not None
    assert np.array_equal(g[~mask], np.zeros((~mask).sum()))


def test_gradient_matches_finite_differences_including_masked_entries() -> None:
    lin, mask, x, y = make()
    loss_of(lin, x, y).backward()
    num = numeric_grad(lin, lin.weight, x, y)
    assert np.array_equal(num[~mask], np.zeros((~mask).sum()))  # FD agrees: masked -> 0
    np.testing.assert_allclose(lin.weight.grad, num, atol=1e-7)
    np.testing.assert_allclose(lin.bias.grad, numeric_grad(lin, lin.bias, x, y), atol=1e-7)


def test_masked_weight_value_cannot_affect_forward() -> None:
    lin, mask, x, y = make()
    before = lin(Tensor(x)).data.copy()
    lin.weight.data[~mask] = 1e6  # garbage hidden behind the mask
    np.testing.assert_array_equal(lin(Tensor(x)).data, before)


def test_dense_counterfactual_gradient_equals_dense_net_gradient() -> None:
    """dL/dW_eff at masked entries = gradient a dense layer with the same W_eff would see.
    This is what regrowth reads; the optimizer never sees it."""
    lin, mask, x, y = make()
    loss_of(lin, x, y).backward()
    dense = Linear(6, 4, np.random.default_rng(1), dtype=np.float64)
    dense.weight.data[...] = lin.weight.data * mask
    dense.bias.data[...] = lin.bias.data
    loss_of(dense, x, y).backward()
    dg = lin.weight.dense_grad()
    assert dg is not None
    np.testing.assert_allclose(dg, dense.weight.grad, atol=1e-12)
    assert np.abs(dg[~mask]).max() > 0  # informative where the real gradient is zero


# ------------------------------------------------------------------- frozen state
@pytest.mark.parametrize("opt_cls", [Adam, SGD])
def test_masked_entries_stay_bit_exact_zero_with_zero_state(opt_cls) -> None:
    lin, mask, x, y = make()
    opt = opt_cls(lin.parameters(), lr=0.01)
    first = loss_of(lin, x, y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_of(lin, x, y)
        loss.backward()
        opt.step()
    assert loss.item() < first
    w = lin.weight
    assert np.array_equal(w.data[~mask], np.zeros((~mask).sum()))
    state = [opt.m, opt.v] if isinstance(opt, Adam) else [opt.vel]
    for s in state:
        assert np.array_equal(s[id(w)][~mask], np.zeros((~mask).sum()))
    if isinstance(opt, Adam):
        assert np.array_equal(opt.t[id(w)][~mask], np.zeros((~mask).sum(), dtype=int))
        assert (opt.t[id(w)][mask] == 60).all()


# --------------------------------------------------------------------- revival
def run_prune_revive(opt_factory, reset_on_change: bool):
    """Train, prune one connection, train, revive it, take one step. Returns
    (weight after pruned steps, delta of first step after revival, lr, grad at revival)."""
    lin, mask, x, y = make(density=0.9)
    opt = opt_factory(lin.parameters())
    w = lin.weight
    e = tuple(np.argwhere(mask)[0])

    def train(n: int) -> None:
        for _ in range(n):
            opt.zero_grad()
            loss_of(lin, x, y).backward()
            opt.step()

    train(30)  # builds up momentum / second moment on e
    pruned = mask.copy()
    pruned[e] = False
    w.set_mask(pruned, opt if reset_on_change else None)
    train(10)
    w_while_pruned = w.data[e]
    revived = pruned.copy()
    revived[e] = True
    w.set_mask(revived, opt if reset_on_change else None)
    opt.zero_grad()
    loss_of(lin, x, y).backward()
    assert w.grad is not None
    g = w.grad[e]
    before = w.data[e]
    opt.step()
    return w_while_pruned, w.data[e] - before, g, opt, w, e


def test_revived_connection_takes_a_fresh_adam_first_step() -> None:
    lr = 0.01
    w_pruned, delta, g, opt, w, e = run_prune_revive(lambda ps: Adam(ps, lr=lr), True)
    assert w_pruned == 0.0
    assert g != 0.0
    assert delta == pytest.approx(-lr * np.sign(g), rel=1e-6)
    assert opt.t[id(w)][e] == 1
    assert opt.m[id(w)][e] == pytest.approx((1 - 0.9) * g)


def test_negative_control_naive_adam_corrupts_pruned_and_revived_weights() -> None:
    """Same scenario with Adam that has a global step counter, updates moments everywhere
    and never resets state. Shows the failure modes the real optimizer avoids."""

    class NaiveAdam(Optimizer):
        def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
            super().__init__(params)
            self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
            self.m = {id(p): np.zeros_like(p.data) for p in self.params}
            self.v = {id(p): np.zeros_like(p.data) for p in self.params}

        def step(self) -> None:
            self.t += 1
            for p in self.params:
                m, v = self.m[id(p)], self.v[id(p)]
                m[...] = self.b1 * m + (1 - self.b1) * p.grad
                v[...] = self.b2 * v + (1 - self.b2) * p.grad**2
                mh, vh = m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)
                p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

        def reset_entries(self, param, entries) -> None:  # deliberately forgets nothing
            pass

    lr = 0.01
    w_pruned, delta, g, _, _, _ = run_prune_revive(lambda ps: NaiveAdam(ps, lr=lr), False)
    assert w_pruned != 0.0  # (a) stale momentum drags the "pruned" weight off zero
    assert abs(delta / (-lr * np.sign(g)) - 1) > 0.05  # (b) revived step is not a fresh step


def test_pruning_resets_optimizer_state_and_zeroes_weight() -> None:
    lin, mask, x, y = make(density=1.0)
    opt = Adam(lin.parameters(), lr=0.01)
    for _ in range(5):
        opt.zero_grad()
        loss_of(lin, x, y).backward()
        opt.step()
    w = lin.weight
    assert (opt.m[id(w)] != 0).all()
    new = mask.copy()
    new[0, 0] = False
    w.set_mask(new, opt)
    assert w.data[0, 0] == 0.0
    assert opt.m[id(w)][0, 0] == 0.0 and opt.v[id(w)][0, 0] == 0.0 and opt.t[id(w)][0, 0] == 0
    assert opt.t[id(w)][1, 1] == 5  # untouched entries keep their history


def test_sgd_velocity_cleared_on_revival() -> None:
    lin, mask, x, y = make(density=0.9)
    opt = SGD(lin.parameters(), lr=0.01, momentum=0.9)
    for _ in range(5):
        opt.zero_grad()
        loss_of(lin, x, y).backward()
        opt.step()
    e = tuple(np.argwhere(mask)[0])
    off = mask.copy()
    off[e] = False
    lin.weight.set_mask(off, opt)
    assert opt.vel[id(lin.weight)][e] == 0.0
    on = off.copy()
    on[e] = True
    lin.weight.set_mask(on, opt)
    opt.zero_grad()
    loss_of(lin, x, y).backward()
    g = lin.weight.grad[e]
    opt.step()
    assert lin.weight.data[e] == pytest.approx(-0.01 * g)  # velocity restarted from 0


# ----------------------------------------------------- masked == physically smaller
def test_masked_network_trains_identically_to_physically_smaller_network() -> None:
    """Remove hidden units by masking vs. by deleting them: same losses, same weights.
    Proves masking is an honest removal, not an approximation."""
    rng = np.random.default_rng(0)
    keep = np.array([0, 2, 3, 6, 7])
    small = Sequential(
        Linear(5, len(keep), rng, dtype=np.float64),
        ReLU(),
        Linear(len(keep), 3, rng, dtype=np.float64),
    )
    big = Sequential(
        Linear(5, 8, rng, dtype=np.float64), ReLU(), Linear(8, 3, rng, dtype=np.float64)
    )
    s1, s2 = small.layers[0], small.layers[2]
    b1, b2 = big.layers[0], big.layers[2]
    b1.weight.data[:, keep] = s1.weight.data
    b1.bias.data[keep] = s1.bias.data
    b2.weight.data[keep, :] = s2.weight.data
    b2.bias.data[...] = s2.bias.data
    m1 = np.zeros((5, 8), dtype=bool)
    m1[:, keep] = True
    m2 = np.zeros((8, 3), dtype=bool)
    m2[keep, :] = True
    b1.weight.set_mask(m1)
    b2.weight.set_mask(m2)

    opt_s, opt_b = Adam(small.parameters(), lr=0.01), Adam(big.parameters(), lr=0.01)
    data_rng = np.random.default_rng(5)
    for _ in range(25):
        x, y = data_rng.standard_normal((16, 5)), data_rng.integers(0, 3, 16)
        losses = []
        for model, opt in ((small, opt_s), (big, opt_b)):
            opt.zero_grad()
            loss = loss_of(model, x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[0] == pytest.approx(losses[1], abs=1e-12)

    np.testing.assert_allclose(b1.weight.data[:, keep], s1.weight.data, atol=1e-10)
    np.testing.assert_allclose(b1.bias.data[keep], s1.bias.data, atol=1e-10)
    np.testing.assert_allclose(b2.weight.data[keep, :], s2.weight.data, atol=1e-10)
    np.testing.assert_allclose(b2.bias.data, s2.bias.data, atol=1e-10)
    assert not b1.weight.data[:, np.setdiff1d(np.arange(8), keep)].any()  # still real zeros
