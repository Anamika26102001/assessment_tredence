from __future__ import annotations

import numpy as np
import pytest

from engine import Tensor, no_grad, softmax_cross_entropy
from nn import mlp
from nn.module import Parameter
from optim import Adam
from prune import (
    CubicSchedule,
    FisherTaylor,
    Magnitude,
    OneShotSchedule,
    Pruner,
    Taylor1,
)
from tests.test_train import two_spirals
from train import evaluate, train


# ------------------------------------------------------------------- schedule
def test_cubic_schedule_endpoints_monotone_and_concave() -> None:
    s = CubicSchedule(0.9, start_step=100, end_step=1100, interval=100)
    assert s.events == list(range(100, 1101, 100))
    assert s.sparsity_at(100) == pytest.approx(0.0)
    assert s.sparsity_at(1100) == pytest.approx(0.9)
    assert s.sparsity_at(5000) == pytest.approx(0.9)  # clamps after the ramp
    vals = [s.sparsity_at(t) for t in s.events]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))
    inc = np.diff(vals)
    assert all(b <= a + 1e-12 for a, b in zip(inc, inc[1:], strict=False))  # slows down
    mid = 0.9 + (0 - 0.9) * (1 - 0.5) ** 3
    assert s.sparsity_at(600) == pytest.approx(mid)


def test_oneshot_schedule() -> None:
    s = OneShotSchedule(0.8, step=50)
    assert s.sparsity_at(49) == 0.0 and s.sparsity_at(50) == 0.8 and s.events == [50]


# ------------------------------------------------------------- pruner fixtures
def tiny(seed: int = 0, sizes=(2, 64, 64, 2)):
    x, y = two_spirals(1200, 0)
    model = mlp(list(sizes), np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=3e-3)
    return model, opt, (x[:1000], y[:1000], x[1000:], y[1000:])


def fit(model, opt, d, pr, epochs=4, **kw):
    """32 steps/epoch at batch 32, so 4 epochs (128 steps) cover all events (10..110)."""
    return train(model, opt, *d, epochs=epochs, batch_size=32, callback=pr, verbose=False, **kw)


def make_pruner(model, opt, crit=None, final=0.8, regrow=0.0, **kw) -> Pruner:
    sched = CubicSchedule(final, start_step=10, end_step=10 + 5 * 20, interval=20)
    return Pruner(model, opt, crit or Magnitude(), sched, regrow_frac=regrow, **kw)


def test_exact_budget_even_with_all_tied_scores() -> None:
    model, opt, _ = tiny()
    for p in model.prunable_parameters():
        p.data[...] = 0.5  # every score identical: ties must still give an exact count
    pr = make_pruner(model, opt)
    pr.event(10)
    assert pr.n_alive() == pr.total
    pr.event(110)
    assert pr.n_alive() == pr.total - round(pr.total * 0.8)
    assert pr.sparsity() == pytest.approx(0.8, abs=1e-3)


def test_global_ranking_keeps_largest_magnitudes() -> None:
    model, opt, _ = tiny()
    pr = make_pruner(model, opt, final=0.9, floor_frac=0.0)
    allw = np.concatenate([np.abs(p.data).ravel() for p in pr.params])
    pr.event(110)
    kept = np.concatenate([np.abs(p.data)[p.alive()].ravel() for p in pr.params])
    n_keep = len(kept)
    assert kept.min() >= np.sort(allw)[-n_keep] - 1e-12


def test_per_layer_floor_keeps_tiny_layer_alive() -> None:
    model, opt, _ = tiny()
    model.layers[0].weight.data *= 1e-6  # layer 0 looks worthless to a global ranking
    pr = make_pruner(model, opt, final=0.95, floor_frac=0.02)
    pr.event(110)
    p0 = pr.params[0]
    assert p0.alive().sum() >= int(np.ceil(0.02 * p0.data.size))


def test_prune_zeroes_weights_resets_state_and_never_revives() -> None:
    model, opt, d = tiny()
    pr = make_pruner(model, opt, final=0.8)
    fit(model, opt, d, pr)
    assert pr.log and pr.sparsity() == pytest.approx(0.8, abs=1e-3)  # events really fired
    for p in pr.params:
        assert not p.data[~p.alive()].any()
        assert not opt.m[id(p)][~p.alive()].any() and not opt.v[id(p)][~p.alive()].any()
        assert not opt.t[id(p)][~p.alive()].any()
    # masks only shrink without regrowth
    alive_before = [p.alive().copy() for p in pr.params]
    assert pr.log[-1]["regrown"] == 0
    fit(model, opt, d, pr, epochs=1, start_epoch=4, start_step=128)
    for p, a in zip(pr.params, alive_before, strict=True):
        assert not (p.alive() & ~a).any()


def test_biases_are_never_pruned() -> None:
    model, opt, d = tiny()
    biases = [p for p in model.parameters() if not p.prunable]
    pr = make_pruner(model, opt)
    fit(model, opt, d, pr)
    assert pr.sparsity() > 0.5  # pruning happened, and still no bias was masked
    assert all(p.mask is None for p in biases)


# ---------------------------------------------------------------------- regrowth
def test_regrowth_swaps_without_changing_sparsity_and_starts_fresh() -> None:
    model, opt, d = tiny()
    pr = make_pruner(model, opt, final=0.8, regrow=0.2)
    fit(model, opt, d, pr)
    events = pr.log
    assert any(e["regrown"] > 0 for e in events)
    for e in events:
        assert e["achieved_sparsity"] == pytest.approx(e["target_sparsity"], abs=2e-3)

    # do one more swap by hand and inspect the regrown entries
    before = [p.alive().copy() for p in pr.params]
    step = pr.log[-1]["step"] + 20
    pr.schedule = CubicSchedule(0.8, 10, 10 + 5 * 20, 20)  # progress<1 so k>0
    n_alive = pr.n_alive()
    pr._regrow(10 + 20, Magnitude().scores(pr.params))
    assert pr.n_alive() == n_alive
    grown_total = 0
    for p, b in zip(pr.params, before, strict=True):
        grown = p.alive() & ~b
        grown_total += int(grown.sum())
        assert not p.data[grown].any()
        assert not opt.m[id(p)][grown].any() and not opt.v[id(p)][grown].any()
        assert not opt.t[id(p)][grown].any()
    assert grown_total > 0
    assert step > 0


def test_regrow_picks_dead_weights_with_largest_dense_gradient_and_spares_dropped() -> None:
    model, opt, _ = tiny()
    pr = make_pruner(model, opt, final=0.5, regrow=0.01)
    pr.event(110)
    rng = np.random.default_rng(0)
    for acc in pr.dense_g2:
        acc[...] = rng.random(acc.shape).astype(acc.dtype)
    before = [p.alive().copy() for p in pr.params]
    g2 = np.concatenate([a.ravel() for a in pr.dense_g2])  # snapshot: regrow resets grown
    pr.schedule = CubicSchedule(0.5, 10, 10 + 5 * 20, 20)
    pr._regrow(30, Magnitude().scores(pr.params))
    grown = np.concatenate(
        [(p.alive() & ~b).ravel() for p, b in zip(pr.params, before, strict=True)]
    )
    dropped = np.concatenate(
        [(~p.alive() & b).ravel() for p, b in zip(pr.params, before, strict=True)]
    )
    dead_before = np.concatenate([(~b).ravel() for b in before])
    assert grown.sum() == dropped.sum() > 0
    assert not (grown & dropped).any()
    assert g2[grown].min() >= np.sort(g2[dead_before])[-int(grown.sum())] - 1e-12


def test_regrow_never_revives_weights_pruned_in_the_same_event() -> None:
    model, opt, _ = tiny()
    pr = make_pruner(model, opt, final=0.5, regrow=0.2, floor_frac=0.0)
    pr.event(10)
    pr.event(30)  # some weights are now dead: valid regrowth candidates
    flat_alive = np.concatenate([p.alive().ravel() for p in pr.params])
    flat_w = np.abs(np.concatenate([p.data.ravel() for p in pr.params]))
    n_drop = flat_alive.sum() - (pr.total - round(pr.total * pr.schedule.sparsity_at(50)))
    weakest = np.argsort(np.where(flat_alive, flat_w, np.inf), kind="stable")[:n_drop]
    dead = np.flatnonzero(~flat_alive)
    flat_g2 = np.zeros(pr.total)
    flat_g2[weakest] = 1e3  # largest dense gradient, but they are pruned in this very event
    flat_g2[dead] = 1.0
    off = 0
    for acc in pr.dense_g2:
        acc[...] = flat_g2[off : off + acc.size].reshape(acc.shape)
        off += acc.size
    pr.event(50)
    alive = np.concatenate([p.alive().ravel() for p in pr.params])
    assert pr.log[-1]["regrown"] > 0
    assert alive[dead].any()
    assert not alive[weakest].any()


def test_grace_period_protects_regrown_weights() -> None:
    model, opt, _ = tiny()
    pr = make_pruner(model, opt, final=0.5, regrow=0.0, grace_events=2, floor_frac=0.0)
    pr.event(110)
    p = pr.params[1]
    cand = np.argwhere(p.alive())[:5]
    for i in map(tuple, cand):
        pr.born[1][i] = 100  # "regrown" at step 100
        p.data[i] = 0.0  # worst possible score
    pr.event(110 + 20)  # age 30 <= 2*20 -> protected
    assert all(p.alive()[tuple(i)] for i in cand)


# ------------------------------------------------------------------ criteria
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_fisher_taylor_tracks_true_loss_increase_better_than_magnitude() -> None:
    """Empirical check of the derivation: leave-one-out loss increase from zeroing a
    weight correlates with the Fisher-Taylor score more than with |w|."""
    x, y = two_spirals(1200, 0)
    xt, yt = x[:1000], y[:1000]
    model = mlp([2, 64, 64, 2], np.random.default_rng(0))
    opt = Adam(model.parameters(), lr=3e-3)
    crit = FisherTaylor(beta=0.99)
    pr = Pruner(model, opt, crit, CubicSchedule(0.5, 10**9, 10**9 + 1, 1))  # never fires
    train(model, opt, xt, yt, x[1000:], y[1000:], epochs=25, callback=pr, verbose=False)

    def full_loss() -> float:
        with no_grad():
            return softmax_cross_entropy(model(Tensor(xt)), yt).item()

    base = full_loss()
    params = model.prunable_parameters()
    f_scores = crit.scores(params)
    m_scores = Magnitude().scores(params)
    rng = np.random.default_rng(1)
    true, fs, ms = [], [], []
    for _ in range(300):
        li = int(rng.integers(len(params)))
        idx = tuple(int(rng.integers(n)) for n in params[li].shape)
        old = params[li].data[idx]
        params[li].data[idx] = 0.0
        true.append(full_loss() - base)
        params[li].data[idx] = old
        fs.append(f_scores[li][idx])
        ms.append(m_scores[li][idx])
    rho_f, rho_m = spearman(np.array(true), np.array(fs)), spearman(np.array(true), np.array(ms))
    print(f"spearman with true loss increase: fisher_taylor {rho_f:.3f}, magnitude {rho_m:.3f}")
    assert rho_f > rho_m
    assert rho_f > 0.3


def test_fisher_ema_is_bias_corrected_per_entry() -> None:
    crit = FisherTaylor(beta=0.99)
    w = Parameter(np.ones(2))
    for step in range(300):
        w.grad = np.full(2, 0.5)  # identical gradient history for both entries
        crit.update([w])
        if step == 249:  # entry 1 is revived: its statistics start from nothing
            crit.reset_entries(w, np.array([False, True]))
    established, revived = crit.scores([w])[0]
    assert established == pytest.approx(0.25)  # w^2 * g^2 for a constant gradient
    assert revived == pytest.approx(established, rel=1e-6)


def test_taylor1_and_magnitude_scores_shapes() -> None:
    model, opt, d = tiny()
    train(model, opt, *d, epochs=1, verbose=False)
    ps = model.prunable_parameters()
    for crit in (Magnitude(), Taylor1(), FisherTaylor()):
        s = crit.scores(ps)
        assert [a.shape for a in s] == [p.shape for p in ps]
        assert all((a >= 0).all() for a in s)


# --------------------------------------------------------------------- end to end
def test_end_to_end_self_pruning_hits_budget_and_stays_accurate() -> None:
    model, opt, (xt, yt, xv, yv) = tiny()
    sched = CubicSchedule(0.8, start_step=32, end_step=32 * 10, interval=16)
    pr = Pruner(model, opt, FisherTaylor(), sched, regrow_frac=0.1)
    train(model, opt, xt, yt, xv, yv, epochs=40, batch_size=32, callback=pr, verbose=False)
    assert pr.sparsity() == pytest.approx(0.8, abs=1e-4)
    assert evaluate(model, xv, yv)[1] > 0.85
    assert all(np.isfinite(p.data).all() for p in model.parameters())
