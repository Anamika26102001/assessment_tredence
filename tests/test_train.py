import numpy as np
import pytest

from nn import mlp
from optim import Adam
from train import evaluate, train
from train.data import split_and_normalize


def two_spirals(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.random(n)) * 3 * np.pi
    y = rng.integers(0, 2, n)
    r = t / (3 * np.pi)
    ang = t + y * np.pi
    x = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + rng.normal(0, 0.02, (n, 2))
    return x.astype(np.float32), y.astype(np.int64)


def run(seed: int):
    x, y = two_spirals(1200, 0)
    model = mlp([2, 64, 64, 2], np.random.default_rng(seed))
    opt = Adam(model.parameters(), lr=3e-3)
    hist = train(
        model, opt, x[:1000], y[:1000], x[1000:], y[1000:], epochs=40, seed=seed, verbose=False
    )
    return model, hist, x[1000:], y[1000:]


def test_mlp_learns_spirals_without_nans() -> None:
    model, hist, xv, yv = run(0)
    losses = [r["train_loss"] for r in hist.epochs]
    assert np.isfinite(losses).all() and losses[-1] < 0.5 * losses[0]
    assert evaluate(model, xv, yv)[1] > 0.9


def test_training_is_reproducible_for_a_fixed_seed() -> None:
    _, h1, _, _ = run(3)
    _, h2, _, _ = run(3)
    assert [r["train_loss"] for r in h1.epochs] == [r["train_loss"] for r in h2.epochs]


def test_nonfinite_loss_raises() -> None:
    x, y = two_spirals(64, 0)
    model = mlp([2, 8, 2], np.random.default_rng(0))
    model.layers[0].weight.data[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        train(model, Adam(model.parameters()), x, y, x, y, epochs=1, verbose=False)


def test_split_is_disjoint_fixed_and_normalised_on_train_only() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (300, 4)).astype(np.uint8)
    y = rng.integers(0, 10, 300)
    a = split_and_normalize(x, y, 150, 50, 100)
    b = split_and_normalize(x, y, 150, 50, 100)
    np.testing.assert_array_equal(a.x_train, b.x_train)
    assert (len(a.x_train), len(a.x_val), len(a.x_test)) == (150, 50, 100)
    assert abs(a.x_train.mean()) < 1e-5 and abs(a.x_train.std() - 1) < 1e-4
    # test rows are exactly the trailing rows of the raw data (standard MNIST test set)
    raw_tail = x[-100:].astype(np.float32) / 255.0
    assert np.corrcoef(raw_tail.ravel(), a.x_test.ravel())[0, 1] == pytest.approx(1.0)
