from __future__ import annotations

import numpy as np
import pytest

from engine import Tensor, no_grad, softmax_cross_entropy
from nn import mlp
from scripts.ablate_lib import average_ranks, jaccard, pair_concordance, spearman
from scripts.ablate_oracle import Oracle, band_around_threshold, locate


def test_average_ranks_ties_share_the_mean_rank() -> None:
    assert average_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_known_values() -> None:
    a = np.arange(20.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert np.isnan(spearman(a, np.ones(20)))
    x, y = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    assert spearman(x, y) == pytest.approx(0.8)  # 1 - 6*4/(5*24)


def test_pair_concordance_perfect_reversed_and_restricted() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert pair_concordance(d, d) == (1.0, 6)
    assert pair_concordance(-d, d) == (0.0, 6)
    only_01 = np.zeros((4, 4), dtype=bool)
    only_01[0, 1] = True
    assert pair_concordance(d, d, only_01) == (1.0, 1)
    assert pair_concordance(np.zeros(4), d)[0] == 0.5  # all score ties count half
    assert pair_concordance(d, np.ones(4))[1] == 0  # equal deltas carry no ordering


def test_jaccard() -> None:
    a = np.array([1, 1, 0, 0], dtype=bool)
    b = np.array([0, 1, 1, 0], dtype=bool)
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert np.isnan(jaccard(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)))


def test_oracle_delta_matches_a_naive_full_forward_with_the_weight_zeroed() -> None:
    rng = np.random.default_rng(0)
    model = mlp([6, 7, 5, 3], rng, dtype=np.float64)
    x, y = rng.standard_normal((40, 6)), rng.integers(0, 3, 40)
    oracle = Oracle(model, x, y)
    with no_grad():
        base = softmax_cross_entropy(model(Tensor(x)), y).item()
    assert oracle.base == pytest.approx(base)
    for layer, p in enumerate(model.prunable_parameters()):
        for idx in [(0, 0), (p.shape[0] - 1, p.shape[1] - 1), (1, 2)]:
            old = p.data[idx]
            p.data[idx] = 0.0
            with no_grad():
                naive = softmax_cross_entropy(model(Tensor(x)), y).item() - base
            p.data[idx] = old
            assert oracle.delta(layer, idx) == pytest.approx(naive, abs=1e-12)


def test_locate_and_band() -> None:
    assert locate(np.array([0, 5, 6, 8]), [6, 3]) == [(0, 0), (0, 5), (1, 0), (1, 2)]
    score = np.arange(100.0)[::-1]  # rank 0 = index 0
    band = band_around_threshold(score, n_keep=50, n=10)
    assert sorted(band.tolist()) == list(range(45, 55))


def test_unknown_optimizer_is_rejected() -> None:
    from prune.experiment import PruneConfig, run_pruning

    with pytest.raises(ValueError, match="unknown optimizer"):
        run_pruning(PruneConfig(optimizer="rmsprop"), None)  # type: ignore[arg-type]
