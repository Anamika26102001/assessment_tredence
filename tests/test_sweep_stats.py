from __future__ import annotations

import numpy as np
import pytest

from prune.experiment import PruneConfig, run_pruning
from scripts.stats import (
    bootstrap_ci,
    claim_text,
    group,
    paired_rows,
    paired_summary,
    pareto_rows,
    signflip_pvalue,
)
from scripts.sweep import plan, run_key, variant_of
from train.data import Dataset


# ------------------------------------------------------------------------- plan
def test_plan_is_complete_and_keys_unique() -> None:
    cfgs = plan()
    keys = [run_key(c) for c in cfgs]
    assert len(keys) == len(set(keys)) == 8 * (1 + 5 * 2 + 3 * 3)
    from dataclasses import asdict

    variants = {variant_of(asdict(c)) for c in cfgs}
    assert variants == {
        "dense",
        "magnitude",
        "fisher_taylor",
        "taylor1",
        "fisher_taylor+regrow",
        "fisher_taylor+oneshot",
    }
    # paired design: every method at a given (sparsity, seed) shares the seed
    assert {c.seed for c in cfgs} == set(range(8))


# ------------------------------------------------------------------------ stats
def test_bootstrap_ci_degenerate_and_deterministic() -> None:
    assert bootstrap_ci(np.full(8, 0.01)) == pytest.approx((0.01, 0.01))
    d = np.array([-2.0, -1.0, 0.5, 1.0, 1.5, -0.5, 0.0, 2.0]) / 100
    assert bootstrap_ci(d) == bootstrap_ci(d)
    lo, hi = bootstrap_ci(d)
    assert lo < d.mean() < hi and lo < 0 < hi


def test_signflip_pvalue_exact_values() -> None:
    assert signflip_pvalue(np.ones(8)) == pytest.approx(2 / 256)
    assert signflip_pvalue(np.ones(5)) == pytest.approx(2 / 32)  # n=5 can never reach 0.05
    assert signflip_pvalue(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(1.0)


def test_paired_summary_fields() -> None:
    a = np.array([0.98, 0.97, 0.99, 0.98])
    b = np.array([0.97, 0.97, 0.98, 0.97])
    s = paired_summary(a, b)
    assert s["n"] == 4 and s["a_wins"] == 3
    assert s["mean_diff"] == pytest.approx(0.0075)
    assert s["ci_excludes_zero"] is True


def fake_runs(gap: float, n: int = 8, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    shared_by_seed = rng.normal(0, 0.003, n)  # seed-level effect common to all methods
    runs = []
    for variant, sparsity, acc0 in [
        ("dense", 0.0, 0.977),
        ("fisher_taylor", 0.95, 0.975 + gap),
        ("magnitude", 0.95, 0.975),
    ]:
        for s in range(n):
            shared = shared_by_seed[s]
            base = {"dense": 0.977, "fisher_taylor": 0.975 + gap, "magnitude": 0.975}[variant]
            runs.append(
                {
                    "config": {
                        "criterion": "magnitude" if variant == "dense" else variant,
                        "sparsity": sparsity,
                        "schedule": "cubic",
                        "regrow": 0.0,
                        "seed": s,
                    },
                    "test_acc": base + shared + rng.normal(0, 0.0003),
                    "val_acc": acc0,
                    "achieved_sparsity": sparsity,
                    "n_active": 1000,
                    "matmul_flops_active": 2000,
                }
            )
    return runs


def test_pairing_removes_shared_seed_noise_and_claim_reflects_evidence() -> None:
    g = group(fake_runs(gap=0.004))
    paired = paired_rows(g)
    row = next(r for r in paired if r["a"] == "fisher_taylor" and r["b"] == "magnitude")
    assert row["a_wins"] == 8 and row["ci_excludes_zero"]  # seed noise (0.003) cancels in pairs
    assert "unlikely to be seed noise" in claim_text(g, paired)

    gneg = group(fake_runs(gap=-0.004))  # fisher worse: the claim must say magnitude wins
    assert "magnitude is better" in claim_text(gneg, paired_rows(gneg))

    g0 = group(fake_runs(gap=0.0))
    assert "NOT distinguishable" in claim_text(g0, paired_rows(g0))


def test_pareto_rows_aggregate() -> None:
    rows = pareto_rows(group(fake_runs(0.0)))
    assert {r["method"] for r in rows} == {"dense", "fisher_taylor", "magnitude"}
    r = next(r for r in rows if r["method"] == "magnitude")
    assert r["n_seeds"] == 8 and r["test_acc_min"] <= r["test_acc_mean"] <= r["test_acc_max"]


# -------------------------------------------------------------- reproducibility
def blobs(n: int, seed: int, d: int = 10, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    centers = np.random.default_rng(99).standard_normal((k, d)) * 2
    return (centers[y] + rng.standard_normal((n, d))).astype(np.float32), y.astype(np.int64)


def tiny_data() -> Dataset:
    xt, yt = blobs(256, 0)
    xv, yv = blobs(64, 1)
    xs, ys = blobs(64, 2)
    return Dataset(xt, yt, xv, yv, xs, ys)


@pytest.mark.parametrize(
    "kw",
    [
        {"criterion": "magnitude"},
        {"criterion": "fisher_taylor"},
        {"criterion": "fisher_taylor", "regrow": 0.1},
        {"criterion": "taylor1"},
        {"criterion": "fisher_taylor", "schedule": "oneshot"},
        {"criterion": "magnitude", "sparsity": 0.0},
    ],
)
def test_same_config_gives_bit_identical_results(kw: dict) -> None:
    cfg = PruneConfig(
        sizes=[10, 24, 12, 3], epochs=5, warmup_epochs=1, ramp_end_epoch=3,
        interval=4, batch_size=32, **{"sparsity": 0.8, **kw},
    )  # fmt: skip
    a, b = run_pruning(cfg, tiny_data()), run_pruning(cfg, tiny_data())
    assert a["test_acc"] == b["test_acc"]
    assert [r["train_loss"] for r in a["history"]] == [r["train_loss"] for r in b["history"]]
    assert a["achieved_sparsity"] == pytest.approx(cfg.sparsity, abs=1e-3)
    assert a["n_active"] == b["n_active"]
