"""Markdown sections of results/RESULTS.md that come from the ablation and benchmark JSONs.

Everything printed here is read from committed files in results/; nothing is re-computed
from training."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "results" / "ablations"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _pp(x: float) -> str:
    return f"{x * 100:+.2f}"


def _ci(p: dict) -> str:
    return f"[{_pp(p['ci_lo'])}, {_pp(p['ci_hi'])}]"


def oracle_md(d: dict) -> str:
    names = {"random": "500 random weights", "boundary_m": "500 nearest the 95% threshold under abs(w)",
             "boundary_f": "500 nearest the 95% threshold under the Fisher score"}  # fmt: skip
    crit = {"magnitude": "abs(w)", "fisher": "w^2 EMA[g^2]", "taylor1": "abs(w g)"}
    out = [
        "### Oracle: do the scores predict the true cost of removing a weight?\n",
        f"Dense seed-0 model (test acc {d['dense_test_acc']:.4f}). Each weight is zeroed on its own and the "
        "exact validation-loss change over the full 5k validation set is measured (float64). "
        "Columns: Spearman rho against abs(dL) and against the signed dL; share of weight pairs ordered "
        "correctly by abs(dL) (all pairs / pairs on opposite sides of that criterion's own 95% threshold).\n",
        "| weight set | score | rho vs abs(dL) | rho vs signed dL | pairs ok (all) | pairs ok (cross-threshold) |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in names.items():
        for c, cl in crit.items():
            e = d["sets"][key]["criteria"][c]
            out.append(
                f"| {label} | {cl} | {e['abs']['spearman']:+.3f} | {e['signed']['spearman']:+.3f} | "
                f"{e['abs']['pair_concordance_all']:.3f} | {e['abs']['pair_concordance_cross_threshold']:.3f} |"
            )
    s = d["sets"]
    neg = s["random"]["n_negative_delta"]
    out.append(
        f"\nReading: {neg} of 500 random weights *lower* validation loss when removed, so the signed change "
        "is mostly noise and no score predicts it. Against its magnitude the Fisher score tracks removal cost "
        f"better than abs(w) on random weights (rho {s['random']['criteria']['fisher']['abs']['spearman']:.2f} vs "
        f"{s['random']['criteria']['magnitude']['abs']['spearman']:.2f}). In each boundary band the selecting "
        "score has almost no spread by construction, so its own rho there is uninformative; the other score is "
        "the fairer comparison. A better single-weight predictor on the dense model is therefore not what "
        "decides the end-to-end result.\n"
    )
    return "\n".join(out)


def rank_corr_md(d: dict) -> str:
    out = [
        "### Rank agreement between the scores on the dense model\n",
        "| pair | metric | layer 1 | layer 2 | layer 3 | all |",
        "|---|---|---|---|---|---|",
    ]
    for key in d["spearman"]:
        for metric, label in (
            ("spearman", "Spearman rho"),
            ("top5pct_jaccard", "Jaccard of top-5% sets"),
        ):
            v = d[metric][key]
            out.append(
                f"| {key.replace('_vs_', ' vs ')} | {label} | "
                + " | ".join(f"{v[n]:.3f}" for n in ("layer1", "layer2", "layer3", "all"))
                + " |"
            )
    return "\n".join(out) + "\n"


def overlap_md(d: dict) -> str:
    sm = d["summary"]
    row = " | ".join(
        f"{sm[k]['mean']:.3f} +/- {sm[k]['std']:.3f}" for k in ("layer1", "layer2", "layer3", "all")
    )
    return (
        "### Kept-set overlap after pruning to 95% (magnitude vs fisher_taylor, seeds 0-7)\n\n"
        "| quantity | layer 1 | layer 2 | layer 3 | all |\n|---|---|---|---|---|\n"
        f"| Jaccard of surviving weights | {row} |\n\n"
        "The two methods end with largely different sparse networks that reach accuracies within a few "
        "tenths of a point.\n"
    )


def sgd_md(d: dict) -> str:
    sm, p = d["summary"], d["paired"]["fisher_taylor_minus_magnitude"]
    lr_rows = ", ".join(f"{lr}: {v['mean_val_acc'] * 100:.2f}" for lr, v in d["lr_search"].items())
    return (
        "### SGD + momentum instead of Adam (95% sparsity, 8 seeds)\n\n"
        f"Learning rate chosen on dense validation accuracy only (seeds 100-102; {lr_rows}) -> "
        f"**{d['selected_lr']}**, then held fixed.\n\n"
        "| method | test acc (%) | val acc (%) |\n|---|---|---|\n"
        + "\n".join(
            f"| {k} | {v['test_mean'] * 100:.2f} +/- {v['test_std'] * 100:.2f} | "
            f"{v['val_mean'] * 100:.2f} +/- {v['val_std'] * 100:.2f} |"
            for k, v in sm.items()
        )
        + f"\n\nfisher_taylor - magnitude: {_pp(p['mean_diff'])} pp, 95% CI {_ci(p)}, sign-flip p = "
        f"{p['p_signflip']:.4f}, fisher_taylor better in {p['a_wins']}/8 seeds. The gap does not depend on Adam. "
        "Under SGD the dense model beats the pruned ones, unlike under Adam.\n"
    )


def beta_md(d: dict) -> str:
    out = [
        "### EMA decay of the Fisher score (fisher_taylor, 95%, validation accuracy only)\n",
        "| beta | window (steps) | val acc (%) | minus magnitude (pp) | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for b, v in d["betas"].items():
        p = v["minus_magnitude"]
        out.append(
            f"| {b} | {v['window_steps']:,} | {v['val_mean'] * 100:.2f} +/- {v['val_std'] * 100:.2f} | "
            f"{_pp(p['mean_diff'])} | {_ci(p)} |"
        )
    out.append(
        f"\nMagnitude validation accuracy (committed runs): {d['magnitude_val_mean'] * 100:.2f}. "
        "Test accuracy was not recorded for this sweep.\n"
    )
    return "\n".join(out)


def dense_md(d: dict) -> str:
    sm = d["summary"]
    rows = []
    for which, label in (("final", "final epoch (12)"), ("best_val", "best-validation epoch")):
        rows.append(
            f"| dense, {label} | {sm[f'{which}_val']['mean'] * 100:.2f} +/- {sm[f'{which}_val']['std'] * 100:.2f} | "
            f"{sm[f'{which}_test']['mean'] * 100:.2f} +/- {sm[f'{which}_test']['std'] * 100:.2f} |"
        )
    out = [
        "### Dense baseline: final epoch vs best-validation epoch (8 seeds)\n",
        "| checkpoint | val acc (%) | test acc (%) |\n|---|---|---|",
        *rows,
        f"\nBest-validation epochs by seed: {sm['best_val_epochs']}. Pruned (95%) minus dense, paired:\n",
        "| method | vs dense final | vs dense best-val epoch |\n|---|---|---|",
    ]
    for m, v in d["pruned_95_vs_dense"].items():
        a, b = v["minus_dense_final"], v["minus_dense_best_val"]
        out.append(
            f"| {m} | {_pp(a['mean_diff'])} pp {_ci(a)} (p={a['p_signflip']:.4f}) | "
            f"{_pp(b['mean_diff'])} pp {_ci(b)} (p={b['p_signflip']:.4f}) |"
        )
    return "\n".join(out) + "\n"


def cost_md(d: dict) -> str:
    env = d["env"]
    out = [
        "### Cost: counted work and measured wall-clock (8 seeds' models per row)\n",
        f"Machine: {env['cpu']} ({env['cpu_count']} cores), NumPy {env['numpy']}, BLAS {env['blas']}, "
        f"threads {env['blas_threads_env']['VECLIB_MAXIMUM_THREADS']}. Speedup = median(dense x zeros) / "
        "median(path) per model; cells show the median over 8 models with the inter-quartile range. "
        "A value below 1 means slower than dense.\n",
        "| method | sparsity | FLOPs / sample, CSR | FLOPs / sample, compacted | CSR b=1 | CSR b=256 | compacted b=1 | compacted b=256 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in d["summary"]:

        def cell(key: str, e: dict = e) -> str:
            v = e[key]
            nd = 3 if v["median"] < 0.1 else 2
            return f"{v['median']:.{nd}f}x [{v['q25']:.{nd}f}, {v['q75']:.{nd}f}]"

        out.append(
            f"| {e['method']} | {e['sparsity']:.0%} | {e['flops_reduction_csr']:.1f}x fewer | "
            f"{e['flops_reduction_compact']:.1f}x fewer | {cell('speedup_b1_csr')} | "
            f"{cell('speedup_b256_csr')} | {cell('speedup_b1_compact')} | {cell('speedup_b256_compact')} |"
        )
    out.append(
        "\nCounted FLOPs and active parameters fall by the sparsity factor, but wall-clock does not follow. "
        "The CSR forward written here is slower than dense BLAS at every sparsity and batch size. Compacting away "
        "disconnected neurons and inputs helps only when pruning leaves many of them disconnected, which depends "
        "on the criterion: it speeds up fisher_taylor models from 95%, and does little or nothing (or is slower) "
        'for magnitude models. "Dense times zeros" is the reference path and is not a saving.\n'
    )
    return "\n".join(out)


def sections() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, fn, path in (
        ("oracle", oracle_md, ABL / "oracle.json"),
        ("rank_corr", rank_corr_md, ABL / "rank_corr.json"),
        ("overlap", overlap_md, ABL / "overlap.json"),
        ("sgd", sgd_md, ABL / "sgd.json"),
        ("beta", beta_md, ABL / "beta.json"),
        ("dense", dense_md, ABL / "dense_baseline.json"),
        ("cost", cost_md, ROOT / "results" / "bench_variance.json"),
    ):
        data = _load(path)
        if data is not None:
            out[key] = fn(data)
    return out
