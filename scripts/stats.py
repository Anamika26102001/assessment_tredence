"""Aggregate sweep runs into the Pareto table, paired statistics and the committed claim.

Statistics are numpy-only and deliberately simple, because n is small (8 seeds):
  * paired design: for a given seed both methods share the same init and data order, so we
    compare per-seed differences, not two independent means;
  * 95% percentile-bootstrap CI of the mean paired difference (resampling seeds);
  * exact two-sided sign-flip permutation test (2^n sign patterns), which makes no
    normality assumption; with n=8 the smallest attainable p is 2/256 ~ 0.008;
  * paired t statistic and the count of seeds where A > B, for intuition.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.sweep import variant_of

ROOT = Path(__file__).resolve().parent.parent

COMPARISONS = [  # (A, B): positive difference means A is better
    ("fisher_taylor", "magnitude"),
    ("fisher_taylor+regrow", "fisher_taylor"),
    ("fisher_taylor", "taylor1"),
    ("taylor1", "magnitude"),
    ("fisher_taylor", "fisher_taylor+oneshot"),
]
CLAIM_SPARSITY = 0.95  # headline level; chosen in the plan, not committed before the sweep results


def load_runs(runs_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))]


def bootstrap_ci(
    diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def signflip_pvalue(diffs: np.ndarray) -> float:
    """Exact two-sided p: under H0 (no effect) each paired difference is symmetric about 0."""
    n = len(diffs)
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((null >= abs(diffs.mean()) - 1e-15).mean())


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    lo, hi = bootstrap_ci(d)
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "std_diff": sd,
        "ci_lo": lo,
        "ci_hi": hi,
        "t_stat": float(d.mean() / (sd / np.sqrt(n))) if sd and sd > 0 else float("nan"),
        "p_signflip": signflip_pvalue(d),
        "a_wins": int((d > 0).sum()),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def group(runs: list[dict]) -> dict[tuple[str, float], dict[int, dict]]:
    g: dict[tuple[str, float], dict[int, dict]] = defaultdict(dict)
    for r in runs:
        c = r["config"]
        g[(variant_of(c), c["sparsity"])][c["seed"]] = r
    return g


def pareto_rows(g: dict) -> list[dict]:
    rows = []
    for (variant, s), by_seed in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        acc = np.array([r["test_acc"] for r in by_seed.values()])
        rows.append(
            {
                "method": variant,
                "target_sparsity": s,
                "achieved_sparsity": float(
                    np.mean([r["achieved_sparsity"] for r in by_seed.values()])
                ),
                "n_seeds": len(acc),
                "test_acc_mean": float(acc.mean()),
                "test_acc_std": float(acc.std(ddof=1)) if len(acc) > 1 else float("nan"),
                "test_acc_min": float(acc.min()),
                "test_acc_max": float(acc.max()),
                "val_acc_mean": float(np.mean([r["val_acc"] for r in by_seed.values()])),
                "active_weights": int(np.mean([r["n_active"] for r in by_seed.values()])),
                "matmul_flops_per_sample": int(
                    np.mean([r["matmul_flops_active"] for r in by_seed.values()])
                ),
            }
        )
    return rows


def paired_rows(g: dict) -> list[dict]:
    rows = []
    for a_name, b_name in COMPARISONS:
        for s in sorted({k[1] for k in g if k[0] == a_name}):
            ga, gb = g.get((a_name, s), {}), g.get((b_name, s), {})
            seeds = sorted(set(ga) & set(gb))
            if len(seeds) < 2:
                continue
            a = np.array([ga[k]["test_acc"] for k in seeds])
            b = np.array([gb[k]["test_acc"] for k in seeds])
            rows.append({"a": a_name, "b": b_name, "sparsity": s, **paired_summary(a, b)})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _verdict(p: dict) -> str:
    if not p["ci_excludes_zero"]:
        return "The 95% bootstrap CI includes 0, so this gap is NOT distinguishable from seed noise at this n."
    winner = p["a"] if p["mean_diff"] > 0 else p["b"]
    return (
        f"The 95% bootstrap CI excludes 0, so the gap is unlikely to be seed noise: {winner} is better "
        f"by about {abs(p['mean_diff']) * 10_000:.0f} of 10,000 test images."
    )


def claim_text(g: dict, paired: list[dict], s: float = CLAIM_SPARSITY) -> str:
    f, m, d = g.get(("fisher_taylor", s)), g.get(("magnitude", s)), g.get(("dense", 0.0))
    p = next(
        (
            r
            for r in paired
            if r["a"] == "fisher_taylor" and r["b"] == "magnitude" and r["sparsity"] == s
        ),
        None,
    )
    if not (f and m and d and p):
        return "(claim unavailable: missing runs)"
    fa, ma, da = (np.array([r["test_acc"] for r in x.values()]) * 100 for x in (f, m, d))
    n = p["n"]
    magnitude_wins = p["ci_excludes_zero"] and p["mean_diff"] < 0
    first, second = (
        ("magnitude", "fisher_taylor") if magnitude_wins else ("fisher_taylor", "magnitude")
    )
    seeds = sorted(set(f) & set(m))
    x = np.array([g[(first, s)][k]["test_acc"] for k in seeds])
    y = np.array([g[(second, s)][k]["test_acc"] for k in seeds])
    q = {**paired_summary(x, y), "a": first, "b": second}
    xa, ya = (ma, fa) if magnitude_wins else (fa, ma)
    return (
        f"At {s:.0%} sparsity (achieved {np.mean([r['achieved_sparsity'] for r in f.values()]):.2%}), "
        f"{first} pruning reaches {xa.mean():.2f}% +/- {xa.std(ddof=1):.2f} test accuracy versus "
        f"{ya.mean():.2f}% +/- {ya.std(ddof=1):.2f} for {second} pruning (mean +/- std over {n} seeds; "
        f"dense reference {da.mean():.2f}% +/- {da.std(ddof=1):.2f}). Paired difference "
        f"({first} - {second}) {q['mean_diff'] * 100:+.2f} pp, 95% bootstrap CI "
        f"[{q['ci_lo'] * 100:+.2f}, {q['ci_hi'] * 100:+.2f}], exact sign-flip p = {q['p_signflip']:.4f}, "
        f"{first} better in {q['a_wins']}/{n} seeds. {_verdict(q)}"
    )


PROVENANCE = (
    "Provenance: the 95% level was named in the project plan as the claim level, but the repository cannot "
    "show that it was fixed before the sweep results existed (`CLAIM_SPARSITY` first appears in the same "
    "commit as the raw runs). It is therefore **not** described as pre-registered. The sign and size of the gap "
    "at 90% and 98% are the same (see the paired table), so the choice of level does not drive the conclusion."
)


def ablation_claims(paired: list[dict], s: float = CLAIM_SPARSITY) -> list[str]:
    out = []
    for r in paired:
        if r["sparsity"] != s or (r["a"], r["b"]) == ("fisher_taylor", "magnitude"):
            continue
        out.append(
            f"At {s:.0%}: {r['a']} - {r['b']} = {r['mean_diff'] * 100:+.2f} pp "
            f"[{r['ci_lo'] * 100:+.2f}, {r['ci_hi'] * 100:+.2f}], {r['a']} better in {r['a_wins']}/{r['n']} seeds, "
            f"p = {r['p_signflip']:.4f}."
        )
    return out


def observation_text(g: dict, sparsities: tuple[float, ...] = (0.9, 0.95, 0.98)) -> str:
    dense = g.get(("dense", 0.0))
    if not dense:
        return ""
    rows = [
        "| method | sparsity | pruned - dense (pp) | 95% CI | p (sign-flip) | pruned better in |",
        "|---|---|---|---|---|---|",
    ]
    for method in ("magnitude", "fisher_taylor"):
        for s in sparsities:
            run = g.get((method, s))
            if not run:
                continue
            seeds = sorted(set(run) & set(dense))
            p = paired_summary(
                np.array([run[k]["test_acc"] for k in seeds]),
                np.array([dense[k]["test_acc"] for k in seeds]),
            )
            rows.append(
                f"| {method} | {s:.0%} | {p['mean_diff'] * 100:+.2f} | [{p['ci_lo'] * 100:+.2f}, {p['ci_hi'] * 100:+.2f}] | "
                f"{p['p_signflip']:.4f} | {p['a_wins']}/{p['n']} |"
            )
    return "\n".join(rows)


LIMITATIONS = """\
- **MNIST ceiling.** All methods land within a few tenths of a point of each other (97.7-98.3%), so the
  criteria are being separated by tens of test images out of 10,000. A harder task could separate them more
  or reverse the ordering; this one cannot say.
- **Unregularised dense baseline.** The dense model reaches 99.4% train accuracy with no regularisation. Pruned
  models beat it under Adam and lose to it under SGD, and the margin shrinks with the best-validation checkpoint.
  Read "pruned beats dense" as a statement about this baseline, not about pruning.
- **Single machine.** Wall-clock numbers come from one Apple-silicon laptop with a single BLAS thread. They show
  direction and rough size, not what a server CPU or GPU would do.
- **NumPy CSR is slower than BLAS.** The sparse-aware forward written here never beats dense BLAS. Counted FLOPs and
  active parameters fall as expected; measured latency improves only through structural compaction, and only for
  some models (see the cost table).
- **No multiple-comparison correction.** Only the 95% comparison is the headline; the other sparsity levels and
  all ablation tests are exploratory, uncorrected for the number of comparisons, and an isolated p-value near 0.05
  should not be trusted.
- **Ablations explain, they do not prove.** The oracle test uses one seed and one dense checkpoint; the SGD
  learning-rate grid is coarse; the beta sweep reports validation accuracy only.
- **fisher_taylor was changed after the first sweep** (regrowth bug and EMA bias fixed). Runs without regrowth
  reproduced bit-for-bit, so the headline numbers are unchanged; only the regrow runs were re-run.
- Eight seeds, one architecture (784-300-100-10), one dataset.
"""


def results_md(
    rows: list[dict],
    paired: list[dict],
    claim: str,
    sections: dict[str, str] | None = None,
    observation: str = "",
) -> str:
    sec = sections or {}
    out = ["# Results (generated by `scripts/stats.py` from the committed runs and ablations)\n"]
    out += ["## Headline claim\n", claim + "\n", PROVENANCE + "\n"]
    out += (
        ["## Secondary results (same sparsity)\n"]
        + [f"- {c}" for c in ablation_claims(paired)]
        + [
            "",
            "Reading: gradient information helps only when it is the smoothed second-order term "
            "(fisher_taylor beats the single-batch first-order score and one-shot pruning), and "
            "regrowth shows no detectable effect. None of these beats plain magnitude pruning.",
            "",
        ]
    )
    if observation:
        out += [
            "## Observation: pruned vs dense (caveated, not a headline)\n",
            observation,
            "\nThis margin depends on the baseline (an unregularised model that overfits): it shrinks with "
            "the best-validation checkpoint and reverses under SGD (see the ablations and Limitations).\n",
        ]
    ablations = [
        sec[k] for k in ("oracle", "rank_corr", "overlap", "sgd", "beta", "dense") if k in sec
    ]
    if ablations:
        out += [
            "## Explanatory ablations\n",
            "These test why magnitude wins. They do not change the headline, and nothing in them was used to "
            "choose a setting on test accuracy.\n",
            *ablations,
        ]
    if "cost" in sec:
        out += ["## Cost evidence\n", sec["cost"]]
    out += ["## Accuracy vs sparsity (test accuracy %, mean +/- std over seeds)\n"]
    out += [
        "| method | target | achieved | seeds | test acc | min | max | active weights |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['method']} | {r['target_sparsity']:.0%} | {r['achieved_sparsity']:.2%} | {r['n_seeds']} | "
            f"{r['test_acc_mean'] * 100:.2f} +/- {r['test_acc_std'] * 100:.2f} | {r['test_acc_min'] * 100:.2f} | "
            f"{r['test_acc_max'] * 100:.2f} | {r['active_weights']:,} |"
        )
    out += [
        "\n## Paired comparisons (A - B, percentage points; same seed = same init and data order)\n"
    ]
    out += [
        "| A | B | sparsity | mean diff | 95% CI | p (sign-flip) | A wins | CI excludes 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in paired:
        out.append(
            f"| {r['a']} | {r['b']} | {r['sparsity']:.0%} | {r['mean_diff'] * 100:+.2f} | "
            f"[{r['ci_lo'] * 100:+.2f}, {r['ci_hi'] * 100:+.2f}] | {r['p_signflip']:.4f} | "
            f"{r['a_wins']}/{r['n']} | {'yes' if r['ci_excludes_zero'] else 'no'} |"
        )
    out += ["\n## Limitations\n", LIMITATIONS]
    return "\n".join(out) + "\n"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    g = group(load_runs(args.runs))
    rows, paired = pareto_rows(g), paired_rows(g)
    claim = claim_text(g, paired)
    write_csv(args.out / "pareto.csv", rows)
    write_csv(args.out / "paired_stats.csv", paired)
    from scripts.report import sections

    md = results_md(rows, paired, claim, sections(), observation_text(g))
    (args.out / "RESULTS.md").write_text(md)
    print(claim)


if __name__ == "__main__":
    main()
