<!-- DRAFT: first pass written for the author to rewrite in their own words and defend. -->

# Design notes

Four questions from the brief, answered against what this repository actually measured.

## 1. The importance criterion

**Goal.** A score `s_i` that approximates the loss increase from removing connection `i`, i.e. moving
`w_i` by `delta = -w_i`. Expanding the loss to second order around the current weights:

    L(w + delta) - L(w) = g^T delta + 1/2 delta^T H delta + O(|delta|^3)

Only coordinate `i` moves, so off-diagonal Hessian terms vanish and

    Delta L_i = -g_i w_i + 1/2 H_ii w_i^2.

**Three truncations.**

- `magnitude`, `|w_i|`: drops both terms. It ranks correctly only if every weight has the same curvature
  and gradient scale, which is false across layers and fan-ins.
- `taylor1`, `|g_i w_i|`: keeps the first-order term. Near a minimum `E[g] ~ 0`, so on one mini-batch this is
  mostly gradient noise, and the sign of `-g_i w_i` is not guaranteed positive.
- `fisher_taylor` (primary), `s_i = w_i^2 * EMA[g_i^2]`: near a minimum `g ~ 0`, so the second-order term is
  the one that survives. The Hessian diagonal is replaced by the diagonal of the Fisher information,
  `H_ii ~ F_ii = E_x[g_i(x)^2]` (equal to the Hessian in expectation when the model is well specified at a
  minimum). That gives `Delta L_i ~ 1/2 F_ii w_i^2`, which is Optimal Brain Damage with the Fisher diagonal.
  The constant `1/2` does not change a ranking.

**What the estimate really is.** `g` is the batch-mean gradient, so `EMA[g^2]` is the diagonal of the
*empirical* Fisher up to a `1/B` factor (`B` = batch size), plus an `(E g)^2` term that is small near
convergence. `1/B` is common to all entries (they see the same batches), so rankings are unaffected. The EMA is
bias-corrected per entry (`/(1 - beta^n)`, with `n` reset on revival), so a regrown weight is scored on the same
scale as an established one. Scores are in units of loss, which is what justifies one *global* ranking instead of
hand-set per-layer sparsities; a per-layer floor (1%) prevents layer collapse.

**Did it work? Not against magnitude.** At 95% sparsity magnitude pruning is better: 98.15% vs 97.97% test accuracy
over 8 seeds, paired difference +0.18 pp, 95% bootstrap CI [+0.07, +0.29], magnitude better in 6 of 8 seeds
(`results/RESULTS.md`). The derivation holds up in the places it can be checked directly, and still the
end-to-end result goes the other way:

- Oracle test (`results/ablations/oracle.json`): zeroing single weights on the dense model and measuring the exact
  validation-loss change, the Fisher score tracks removal cost better than `|w|` (Spearman 0.80 vs 0.64 against
  `|dL|` on random weights) and separates weights that `|w|` cannot (0.50 vs -0.02 inside the `|w|` boundary band).
  So "the Fisher score is a worse single-weight predictor" is **not** the explanation.
- The gap survives SGD+momentum (-0.19 pp, 8/8 seeds), so it is not an Adam artefact, and every EMA window from 10
  to 10,000 steps (validation only), so it is not the window length.
- The two criteria keep very different sets (Jaccard 0.285 overall) yet end within about 0.2 pp.

What these results do and do not say: single-weight removal cost on a converged dense model is not the same
problem as choosing 5% of weights to keep while the network keeps training between cuts. The per-weight additivity
assumption in the Taylor expansion breaks down when most weights go at once, and the network recovers between
events. I did not test that explanation, so it is a hypothesis. The gradient information does help in one place:
`fisher_taylor` beats `taylor1` (+0.29 pp, 8/8 seeds) and one-shot pruning (+0.56 pp, 8/8 seeds) at 95%.

## 2. The gradient of a masked weight

The forward pass uses `W_eff = W (.) M` with `M` a constant 0/1 array. Then

    dL/dW_eff = x^T G                 (dense: nonzero everywhere)
    dL/dW     = dL/dW_eff (.) M       (the multiplication's backward pushes g * M)

so for a masked entry `dL/dW_ij = 0` **exactly**. This is not a convention; it is the true derivative of
`L(W (.) M)` with respect to `W_ij`, because perturbing a masked weight cannot change the forward pass. Finite
differences return 0 there too, which is why the gradient check agrees with no special case
(`tests/test_masked.py`).

The unmasked `dL/dW_eff` is still computed, kept on the `W_eff` node, and used for one thing only: regrowth, which
asks "how much would the loss drop if this connection existed?". It never reaches an optimizer. If regrowth read
`dL/dW` instead, every dead weight would score exactly 0 and regrowth could never fire.

**Optimizer state.** With a zero gradient, plain Adam would still move a pruned weight (its first moment `m` is a
decaying average of old gradients), and a revived weight would inherit stale `m`/`v` plus a large global step count
(first step about `3.16 * lr` instead of `lr`). So Adam here keeps a per-entry step count `t`; masked entries are
frozen (weight, `m`, `v`, `t`); and a connection that is pruned *or* revived has its weight, `m`, `v` and `t` zeroed
(`Parameter.set_mask` -> `optimizer.reset_entries`). A revived weight starts at exactly 0 and its first step is
`-lr * sign(g)`. Tests: masked entries stay bit-exactly zero with zero state; the revival step; a deliberately naive
Adam that fails both; and a masked network that trains identically to a physically smaller one. SGD+momentum gets
the same treatment.

One bug found late and fixed: regrowth could re-select weights pruned in the *same* event (41-46% of regrown weights
in early events at 95%), undoing the prune and resetting their values. They are now excluded; only the 24 regrowth
runs were affected and were re-run (no detectable benefit from regrowth before or after: +0.05 pp at 95%, CI
[-0.03, +0.11]).

## 3. Where the engine bottlenecks

Profile of 100 training steps (batch 128, 784-300-100-10, single BLAS thread, a random 95% mask installed so the
masked code paths run; `cProfile`, 2.3 ms per step):

| share of step time | where |
|---|---|
| 61% | `Adam.step` masked branch |
| 8% | `FisherTaylor.update` (EMA of `g^2`) |
| about 7% | forward and backward matmuls and their closures |
| rest | `astype` copies, reductions, loop overhead |

The matrix multiplies are not the cost; the optimizer is. The masked Adam builds several full-size temporaries per
parameter every step (`np.where` over `m`, `v`, and the update, plus an int64 step-count array), touching all 266k
entries even though only 5% are alive.

**Fixes, in order of expected payoff** (not implemented; ordered by the profile):

1. Keep optimizer state only for alive entries (flat arrays indexed by the alive set, rebuilt at each pruning event,
   which happens every 100 steps). Step cost then scales with the number of alive weights, not the dense size.
2. Update in place with `out=` arguments instead of allocating temporaries; fuse the `m`, `v` and weight updates into
   one pass.
3. Drop the per-step forward multiply `W (.) M`: dead entries of `W` are already exactly zero, so the forward can
   use `W` directly and the mask can be applied once to the gradient. The dense counterfactual gradient that
   regrowth needs would then be computed only at pruning events, not every step.
4. Fuse `Linear + bias + ReLU` into one node: fewer graph nodes, fewer retained intermediates. Keep float32.
5. Beyond NumPy: the per-op Python closure and node allocation overhead matters once the matrices are small. Fixed by
   tracing the graph once (static graph) and compiling it, or moving the hot loop to a compiled backend.

**Inference.** The sparse forward written here gathers `x[:, indices] * data` and reduces with `np.add.reduceat`.
Measured over 8 seeds' models it is **slower than dense BLAS at every sparsity and batch size** (median speedup
0.24-0.49x at batch 1, 0.013-0.066x at batch 256). BLAS runs at hardware speed on dense data; a Python-level gather
does not. A real sparse speedup needs a compiled kernel or structured sparsity.

## 4. Serving a self-pruned model in a multi-tenant service

This section is a design, not something that was built or load-tested here. What the measurements do support:

- **Counted work is not latency.** FLOPs and parameters fall by the sparsity factor (19.7x at 95%), but measured
  latency only improves through structural compaction (removing disconnected neurons and inputs, then plain dense
  matrices), and only for some models: at 95%, 1.37x (batch 1) and 1.07x (batch 256) for `fisher_taylor` models, but
  0.91x and 0.66x for magnitude models, which leave few neurons fully disconnected. So the artifact to ship, and the
  criterion that produced it, must be chosen by **measured latency on the target hardware**, not by sparsity.
- **Therefore: benchmark every candidate before it ships.** The release gate is accuracy on a held-out set plus
  p50/p99 latency at the production batch sizes against the dense model.

**Proposed architecture.**

1. *Export a frozen artifact.* No autodiff at inference. Compact the pruned model (exact: the function does not
   change), then store dense matrices (or CSR for kernels that actually win), versioned and content-hashed, with the
   accuracy and latency numbers from the gate attached.
2. *Dynamic batching.* Thousands of requests per second means grouping requests per model into batches within a small
   latency budget; the batch size, not sparsity, dominates throughput on CPU. Batch 1 and batch 256 favour different
   layouts, so benchmark both.
3. *Multi-tenancy.* Share the base weights across tenants and keep per-tenant masks or small deltas, rather than one
   full copy per tenant; this matters because memory, not compute, usually limits how many models stay hot. Keep hot
   models resident, load cold ones on demand, and cap per-tenant concurrency so one tenant cannot starve others.
4. *Hardware.* On CPU, compaction plus dense BLAS is the safe path. Unstructured sparsity needs a compiled kernel.
   On GPUs, unstructured sparsity does not help; structured patterns (2:4 on supporting tensor cores, or block
   sparsity) do, which means pruning with that structure in the loop, not applying it afterwards.
5. *Rollout and monitoring.* Canary a new version on a slice of traffic, compare accuracy proxies and latency
   against the incumbent, roll back automatically on regression. Track input drift and per-tenant accuracy where
   labels arrive; a pruned model has less slack to absorb drift.
6. *Determinism and reproducibility.* Pin the artifact version and the numerics (dtype, BLAS thread count) so a
   result can be reproduced; the same config here gives bit-identical numbers.

**Honest limits of the evidence.** One architecture, one dataset (MNIST), one Apple-silicon laptop with a single BLAS
thread. The latency conclusions show direction and rough size, not what a server CPU or GPU would do.
