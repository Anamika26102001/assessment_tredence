# Single-threaded BLAS: deterministic reductions and fair wall-clock comparisons.
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

RUN := uv run

.PHONY: install lint typecheck test gradcheck train prune sweep sweep-quick bench bench-variance ablate demo check

install:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

gradcheck:
	$(RUN) pytest tests/test_gradcheck.py tests/test_broadcast.py tests/test_masked.py

check: lint typecheck test

train:
	$(RUN) python -m scripts.train_dense

prune:
	$(RUN) python -m scripts.train_prune --criterion fisher_taylor --sparsity 0.9 --seed 0

bench:
	$(RUN) python -m scripts.bench_inference

sweep:
	$(RUN) python -m scripts.sweep --jobs 4
	$(RUN) python -m scripts.stats
	$(RUN) python -m scripts.make_plots

sweep-quick:
	$(RUN) python -m scripts.sweep --quick --out results/quick_runs

bench-variance:
	$(RUN) python -m scripts.bench_variance

ablate:
	$(RUN) python -m scripts.ablate_oracle
	$(RUN) python -m scripts.ablate_rankcorr
	$(RUN) python -m scripts.ablate_overlap
	$(RUN) python -m scripts.ablate_sgd
	$(RUN) python -m scripts.ablate_beta
	$(RUN) python -m scripts.ablate_dense_baseline
	$(RUN) python -m scripts.stats

demo:
	$(RUN) python -m scripts.export_demo
	@echo "open demo/index.html in a browser"
