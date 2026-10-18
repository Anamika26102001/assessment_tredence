# Single-threaded BLAS: deterministic reductions and fair wall-clock comparisons.
export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

RUN := uv run

.PHONY: install lint typecheck test gradcheck train prune sweep sweep-quick bench stats plots check

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
