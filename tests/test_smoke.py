import numpy as np


def test_numpy_available() -> None:
    assert np.zeros(3).sum() == 0.0
