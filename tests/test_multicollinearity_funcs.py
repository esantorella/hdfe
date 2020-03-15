import numpy as np
from scipy import sparse as sps
from hdfe.multicollinearity import (
    remove_cols_from_csc,
    find_collinear_cols,
    remove_collinear_cols,
)


def test_remove_cols_from_csc() -> None:
    x = sps.eye(4, dtype=int).tocsc()
    cols_to_remove = [1, 2]
    result = remove_cols_from_csc(x, cols_to_remove)
    expected_result = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    np.testing.assert_equal(result.A, expected_result)


def test_find_collinear_cols() -> None:
    x = np.array([[1, 1], [0, 0]])
    collinear, not_collinear = find_collinear_cols(x)
    assert collinear == [1]
    assert not_collinear == [0]


def test_remove_collinear_cols() -> None:
    x = np.array([[1, 1], [0, 0]])
    res = remove_collinear_cols(x)
    expected = np.array([[1], [0]])
    np.testing.assert_equal(res, expected)
