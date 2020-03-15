import pandas as pd
import numpy as np
from hdfe.hdfe import make_dummies, get_all_dummies


def test_make_dummies_arr() -> None:
    x = np.array([1, 0, 0])
    results = make_dummies(x, False)
    expected = np.array([[0, 1], [1, 0], [1, 0]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_make_dummies_ser() -> None:
    x = pd.Series([1, 0, 0])
    results = make_dummies(x, False)
    expected = np.array([[0, 1], [1, 0], [1, 0]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_make_dummies_cat() -> None:
    x = pd.Series(["horse", "cat", "cat"]).astype("category")
    results = make_dummies(x, False)
    expected = np.array([[0, 1], [1, 0], [1, 0]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_make_dummies_arr_drop() -> None:
    x = np.array([1, 0, 0])
    results = make_dummies(x, True)
    expected = np.array([[0], [1], [1]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_make_dummies_ser_drop() -> None:
    x = pd.Series([1, 0, 0])
    results = make_dummies(x, True)
    expected = np.array([[0], [1], [1]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_make_dummies_cat_drop() -> None:
    x = pd.Series(["horse", "cat", "cat"]).astype("category")
    results = make_dummies(x, True)
    expected = np.array([[0], [1], [1]], dtype=float)
    np.testing.assert_almost_equal(results.A, expected)


def test_get_all_dummies() -> None:
    x = np.array([[0, 0], [1, 0], [0, 1]])
    result = get_all_dummies(x)
    expected = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=float)
    np.testing.assert_almost_equal(result.A, expected)
