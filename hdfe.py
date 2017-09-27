import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg
import scipy.linalg
from multicollinearity import remove_collinear_cols
from itertools import chain
import pandas as pd
from groupby import Groupby


def make_dummies(elt, drop_col):
    _, elt = np.unique(elt, return_inverse=True)

    dummies = sps.csc_matrix((np.ones(len(elt)), (range(len(elt)), elt)))
    if drop_col:
        return dummies[:, :-1]
    else:
        return dummies


def get_all_dummies(categorical_data):
    if len(categorical_data.shape) == 1 or categorical_data.shape[1] == 1:
        return make_dummies(categorical_data, False)

    num_fes = categorical_data.shape[1]
    first = make_dummies(categorical_data[:, 0], False)
    others = [make_dummies(categorical_data[:, col], True)
              for col in range(1, num_fes)]
    others = sps.hstack(others)
    return sps.hstack((first, others))


# Automatically picks best method, takes pandas df
# TODO: return variance estimate if desired
# TODO: verbose option
# TODO: replace data with categorical data, and get rid of 'cat' argument
def estimate(data, y: np.ndarray, x, categorical_controls: list, check_rank=False,
             estimate_variance=False, get_residual=False, cluster=None):
    """

    :param data: Pandas DataFrame
    :param y: 2d Numpy array
    :param x: Numpy array
    :param categorical_controls:
    :param check_rank:
    :param estimate_variance:
    :param get_residual:
    :param cluster:
    :return:
    """
    assert y.ndim == 2

    if categorical_controls is None or len(categorical_controls) == 0:
        print('No categorical controls')
        b = np.linalg.lstsq(x, y)[0]
        assert b.ndim == 2
        if estimate_variance or get_residual:
            error = y - x.dot(b)
            assert error.shape == y.shape
    # within estimator
    elif len(categorical_controls) == 1:
        print('Using within estimator')
        grouped = Groupby(data[categorical_controls[0]].values)
        x_demeaned = grouped.apply(lambda z: z - np.mean(z, 0), x, shape=x.shape)
        # k x n_outcomes
        b = np.linalg.lstsq(x_demeaned, y)[0]
        assert b.ndim == 2
        error = y - x.dot(b)
        assert error.shape == y.shape
        # n_teachers x n_outcomes
        fixed_effects = grouped.apply(lambda arr: np.mean(arr, 0), error, broadcast=False,
                                        shape=(grouped.n_keys, y.shape[1]))
        assert fixed_effects.ndim == 2
        # (n_teachers + k) x n_outcomes
        b = np.concatenate((fixed_effects, b))
        x = sps.hstack((make_dummies(data[categorical_controls[0]], False), x)).tocsr()
        assert b.shape[0] == x.shape[1]
        if estimate_variance or get_residual:
            error -= fixed_effects[data[categorical_controls[0]].values]
    else:
        dummies = get_all_dummies(data[categorical_controls].values)
        x = sps.hstack((dummies, x))
        assert sps.issparse(x)
        if check_rank:
            x = sps.csc_matrix(remove_collinear_cols(x.A))
        print(x.shape)
        print(y.shape)
        if y.ndim == 1 or y.shape[1] == 1:
            b = sps.linalg.lsqr(x, y)[0]
        else:
            b = np.zeros((x.shape[1], y.shape[1]), order='F')
            for i in range(y.shape[1]):
                b[:, i] = sps.linalg.lsqr(x, y[:, i], atol=1e-10)[0]

        if estimate_variance or get_residual:
            if b.ndim == 1:
                b = b[:, None]
            assert b.ndim == 2
            predicted = x.dot(b)
            assert y.shape == predicted.shape
            error = y - predicted
            assert error.shape == y.shape

    assert np.all(np.isfinite(b))
    if not estimate_variance and not get_residual:
        return b, x

    if get_residual:
        return b, x, error

    if estimate_variance:
        assert b.shape[0] == x.shape[1]
        _, r = np.linalg.qr(x if type(x) is np.array else x.A)

        inv_r = scipy.linalg.solve_triangular(r, np.eye(r.shape[0]))
        inv_x_prime_x = inv_r.dot(inv_r.T)
        if cluster is not None:
            grouped = Groupby(data[cluster])

            def f(mat):
                return mat[:, 1:].T.dot(mat[:, 0])

            V = []
            for i in range(y.shape[1]):
                u_ = grouped.apply(f, np.hstack((error[:, i, None], x.A)),
                                     shape=(grouped.n_keys, x.shape[1]), broadcast=False)

                inner = u_.T.dot(u_)
                # really mysterious why V is rank deficient. Surely I checked this formula?
                # Oh, I remember. the issue is having teachers with only one cluster,
                # So fixed effects make average error zero within a cluster.
                # TODO: see if I can take this out with more covariates.
                V.append(inv_x_prime_x.dot(inner).dot(inv_x_prime_x))
        else:
            error_sums = np.sum(error**2, 0)
            assert len(error_sums) == y.shape[1]
            V = [inv_x_prime_x * es / (len(y) - x.shape[1]) for es in error_sums]

        return b, x, error, V


def make_one_lag(array, lag, axis, fill_missing=False):
    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
        assert (axis == 1)

    # I have no idea why this is here, but it doesn't apply
    # for usual data format
    if abs(lag) > array.shape[axis]:
        if fill_missing:
            lags = np.zeros(array.shape)
            missing = np.ones(array.shape)
            if axis == 1:
                return np.vstack((lags, missing))
            else:
                return np.hstack((lags, missing))
        else:
            return np.full(array.shape, np.nan)

    # (1, 5) when starting with an array of size (93,5) and lag 1
    missing_shape = (array.shape[0], abs(lag)) if axis == 1 \
        else (abs(lag), array.shape[1])
    # (92, 5) when starting with an array of size (93, 5) and lag 1
    other_shape = (array.shape[0], array.shape[1] - abs(lag)) if axis == 1 \
        else (array.shape[0] - abs(lag), array.shape[1])

    if fill_missing:
        missing_ind = np.ones(missing_shape)
        missing_zero = np.zeros(missing_shape)
        not_missing = np.zeros(other_shape)

        if axis == 1:
            if lag > 0:
                lags = np.hstack((missing_zero, array[:, :-lag]))
                missing = np.hstack((missing_ind, not_missing))
            if lag < 0:
                lags = np.hstack((array[:, -lag:], missing_zero))
                missing = np.hstack((not_missing, missing_ind))
            return np.vstack((lags, missing))
        else:
            if lag > 0:
                # So with one lag, first row is zeros
                lags = np.vstack((missing_zero, array[:-lag, :]))
                missing = np.vstack((missing_ind, not_missing))
            if lag < 0:
                lags = np.vstack((array[-lag:, :], missing_zero))
                missing = np.vstack((not_missing, missing_ind))

            return np.hstack((lags, missing))

    else:
        missing_nan = np.full(missing_shape, np.nan)
        if axis == 1:
            if lag > 0:
                return np.hstack((missing_nan, array[:, :-lag]))
            if lag < 0:
                return np.hstack((array[:, -lag:], missing_nan))
        else:
            if lag > 0:
                return np.vstack((missing_nan, array[:-lag, :]))
            if lag < 0:
                return np.vstack((array[-lag:, :], missing_nan))


def make_lags(df, n_lags_back, n_lags_forward, outcomes, groupby,
              fill_zeros):
    lags = list(range(-1 * n_lags_forward, 0)) + list(range(1, n_lags_back + 1))
    grouped = Groupby(df[groupby].values)
    outcome_data = df[outcomes].values

    for lag in lags:
        def f(x):
            return make_one_lag(x, lag, 0, fill_zeros)

        width = 2 * len(outcomes) if fill_zeros else len(outcomes)

        new_data = grouped.apply(f, outcome_data, True, shape=(len(df), width))
        new_cols = [out + '_lag_' + str(lag) for out in outcomes]
        if fill_zeros:
            new_cols += [out + '_lag_' + str(lag) + '_mi'
                         for out in outcomes]

        for i, c in enumerate(new_cols):
            df.loc[:, c] = new_data[:, i]

    if fill_zeros:
        lag_vars = {out: list(chain(*([out + '_lag_' + str(lag), out + '_lag_' + str(lag) + '_mi']
                                      for lag in lags))) for out in outcomes}
        for out in outcomes:
            for lag in lags:
                name = out + '_lag_' + str(lag)
                missing = pd.isnull(df[name]) | df[name + '_mi'] == 1
                df.loc[missing, name] = 0
                df.loc[missing, name + '_mi'] = 1

    else:
        lag_vars = {out: [out + '_lag_' + str(lag) for lag in lags]
                    for out in outcomes}

    return df, lag_vars
