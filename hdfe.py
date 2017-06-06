import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg
import scipy.linalg
import time
from multicollinearity import remove_collinear_cols
from itertools import chain
import pandas as pd
import warnings


expand_dims = lambda v: np.expand_dims(v, 1) if len(v.shape) == 1 else v

def make_dummies(elt, drop_col):
    if np.max(elt) >= len(set(elt)):
        _, elt = np.unique(elt, return_inverse=True)

    dummies = sps.csc_matrix((np.ones(len(elt)), (range(len(elt)), elt)))
    if drop_col:
        return dummies[:, :-1]
    else:
        return dummies

        
def get_all_dummies(categorical_data, drop):
    if len(categorical_data.shape) == 1 or categorical_data.shape[1] == 1:
        return make_dummies(categorical_data, False)

    num_fes = categorical_data.shape[1]
    first = make_dummies(categorical_data[:, 0], False)
    others = [make_dummies(categorical_data[:, col], True)
              for col in range(1, num_fes)]
    others = sps.hstack(others)
    return sps.hstack((first, others))


cpp = False
if cpp:
    import cppimport
    cppplay = cppimport.imp("cppplay")
    class Groupby:
        def __init__(self, keys):
            _, self.keys_as_int = np.unique(keys, return_inverse = True)
            self.internal = cppplay.Groupby(self.keys_as_int)
        
        def apply(self, function, vector):
            return self.internal.apply_2(vector, function)
else:
    class Groupby:
        def __init__(self, keys):
            _, self.first_occurrences, self.keys_as_int = \
                     np.unique(keys, return_index = True, return_inverse = True)
            self.n_keys = max(self.keys_as_int) + 1
            self.set_indices()

        def set_indices(self):
            self.indices = [[] for i in range(self.n_keys)]
            for i, k in enumerate(self.keys_as_int):
                self.indices[k].append(i)
            self.indices = [np.array(elt) for elt in self.indices]

        def apply(self, function, array, broadcast=True, width=None):
            if broadcast:
                if width is None:
                    result = np.zeros(array.shape)
                else:
                    result = np.zeros((array.shape[0], width))

                for k in range(self.n_keys):
                    result[self.indices[k]] = function(array[self.indices[k]])

            else:
                if width is None:
                    result = np.zeros(self.n_keys)
                    for k in range(self.n_keys):
                        result[self.keys_as_int[self.first_occurrences[k]]] =\
                                function(array[self.indices[k]])
                else:
                    result = np.zeros((self.n_keys, width))
                    for k in range(self.n_keys):
                        result[self.keys_as_int[self.first_occurrences[k]], :] = \
                                             function(array[self.indices[k]])
            return result
        


# TODO: recover fixed effects
def estimate_with_alternating_projections(y, z, categorical_data):
    k =  z.shape[1]
    if k == 1: 
        z = np.expand_dims(z, 1) 
    n, num_fes = categorical_data.shape

    grouped = [Groupby(categorical_data[:, i]) for i in range(num_fes)]
    z_projected = np.zeros(z.shape)

    # Project each column of z onto dummies
    for col in range(k):
        fixed_effects = np.zeros((n, num_fes))
        ssr = np.dot(z[:, col], z[:, col])
        ssr_last = 10 * ssr

        while (ssr_last - ssr) / ssr_last > 10**(-5):
            ssr_last = ssr
            residual = z[:, col] - np.sum(fixed_effects, 1)
            for fe in range(num_fes):
                fixed_effects[:, fe] = fixed_effects[:, fe] \
                            + grouped[fe].apply(lambda x: x.mean(), residual)
                residual = z[:, col] - np.sum(fixed_effects, 1)
            ssr = np.dot(residual, residual)

        z_projected[:, col] = z[:, col] - np.sum(fixed_effects, 1)

    beta = np.linalg.lstsq(z_projected, y)[0]
    return beta, fixed_effects
    

def estimate_within(y, z, categorical_data):
    assert(len(categorical_data.shape) == 1 or categorical_data.shape[1] == 1)
    grouped = Groupby(categorical_data)
    if z is None:
        fixed_effects = grouped.apply(lambda x: x.mean(), y)    
        return None, fixed_effects

    if sps.issparse(z):
        z = z.todense()
    if len(z.shape) == 1: z = np.expand_dims(z, 1)
    k = z.shape[1]
    z_projected_resid = np.zeros(z.shape)
    for col in range(k):
        z_projected_resid[:, col] = np.squeeze(z[:, col] - \
                                   grouped.apply(lambda x: x.mean(), z[:, col]))

    beta = np.linalg.lstsq(z_projected_resid, y)[0]
    fixed_effects = grouped.apply(lambda x: x.mean(), y - np.dot(z, beta))
    return beta, fixed_effects
        


def estimate_brute_force(y, z, categorical_data):
    if len(z.shape) == 1: z = np.expand_dims(z, 1)
    if len(categorical_data.shape) == 1:
        categorical_data = np.expand_dims(categorical_data, 1)

    num_fes = categorical_data.shape[1]

    dummies= get_all_dummies(categorical_data, drop=True) 
    
    rhs = sps.hstack((z, dummies))
    params = sps_linalg.lsqr(rhs, y)[0]

    k = z.shape[1]
    return params[:k], dummies * params[k:]
    

def estimate_coefficients(y, z, categorical_data, method):
    if method == 'alternating projections':
        return estimate_with_alternating_projections(y, z, categorical_data)
    elif method == 'brute force':
        return estimate_brute_force(y, z, categorical_data)
    elif method == 'exploit sparsity pattern':
        return exploit_sparsity_pattern(y, z, categorical_data)
    elif method == 'within':
        return estimate_within(y, z, categorical_data)

    print('You did not specify a valid method.')
    assert(False)
    return

# Automatically picks best method, takes pandas df
# TODO: return variance estimate if desired
def estimate(data, y, x, categorical_controls, check_rank=False, 
             estimate_variance=False, get_residual=False,
             cluster=None):
    """
    data: pandas dataframe
    y: 1d numpy array
    x: numpy array
    categorical_controls: list of strings that are columns in data
    """

    if categorical_controls is None:
        b = np.linalg.lstsq(x, y)[0]
        if estimate_variance or get_residual:
            error = y - x.dot(b)
    # within estimator
    elif len(categorical_controls) == 1:
        grouped = Groupby(data[categorical_controls[0]].values)
        x_demeaned = grouped.apply(lambda z: z - np.mean(z, 0), x)
        b = np.linalg.lstsq(x_demeaned, y)[0]
        error = y - x.dot(b)
        fixed_effects = grouped.apply(np.mean, error, broadcast=False)
        b = np.concatenate((fixed_effects, b))
        x = sps.hstack((make_dummies(data[categorical_controls[0]], False), x))
        assert b.shape[0] == x.shape[1]
        if estimate_variance or get_residual:
            error -= fixed_effects[data[categorical_controls[0]].values]
    else:
        dummies = get_all_dummies(data[categorical_controls].values, True)
        dense_shape = x.shape[1]
        x = sps.hstack((dummies, x))
        if check_rank:
            rank = np.linalg.matrix_rank(x.todense())
            if rank < x.shape[1]:
                warnings.warn('x is rank deficient, attempting to correct')
                x = remove_collinear_cols(x)
                try:
                    x = x.A
                except AttributeError:
                    pass

                rank = np.linalg.matrix_rank(x)
                if rank < x.shape[1]:
                    # Not sure why this happened, but drop a dummy as
                    # a stupid hack
                    x = np.hstack((x[:, :-1-dense_shape], x[:, -1 * dense_shape:]))
                    rank = np.linalg.matrix_rank(x)
                    if rank < x.shape[1]:
                        import ipdb; ipdb.set_trace()

        b = sps.linalg.lsqr(x, y)[0]
        if estimate_variance or get_residual:
            error = y - x.dot(b)

    assert np.all(np.isfinite(b))
    if not estimate_variance and not get_residual:
        return b, x

    if get_residual:
        return b, x, error

    if estimate_variance: 
        assert b.shape[0] == x.shape[1]
        if type(x) is np.ndarray:
           _, r = np.linalg.qr(x)
        else:
            _, r = np.linalg.qr(x.todense())

        inv_r = scipy.linalg.solve_triangular(r, np.eye(r.shape[0]))
        inv_x_prime_x = inv_r.dot(inv_r.T)
        if cluster is not None:
            assert False
            grouped = Groupby(data[cluster])
            def f(mat):
                return mat[:, 1:].T.dot(mat[:, 0])

            u_ = grouped.apply(f, np.hstack((np.expand_dims(error, 1), x.A)), 
                              width = x.shape[1], broadcast=False)

            inner = u_.T.dot(u_)
            V = inv_x_prime_x.dot(inner).dot(inv_x_prime_x)
        else:
            V = inv_x_prime_x * error.dot(error) / (len(y) - x.shape[1])

        return b, x, error, V


def make_one_lag(array, lag, axis, fill_missing = False):

    if len(array.shape) == 1:
        array = np.expand_dims(array, 0)
        assert(axis == 1)

    # I have no idea why this is here, but it doesn't apply
    # for usual data format
    if abs(lag) > array.shape[axis]:
        if fill_missing:
            lags = np.zeros((array.shape))
            missing = np.ones((array.shape))
            if axis == 1:
               return np.vstack((lags, missing))
            else:
                return np.hstack((lags, missing))
        else:
            return np.full(array.shape, np.nan)

    # (1, 5) when starting with an array of size (93,5) and lag 1
    missing_shape = (array.shape[0], abs(lag)) if axis == 1\
                    else (abs(lag), array.shape[1])
    # (92, 5) when starting with an array of size (93, 5) and lag 1
    other_shape = (array.shape[0], array.shape[1] - abs(lag)) if axis ==1\
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
    lags = list(range(-1*n_lags_forward, 0)) + list(range(1,n_lags_back+1))
    grouped = Groupby(df[groupby].values)
    outcome_data = df[outcomes].values

    for lag in lags:
        f = lambda x: make_one_lag(x, lag, 0, fill_zeros)
        shape = 2 * len(outcomes) if fill_zeros else len(outcomes)

        new_data = grouped.apply(f, outcome_data, True, 
                                 width = shape)
        new_cols = [out + '_lag_' + str(lag) for out in outcomes]
        if fill_zeros:
            new_cols += [out + '_lag_' + str(lag) + '_mi' 
                         for out in outcomes]

        for i, c in enumerate(new_cols):
            df.loc[:, c] = new_data[:, i]

    if fill_zeros:
        lag_vars = {out: 
    list(chain(*([out + '_lag_' + str(lag), out + '_lag_' + str(lag) + '_mi']
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
