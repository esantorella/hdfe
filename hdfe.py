import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg
import scipy.linalg
import time
from multicollinearity import remove_collinear_cols


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
        return get_dummies(categorical_data, False)

    num_fes = categorical_data.shape[1]
    return sps.hstack((get_dummies(categorical_data[:, 0]),
                    sps.hstack((get_dummies(categorical_data[:, col]))
                       for col in range(num_fes))))


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
             estimate_variance=False, get_residual=False):
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
        dummies = get_all_dummies(data[categorical_controls].values)
        x = sps.hstack((dummies, x))
        if check_rank:
            rank = np.linalg.matrix_rank(x.todense())
            if rank < x.shape[1]:
                warnings.warn('x is rank deficient, attempting to correct')
                x = remove_collinear_cols(x)
        b = sps.linalg.lsqr(x, y)[0]
        if estimate_variance or get_residual:
            errors = y - x.dot(b)

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
        V = inv_x_prime_x * error.dot(error) / (len(y) - x.shape[1])
        return b, x, error, V
