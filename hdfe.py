import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg

# whether to profile
if False:
    def profile(function):
        return function

cpp = True
if cpp:
    import cppimport
    cppplay = cppimport.imp("cppplay")
    class Groupby:
        @profile
        def __init__(self, keys):
            _, self.keys_as_int = np.unique(keys, return_inverse = True)
            self.internal = cppplay.Groupby(self.keys_as_int)
        
        @profile
        def apply(self, function, vector):
            return self.internal.apply_2(vector, function)
else:
    class Groupby:
        @profile
        def __init__(self, keys):
            _, self.keys_as_int = np.unique(keys, return_inverse = True)
            self.n_keys = max(self.keys_as_int) + 1
            self.set_indices()

        @profile
        def set_indices(self):
            self.indices = [[] for i in range(self.n_keys)]
            for i, k in enumerate(self.keys_as_int):
                self.indices[k].append(i)
            self.indices = [np.array(elt) for elt in self.indices]

        @profile
        def apply(self, function, vector):
            result = np.zeros(len(vector))
            for k in range(self.n_keys):
                result[self.indices[k]] = function(vector[self.indices[k]])
            return result

@profile
def estimate_with_alternating_projections(y, z, categorical_data):
    z = np.expand_dims(z, 1)
    n_cols =  z.shape[1]
    n, num_fes = categorical_data.shape

    grouped = [Groupby(categorical_data[:, i]) for i in range(num_fes)]
    z_projected = np.zeros(z.shape)

    # Project each column of z onto dummies
    for col in range(n_cols):
        fixed_effects = np.zeros((n, num_fes))
        ssr = np.dot(z[:, col], z[:, col])
        ssr_last = 10 * ssr

        while (ssr_last - ssr) / ssr_last > 10**(-5):
            ssr_last = ssr
            residual = z[:, col] - np.sum(fixed_effects, 1)
            for fe in range(num_fes):
                fixed_effects[:, fe] = fixed_effects[:, fe] \
                                     + grouped[fe].apply(np.mean, residual)
                residual = z[:, col] - np.sum(fixed_effects, 1)
            ssr = np.dot(residual, residual)

        z_projected[:, col] = z[:, col] - np.sum(fixed_effects, 1)

    beta = np.linalg.lstsq(z_projected, y)[0]

    return beta, 1

def estimate_brute_force(y, z, categorical_data):
    k = z.shape[1] if len(z.shape) > 1 else 1
    print(k)
    num_fes = categorical_data.shape[1]

    def get_dummies(v):
        _, data_as_int = np.unique(v, return_inverse = True)
        return sps.csc_matrix((np.ones(len(v)), (range(len(v)), v)))

    dummies = sps.hstack([get_dummies(categorical_data[:, i])
                          for i in range(num_fes)]).astype(int)
    z = sps.csc_matrix(np.expand_dims(z, 1))
    rhs = sps.hstack((z, dummies))
    params = sps_linalg.lsqr(rhs, y)[0]
    return params[:k], params[k:]


def estimate_coefficients(y, z, categorical_data, method):
    if method == 'alternating projections':
        return estimate_with_alternating_projections(y, z, categorical_data)
    elif method == 'brute force':
        return estimate_brute_force(y, z, categorical_data)

    print('You did not specify a valid method.')
    return
