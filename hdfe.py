import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_linalg

class Groupby:
    def __init__(self, keys):
        self.unique_keys = frozenset(keys)
        self.set_indices(keys)
        
    def set_indices(self, keys):
        self.indices = {k:[] for k in self.unique_keys}
        for i, k in enumerate(keys):
            self.indices[k].append(i)
            
    def apply(self, function, vector):
        result = np.zeros(len(vector))
        for k in self.unique_keys:
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
