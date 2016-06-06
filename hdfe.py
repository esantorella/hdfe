import numpy as np
from va_functions import *

class Groupby:
    def __init__(self, keys):
        self.unique_keys = frozenset(keys)
        self.set_indices(keys)
        
    def set_indices(self, keys):
        self.indices = {k:[] for k in self.unique_keys}
        for i, k in enumerate(keys):
            self.indices[k].append(i)
            
    def apply(self, values, function):
        result = np.zeros(len(values))
        for k in self.unique_keys:
            result[self.indices[k]] = function(values[self.indices[k]])
        return result

def get_beta(y, z_projection, fixed_effects):
    residual = y - np.sum(fixed_effects, axis=1)
    return z_projection @ residual 
   
def get_fes(y, fixed_effects, index, key, unique_keys, group_indices):
    use_fes = list(range(0, index)) + list(range(index + 1, fixed_effects.shape[1]))
    residual =  y - np.sum(fixed_effects[:, use_fes], axis=1)
    return get_group_mean(residual, key, unique_keys, group_indices)
    
def estimate_coefficients(y, z, categorical_data):
    z_projection = np.linalg.lstsq(z.T @ z, z.T)
    n, num_fes = categorical_data.shape
    
    # initialize
    fixed_effects = np.zeros((n, num_fes))
    unique_keys = [frozenset(categorical_data[:, i]) for i in range(num_fes)]
    group_indices = [get_group_indices(categorical_data[:, i], k) 
                     for i, k in enumerate(unique_keys)]
    
    beta = z_projection @ y
    beta_resid = y - z @ beta
    ssr_initial = np.sum(residual**2)
    ssr = ssr_initial
    i = 0
    
    while (ssr - ssr_initial) / ssr_initial > .1:
        # first update fixed effects
        for j in range(len(categorical_vars)):
            fixed_effects[:, j] = get_fes(beta_resid, fixed_effects, j, 
                                          categorical_data[:, j], 
                                          unique_keys[j], group_indices[j])
            

        beta = get_beta(y, z_projection, fixed_effects) # then update beta
        beta_resid = y - z @ beta
        ssr = np.sum((beta_resid - np.sum(fixed_effects, axis=1))**2)
        print(beta)
        print((ssr - ssr_initial) / ssr_initial)
        i+= 1

    return beta, fixed_effects 
