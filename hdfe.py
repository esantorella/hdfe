import numpy as np

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
    
    
def estimate_with_alternating_projections(y, z, categorical_data):
    n, num_fes = categorical_data.shape
    
    # initialize
    y = y - np.mean(y)  
    if len(z.shape) ==1:
        z = np.expand_dims(z, 1)
        
    beta = np.linalg.lstsq(z, y)[0]
    residual = y - np.dot(z, beta)
    ssr =  np.dot(residual, residual)
    ssr_last = 10 * ssr
    i = 0
    
    fixed_effects = np.zeros((n, num_fes))
    grouped = [Groupby(categorical_data[:, i]) for i in range(num_fes)]

    print('\n ratio')
    print((ssr_last - ssr) / ssr_last)
    print("beta " + str(beta))
    
    while (ssr_last - ssr) / ssr_last > 10**(-5):
        print("i" + str(i))
        # first update fixed effects
        for j in range(num_fes):
            print(j)
            residual = residual + fixed_effects[:, j]
            fixed_effects[:, j] = grouped[j].apply(np.mean, residual)
            residual = residual - fixed_effects[:, j]
            assert(np.dot(residual, residual) < ssr_last)
            
        # then update beta
        beta = np.linalg.lstsq(z, y - np.sum(fixed_effects, axis = 1))[0]
        # Then update loop variables
        ssr_last = ssr
        ssr = np.dot(residual, residual)
        assert(ssr < ssr_last)
        print('\n ratio')
        print((ssr_last - ssr) / ssr_last)
        print("beta " + str(beta))
        i+= 1

    return beta, fixed_effects 
    
def estimate_coefficients(y, z, categorical_data, method):
    if method == 'alternating projections':
        return estimate_with_alternating_projections(y, z, categorical_data)
    print('You did not specify a valid method.')
    return
