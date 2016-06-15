import pandas as pd
from hdfe import estimate_coefficients
import numpy as np

def test_hdfe():
    beta = 3
    n = 10**4
    k1 = 10**3
    k2 = 10**2
    
    fes1 = np.random.gamma(1, 1, k1)
    fes2 = np.random.gamma(1, 1, k2)
    
    z = np.array(np.random.normal(0, 1, n))
    cat1 = np.random.choice(k1, n)
    cat2 = np.random.choice(k2, n)
    y = z * beta + fes1[cat1] + fes2[cat2] + np.random.normal(0, 20, n)
              
    beta, fixed_effects = estimate_coefficients(y, z, np.vstack((cat1, cat2)).T)
    return
    
if __name__ == "__main__":
    test_hdfe()
