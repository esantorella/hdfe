import pandas as pd
from hdfe import estimate_coefficients
import numpy as np
import time

def test_hdfe():
    beta = 3
    n = 10**4
    k1 = 10**3
    k2 = 10**2
    
    fes1 = np.random.gamma(1, 1, k1)
    fes2 = np.random.gamma(1, 1, k2)
    
    z = np.random.normal(0, 1, n)
    cat1 = np.random.choice(k1, n)
    cat2 = np.random.choice(k2, n)
    y = z * beta + fes1[cat1] + fes2[cat2]
    
    start = time.clock()
    beta_hat_bf, _ = estimate_coefficients(y, z, np.vstack((cat1, cat2)).T, 
                                           'brute force')
    end = time.clock()
    print('brute force error:')
    print(abs(beta_hat_bf - beta))
    print('brute force time:')
    print(end - start)
    start = time.clock()
    beta_hat_ap, _ = estimate_coefficients(y, z, np.vstack((cat1, cat2)).T, 
                                           'alternating projections')
    end = time.clock()
    print('alternating projections error:')
    print(abs(beta_hat_ap - beta))
    print('alternating projections time:')
    print(end - start)
    return
    
if __name__ == "__main__":
    test_hdfe()
