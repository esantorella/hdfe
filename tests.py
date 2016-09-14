from hdfe import estimate_coefficients
import numpy as np
import time

def test_hdfe():
    n = 10**6

    # Two categorical variables, 1 and 2
    k1 = 10**4
    k2 = 10**2

    # Two outcomes, a and b
    beta_a = [3, 4]
    beta_b = [-1, 7]

    fes_1a = np.random.gamma(1, 1, k1)
    fes_1b = np.random.gamma(1, 1, k1)
    fes_2a = np.random.gamma(1, 1, k2)
    fes_2b = np.random.gamma(1, 1, k2)

    # Controls: Same for both outcomes
    z = np.random.normal(0, 1, (n, 2))
    cat1 = np.random.choice(k1, n)
    cat2 = np.random.choice(k2, n)
    categorical_data = np.vstack((cat1, cat2)).T

    y_a = np.dot(z, beta_a) + fes_1a[cat1] + fes_2a[cat2]
    y_b = np.dot(z, beta_b) + fes_1b[cat1] + fes_2b[cat2]
    

    start = time.clock()
    beta_hat_bf, _ = estimate_coefficients(y_a, z, categorical_data, 'brute force')
    end = time.clock()
    print('brute force time:')
    print(end - start)
    start = time.clock()
    beta_hat_ap, _ = estimate_coefficients(y_a, z, categorical_data,
                                           'alternating projections')
    end = time.clock()
#    print('alternating projections error:')
#    print(abs(beta_hat_ap - beta_hat_bf))
    print('alternating projections time:')
    print(end - start)
    return

if __name__ == "__main__":
    test_hdfe()
