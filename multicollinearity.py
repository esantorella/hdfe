import numpy as np
import scipy.sparse as sps
import numpy.linalg as linalg


def remove_cols_from_csc(x, cols_to_remove):
    def remove_one_col(indices, ptr, data, col):
        n_elts_to_remove = ptr[col+1] - ptr[col]
        indices = indices[:ptr[col]] + indices[ptr[col+1]:]
        data = data[:ptr[col]] + data[ptr[col+1]:]
        ptr = np.concatenate((ptr[:col], ptr[col+1:] - n_elts_to_remove))
        return indices, ptr, data

    indices = list(x.indices)
    ptr = x.indptr
    data = list(x.data)

    for i, col in enumerate(cols_to_remove):
        indices, ptr, data = remove_one_col(indices, ptr, data, col - i)

    return sps.csc_matrix((data, indices, ptr))


def find_collinear_cols(x):
    # TODO: this only works on dense matrices
    if type(x) is np.ndarray:    
        _, r = np.linalg.qr(x)
    else:
        _, r = np.linalg.qr(x.A)
    row = 0

    non_collinear_cols = []
    collinear_cols = []
    min_not_deleted = 1
    for col in range(r.shape[1]):
        if row >= r.shape[0]:
            collinear_cols += list(range(col, r.shape[1]))
            break
        if abs(r[row, col]) < 10**(-11):
            collinear_cols.append(col)
        else:
            non_collinear_cols.append(col)
            min_not_deleted = min(min_not_deleted, abs(r[row, col]))
            row += 1
    print('Minimum value of non-deleted elements', min_not_deleted)
    return collinear_cols, non_collinear_cols


def remove_collinear_cols(x):
    collinear, not_collinear = find_collinear_cols(x)
    if len(collinear) > 0:
        print('Number of collinear columns: ', len(collinear))
        print('Number not collinear: ', len(not_collinear))
        print('Earliest column being removed: ', np.min(collinear))
        print('Collinear columns: ', collinear)
        if type(x) is sps.coo.coo_matrix:
            x = x.asformat('csc')
        if type(x) is sps.csc.csc_matrix:
            return remove_cols_from_csc(x, collinear)
        if type(x) is np.ndarray:
            return x[:, not_collinear]
        raise TypeError('Not implmented for type ', type(x))
    else:
        print('Rank-deficient, but no collinear columns found')
        return x

