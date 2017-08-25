import numpy as np
import scipy.sparse as sps
import sys
sys.path.append('/Users/lizs/Dropbox/India bureaucrats/Bureaucrat-Value-Added/PySPQR/')
import spqr


def remove_cols_from_csc(x, cols_to_remove):
    def remove_one_col(idx, ptr, data, col):
        n_elts_to_remove = ptr[col+1] - ptr[col]
        idx = idx[:ptr[col]] + idx[ptr[col + 1]:]
        data = data[:ptr[col]] + data[ptr[col+1]:]
        ptr = np.concatenate((ptr[:col], ptr[col+1:] - n_elts_to_remove))
        return idx, ptr, data

    indices = list(x.indices)
    ptr = x.indptr
    data = list(x.data)

    for i, col in enumerate(cols_to_remove):
        indices, ptr, data = remove_one_col(indices, ptr, data, col - i)

    return sps.csc_matrix((data, indices, ptr))


def find_collinear_cols(x, tol=10**(-12)):
    # if np.any((x!= 0).sum(0) == 0):
    #     raise ValueError('Some columns are all zero; remove and try again')
    k = x.shape[1]
    if type(x) is np.ndarray:
        full_rank = np.linalg.matrix_rank((x.T.dot(x))) == k
        if not full_rank:
            _, r = np.linalg.qr(x)
    else:
        #    assert np.all(x.sum(0) > 0)
        _, r, e, rank = spqr.qr(x)
        full_rank = rank == k
        if not full_rank:
            r = sps.csc_matrix(r)

    if full_rank:
        print('Full rank')
        return [], list(range(k))
    row = 0

    non_collinear_cols = []
    collinear_cols = []
    min_not_deleted = 1
    for col in range(r.shape[1]):
        if row >= r.shape[0]:
            collinear_cols += list(range(col, r.shape[1]))
            break
        if abs(r[row, col]) < tol:
            collinear_cols.append(col)
        else:
            non_collinear_cols.append(col)
            min_not_deleted = min(min_not_deleted, abs(r[row, col]))
            row += 1
    print('Minimum not deleted:', min_not_deleted)
    if sps.issparse(x):
        collinear_cols = e[collinear_cols]
        non_collinear_cols = np.sort(e[non_collinear_cols])
    return collinear_cols, non_collinear_cols


def remove_collinear_cols(x):
    collinear, not_collinear = find_collinear_cols(x)
    if len(collinear) == 0:
        return x
    print('Number of collinear columns:', len(collinear))
    print('Number of non-collinear columns:', len(not_collinear))
    if type(x) is sps.coo.coo_matrix:
        x = x.asformat('csc')
    if type(x) is sps.csc.csc_matrix:
        return remove_cols_from_csc(x, collinear)
    if type(x) is np.ndarray:
        return x[:, not_collinear]
    raise TypeError('Not implmented for type ', type(x))
