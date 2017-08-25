import numpy as np
import scipy.sparse as sps


def remove_cols_from_csc(x, cols_to_remove):

    def remove_one_col(idx, ptr_, data_, col_):
        n_elts_to_remove = ptr_[col_ + 1] - ptr_[col_]
        idx = idx[:ptr_[col_]] + idx[ptr_[col_ + 1]:]
        data_ = data_[:ptr_[col_]] + data_[ptr_[col_ + 1]:]
        ptr_ = np.concatenate((ptr_[:col_], ptr_[col_ + 1:] - n_elts_to_remove))
        return idx, ptr_, data_

    indices = list(x.indices)
    ptr = x.indptr
    data = list(x.data)

    for i, col in enumerate(cols_to_remove):
        indices, ptr, data = remove_one_col(indices, ptr, data, col - i)

    return sps.csc_matrix((data, indices, ptr))


def find_collinear_cols(x, tol=10**(-12), verbose=False):
    k = x.shape[1]
    x = np.asarray(x)
    full_rank = np.linalg.matrix_rank((x.T.dot(x))) == k
    if full_rank:
        if verbose:
            print('Full rank')
        return [], list(range(k))

    _, r = np.linalg.qr(x)
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
    if verbose:
        print('Minimum not deleted:', min_not_deleted)

    return collinear_cols, non_collinear_cols


def remove_collinear_cols(x, verbose=False):
    collinear, not_collinear = find_collinear_cols(x, verbose=verbose)
    if len(collinear) == 0:
        if verbose:
            print('No collinear columns')
        return x
    if verbose:
        print('Number of collinear columns:', len(collinear))
        print('Number of non-collinear columns:', len(not_collinear))
    if type(x) is sps.coo.coo_matrix:
        x = x.asformat('csc')
    if type(x) is sps.csc.csc_matrix:
        return remove_cols_from_csc(x, collinear)
    if type(x) is np.ndarray:
        return x[:, not_collinear]
    raise TypeError('Not implmented for type ', type(x))
