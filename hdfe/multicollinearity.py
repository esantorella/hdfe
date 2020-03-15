import warnings
from typing import Iterable, List, Tuple, Union

import numpy as np
import scipy.sparse as sps


def remove_cols_from_csc(
    x: sps.csc_matrix, cols_to_remove: Iterable[int]
) -> sps.spmatrix:
    """
    Efficiently removes columns from a CSC sparse matrix by efficiently editing the
    underlying data.
    :param x: CSC sparse matrix
    :param cols_to_remove:
    :return: CSC sparse matrix

    >>> from scipy import sparse as sps
    >>> x = sps.eye(3, dtype=int).tocsc()
    >>> cols_to_remove = [1]
    >>> remove_cols_from_csc(x, cols_to_remove).A
    array([[1, 0],
           [0, 0],
           [0, 1]])
    """

    if not sps.issparse(x):
        raise ValueError

    if not sps.isspmatrix_csc(x):
        raise ValueError("Can only remove columns from a csc matrix.")

    def remove_one_col(idx: List[int], ptr_: np.ndarray, data_: List[int], col_: int):
        n_elts_to_remove = ptr_[col_ + 1] - ptr_[col_]
        idx = idx[: ptr_[col_]] + idx[ptr_[col_ + 1] :]
        data_ = data_[: ptr_[col_]] + data_[ptr_[col_ + 1] :]
        ptr_ = np.concatenate((ptr_[:col_], ptr_[col_ + 1 :] - n_elts_to_remove))
        return data_, idx, ptr_

    indices = list(x.indices)
    ptr = x.indptr
    data = list(x.data)

    for i, col in enumerate(cols_to_remove):
        data, indices, ptr = remove_one_col(indices, ptr, data, col - i)

    return sps.csc_matrix((data, indices, ptr))


def find_collinear_cols(
    x: Union[np.ndarray, sps.spmatrix], tol: float = 10 ** (-12), verbose: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Identifies a minimal subset of columns of x that, when removed, make x full rank.
    Note that there may be many such subsets. This function relies on a QR decomposition
    and may be numerically unstable.

    :param x: Numpy array or something that can be converted to a Numpy array. It will
        be converted to a Numpy array.
    :param tol: A higher tolerance leads to erring on the side of identifying more
        columns as collinear.
    :param verbose:
    :return: List of columns that when removed make x full rank, and a list of all of
        the other columns

    >>> x = np.array([[1, 1], [0, 0]])
    >>> x
    array([[1, 1],
           [0, 0]])
    >>> find_collinear_cols(x)
    ([1], [0])
    """
    k = x.shape[1]
    x = np.asarray(x)
    if x.shape[0] == k:
        rank = np.linalg.matrix_rank(x)
    else:
        rank = np.linalg.matrix_rank((x.T.dot(x)))
    full_rank = rank == k

    if full_rank:
        if verbose:
            print("Full rank")
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
        print("Minimum not deleted:", min_not_deleted)
        print("Number collinear", len(collinear_cols))
    if len(non_collinear_cols) != rank:
        warnings.warn(f"Rank is {rank}, but there are {len(non_collinear_cols)} left.")

    return collinear_cols, non_collinear_cols


def remove_collinear_cols(
    x: Union[sps.spmatrix, np.ndarray], verbose: bool = False
) -> Union[sps.spmatrix, np.ndarray]:
    """
    Removes a minimal subset of columns from x such that x becomes full rank. Note that
        these columns are not uniquely defined.

    >>> x = np.array([[1, 1], [0, 0]])
    >>> remove_collinear_cols(x)
    array([[1],
           [0]])
    """
    collinear, not_collinear = find_collinear_cols(x, verbose=verbose)
    if len(collinear) == 0:
        if verbose:
            print("No collinear columns")
        return x
    if verbose:
        print("Number of collinear columns:", len(collinear))
        print("Number of non-collinear columns:", len(not_collinear))

    if isinstance(x, sps.csc.csc_matrix):
        return remove_cols_from_csc(x, collinear)
    if isinstance(x, sps.coo.coo_matrix):
        x = x.asformat("csc")
    if isinstance(x, np.ndarray):
        return x[:, not_collinear]
    raise TypeError("Not implmented for type ", type(x))
