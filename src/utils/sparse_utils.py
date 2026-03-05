"""Sparse-matrix utility functions for the MAIS simulation.

This module is no longer used in current versions of the model. It was
originally used when a contact graph was compressed into a sparse matrix
where multi-edges were aggregated into single weighted entries.

The functions operate on SciPy CSR/CSC sparse matrices and provide
element-wise row/column scaling, row/column product aggregation, and a
specialised element-wise multiplication that treats structurally-zero entries
as ones (useful for probability-complement arithmetic).
"""

# this is not used in a new versions of a model
# was used when a grah was compressed into a matrix (multi-edges were aggregated)

import numpy as np
import scipy.sparse as sparse


def multiply_row(A, row_idx, alpha, trunc=False):
    """Scale all stored values in a single row of a CSR matrix in place.

    Only the explicitly stored (non-zero) entries in the row are affected.
    Structural zeros are untouched.

    Args:
        A (scipy.sparse.csr_matrix): The sparse matrix to modify in place.
            Must be in CSR format.
        row_idx (int): Zero-based index of the row to scale.
        alpha (float): Scalar multiplier applied to every stored entry in the
            specified row.
        trunc (bool, optional): If ``True``, the scaled values are clipped to
            the interval ``[0.0, 1.0]`` after multiplication. Defaults to
            ``False``.
    """

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]

    A.data[idx_start_row:idx_end_row] = (alpha *
                                         A.data[idx_start_row:idx_end_row]
                                         )
    if trunc:
        A.data[idx_start_row:idx_end_row] = np.clip(
            A.data[idx_start_row:idx_end_row], 0.0, 1.0)


def multiply_col(A, col_idx, alpha, trunc=False):
    """Scale all stored values in a single column of a CSR matrix in place.

    Locates every explicitly stored entry in the given column and multiplies
    it by ``alpha``. Works on CSR format by scanning the ``indices`` array.

    Args:
        A (scipy.sparse.csr_matrix): The sparse matrix to modify in place.
            Must be in CSR format.
        col_idx (int): Zero-based index of the column to scale.
        alpha (float): Scalar multiplier applied to every stored entry in the
            specified column.
        trunc (bool, optional): If ``True``, the scaled values are clipped to
            the interval ``[0.0, 1.0]`` after multiplication. Defaults to
            ``False``.
    """
    col_indices = A.indices == col_idx
    A.data[col_indices] = (alpha * A.data[col_indices])
    if trunc:
        A.data[col_indices] = np.clip(A.data[col_indices], 0.0, 1.0)


def prop_of_row(A):
    """Compute the product of stored values in each row of a CSR matrix.

    For each row the function multiplies all explicitly stored (non-zero)
    entries together. Rows with no stored entries retain a product of ``1.0``
    (identity for multiplication), which corresponds to the convention that
    missing entries represent the value 1.

    Args:
        A (scipy.sparse.csr_matrix): Input sparse matrix in CSR format.

    Returns:
        numpy.ndarray: A 1-D array of shape ``(A.shape[0],)`` where element
        ``i`` is the product of all stored values in row ``i``.
    """

    result = np.ones(A.shape[0])

    i = 0
    n = len(A.indptr)
    while i < n-1:
        s, e = A.indptr[i], A.indptr[i+1]
        result[i] = np.prod(A.data[s:e])
        i += 1
    if A.indptr[i] < len(A.data):
        s = A.indptr[i]
        result[i] = np.prod(A.data[s:])
    return result


def prop_of_column(A):
    """Compute the product of stored values in each column of a CSR matrix.

    For each unique column index present in the matrix the function multiplies
    all explicitly stored entries in that column. Columns with no stored
    entries retain a product of ``1.0``.

    Args:
        A (scipy.sparse.csr_matrix): Input sparse matrix in CSR format.

    Returns:
        numpy.ndarray: A 1-D array of shape ``(A.shape[1],)`` where element
        ``j`` is the product of all stored values in column ``j``.
    """

    result = np.ones(A.shape[1])
    col_indices = A.indices
    #    print("columns", np.unique(col_indices))

    # print(".... prop_of_column fce ", A.indices, np.unique(col_indices), A.data)

    for col_idx in np.unique(col_indices):
        current_indices = A.indices == col_idx
        #        print(" .... ", A.data[current_indices], np.prod(1 - A.data[current_indices]))
        result[col_idx] = np.prod(A.data[current_indices])
    return result


def multiply_zeros_as_ones(a, b):
    """Element-wise multiply two sparse matrices treating structural zeros as ones.

    Standard sparse element-wise multiplication treats structurally-zero
    positions as ``0 * 0 = 0``. This function instead treats a missing entry
    (structural zero) in *either* matrix as the value ``1.0``, so that:

    - positions present in both ``a`` and ``b`` → ``a[i,j] * b[i,j]``
    - positions present only in ``a``             → ``a[i,j]``  (b treated as 1)
    - positions present only in ``b``             → ``b[i,j]``  (a treated as 1)
    - positions absent in both                    → ``0``  (stored as structural zero)

    This is useful for computing the joint probability of *no contact* across
    multiple probability layers, where an absent entry means "no edge, hence
    probability 1 of no contact on this layer".

    Args:
        a (scipy.sparse.csr_matrix): First sparse matrix.
        b (scipy.sparse.csr_matrix): Second sparse matrix. Must have the same
            shape as ``a``.

    Returns:
        scipy.sparse.csr_matrix: Result matrix with the same shape as ``a``
        and ``b``, where element-wise multiplication respects the
        zeros-as-ones convention described above.
    """
    c = a.minimum(b)
    r, c = c.nonzero()

    data = np.ones(len(r))
    ones = sparse.csr_matrix((data, (r, c)), shape=a.shape)

    # get common elements
    ones_a = ones.multiply(a)
    ones_b = ones.multiply(b)

    a_dif = a - ones_a
    b_dif = b - ones_b

    result = ones_a.multiply(ones_b)
    return result + a_dif + b_dif


if __name__ == "__main__":

    a = sparse.csr_matrix((5, 5))
    b = sparse.csr_matrix((5, 5))
    c = sparse.csr_matrix((5, 5))

    a[2, 1] = 0.2
    a[1, 2] = 0.2

    b[3, 4] = 0.3
    b[4, 3] = 0.3

    c[2, 1] = 0.2
    c[1, 2] = 0.2

    N = 5
    prob_no_contact = sparse.csr_matrix((N, N))  # empty values = 1.0

    for prob in [a, b, c]:
        A = prob
        if len(A.data) == 0:
            continue
        not_a = A  # empty values = 1.0
        not_a.data = 1.0 - not_a.data
        # print(prob_no_contact.todense())
        # print(not_a.todense())
        prob_no_contact = multiply_zeros_as_ones(prob_no_contact, not_a)
        # print(prob_no_contact.todense())
        # print()

    # probability of contact (whatever layer)
    prob_of_contact = prob_no_contact
    prob_of_contact.data = 1.0 - prob_no_contact.data
#    print(prob_of_contact.todense())

    a = np.zeros((5, 5))
    b = np.zeros((5, 5))
    c = np.zeros((5, 5))

    a[2, 1] = 0.2
    a[1, 2] = 0.2

    b[3, 4] = 0.3
    b[4, 3] = 0.3

    c[2, 1] = 0.2
    c[1, 2] = 0.2

#    print()
#    print()

    prob_no_contact = np.ones((N, N))

    for prob in [a, b, c]:
        A = prob
        not_a = 1.0 - A
        # print(prob_no_contact)
        # print(not_a)
        prob_no_contact = prob_no_contact * not_a
        # print(prob_no_contact)
        # print()

    # probability of contact (whatever layer)
    prob_of_contact = 1.0 - prob_no_contact
#    print(prob_of_contact)
