import numpy as np

from fgw.utils import sym


def shortest_path(A, method="auto"):
    """ Calculate the shortest path distance matrix based on A.

    Args:
        A (np.ndarray): Adjacency matrix with dim (n, n).
        method (str, optional): Algorithm to use for shortest paths. ["auto" | "FW" | "D"].

    Returns:
        (np.ndarray): The commute time.
    """
    from scipy.sparse.csgraph import shortest_path
    assert isinstance(A, np.ndarray), "Input needs to be in (np.ndarray) type"
    C = shortest_path(A, directed=False)
    return C


def commute_time(A):
    """ Update commute time given the adjacency matrix.

    .. math::
        C_ij = vol(G) <e_i - e_j, L^\\dag (e_i - e_j)>

    Args:
        A (np.ndarray): Adjacency matrix with dim (n, n).

    Returns:
        (np.ndarray): The commute time.

    References:
        * Von Luxburg, Ulrike, Agnes Radl, and Matthias Hein.
        "Hitting and commute times in large random neighborhood graphs."
        The Journal of Machine Learning Research 15.1 (2014): 1751-1798.
    """
    # TODO: more efficient implementation
    assert isinstance(A, np.ndarray), "Input needs to be in (np.ndarray) type"
    N = A.shape[0]
    _D = np.diag(A.sum(1))
    _volG = A.sum()
    _L = _D - A
    _L_pinv = np.linalg.pinv(_L, hermitian=True)
    C = np.zeros((N, N))
    E = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            e_ij = E[:, i] - E[:, j]
            C[i, j] = np.inner(e_ij, np.array(_L_pinv) @ e_ij.reshape((-1, 1)).squeeze())
    C += C.T
    C *= _volG
    # C = A.sum() * cttil(A)
    # print(C)
    return C


def commute_time_v2(A):
    """ Update commute time given the adjacency matrix.

    .. math::
        C = vol(G) * (T \\circ (11^\top - I) )

    Args:
        A (np.ndarray): Adjacency matrix with dim (n, n).

    Returns:
        (np.ndarray): The commute time.
    """
    assert isinstance(A, np.ndarray), "Input needs to be in (np.ndarray) type"
    N = A.shape[0]
    D = np.diag(A.sum(0))
    L = D - A
    eet = np.ones((N, N))
    Z = np.linalg.inv(L + eet / N)
    #
    # np.testing.assert_array_almost_equal(np.linalg.inv(Z), L - eet / N)

    vol = A.sum()
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            T[i, j] = Z[i, i] + Z[j, j] - 2 * Z[i, j] if i != j else Z[i, j]

    C = vol * (T * (eet - np.eye(N)))
    return C


def commute_time_approx(A):
    """ Update approx commute time given the adjacency matrix according to
    proposition 1 in [1].

    .. math::
        C = vol(G) <(e_i / sqrt(deg_i) - e_j / sqrt(deg_j)), L_sym^\\inv (e_j / sqrt(deg_j) - e_i / sqrt(deg_i))>

    Args:
        A (sp.matrix): Adjacency matrix with dim (n, n).

    Returns:
        (np.ndarray): The approximate commute time.

    References:
        [1] Von Luxburg, Ulrike, Agnes Radl, and Matthias Hein.
        "Hitting and commute times in large random neighborhood graphs."
        The Journal of Machine Learning Research 15.1 (2014): 1751-1798.

    See Also:
        `commute_time`
    """
    n = A.shape[0]
    deg = A.sum(1).A1
    deg_sqrt = np.power(deg, 0.5)
    _D_sqrt = np.diag(np.power(deg, -0.5))
    _D = np.diag(A.sum(1))
    _volG = A.sum()
    # _L = _D - A
    _L_sym = np.eye(n) - _D_sqrt @ A @ _D_sqrt
    _L_pinv = np.linalg.pinv(_L_sym, hermitian=True)
    C = np.zeros((n, n))
    E = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            e_ij = E[:, i] / deg_sqrt[i] - E[:, j] / deg_sqrt[j]
            # e_ij = e_ij.reshape((-1, 1))
            C[i, j] = np.inner(e_ij, _L_pinv @ -e_ij)
    C += C.T
    C *= _volG
    return C


def cttil(A):
    """ Calculate simplified commute time given the adjacency matrix.

    .. math::
        C_ij = <e_i - e_j, L^\\dag (e_i - e_j)>

    Args:
        A (np.ndarray): Adjacency matrix with dim (n, n).

    Returns:
        (np.ndarray): The commute time.

    References:
        * Von Luxburg, Ulrike, Agnes Radl, and Matthias Hein.
        "Hitting and commute times in large random neighborhood graphs."
        The Journal of Machine Learning Research 15.1 (2014): 1751-1798.

    See Also:
        `commute_time`
    """
    assert isinstance(A, np.ndarray), "Input needs to be in (np.ndarray) type"
    N = A.shape[0]
    _D = np.diag(A.sum(1))
    _L = _D - A
    Z = np.linalg.pinv(_L + np.ones((N, N)) / N)
    # NOTE: force to symmetrify in case the pinv not return a symmetric matrix.
    Z = sym(Z)
    C = cttil_Z(Z)
    return C


def calc_T(Z):
    """ Calculate the T matrix from Z.

    Args:
        Z (np.ndarray): Z matrix defined in `cttil_Z`.

    Returns:
        np.ndarray: Z matrix.

    See Also:
        `cttil_Z`
    """
    N = Z.shape[0]
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2
    return T


def cttil_Z(Z):
    """ Simplified commute time by removing the vol(G).

    .. math::
        C = T \\circ (11^\top - I)
        where T_ij = Z_ii + Z_jj - 2 Z_ij   if i != j
                   = Z_ii                   if i = j

    Args:
        Z (np.ndarray): Inverse of augmented Laplacian

    Returns:
        np.ndarray: Simplified commute time
    """
    assert isinstance(Z, np.ndarray)
    # verify the symmetricity
    np.testing.assert_array_almost_equal(Z, Z.T)
    N = Z.shape[0]
    T = calc_T(Z)
    C = T * (np.ones((N, N)) - np.eye(N))
    return C


def diffusion_distance(A, t=1):
    """ TODO: add diffusion distance.

    Ref
    [1]: https://www.math.pku.edu.cn/teachers/yaoy/Fall2011/lecture10.pdf
    """
    n = A.shape[0]
    d = A.sum(1).astype(float)
    D = np.diag(d)
    D_inv = np.diag(np.power(d, -1))
    P = D_inv @ A
    P = np.linalg.matrix_power(P, t)

    C = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = sum((P[i, k] - P[j, k])**2 / d[k] for k in range(n))
    C += C.T
    return C
