""" Implementation of gwtil and its associated utilities. """
import numpy as np
from scipy.linalg import svd

from fgw.gromov_prox import linear_solver, projection_matrix, quad_solver
from fgw.spg import SPG, default_options
from fgw.utils import padding, squarify


def Qcal_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of Q function, by jointly optimize linear and quadratic term in trace.

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q matrix is not semi-orthogonal.
    """
    m, n = C.shape[0], D.shape[0]

    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()
    q_dim = max(m, n) - 1

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding(C_hat, D_hat)

    E_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
    _E_hat = squarify(E_hat)

    def obj_func(Q):
        # obj = tr(_C_hat Q _D_hat Q.T) + tr(E_hat Q.T)
        Q = Q.reshape((q_dim, -1))
        fval = 0
        quad_ = np.trace(_C_hat @ Q @ _D_hat @ Q.T)
        lin_ = np.trace(_E_hat @ Q.T)

        fval = -quad_ - lin_

        g_quad_ = 2 * _C_hat @ Q @ _D_hat
        g_lin_ = _E_hat
        grad = -g_quad_ - g_lin_
        return fval, grad.flatten()

    def proj_func(Q):
        Q = Q.reshape((q_dim, -1))
        # DEBUG: SVD did not converge in the dataset IMDB-BINARY
        try:
            u, s, vh = svd(Q)
        except BaseException:
            u, s, vh = svd(Q + 1e-7)
        return (u @ vh).flatten()

    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.testOpt = False
    spg_options.verbose = 0 if "verbose" not in kwargs else kwargs['verbose']

    # NOTE: init of Q matters
    if "Q_init" not in kwargs:
        # fix randomness with init
        np.random.seed(0)
        Q_init = np.random.randn(q_dim, q_dim).flatten()
    elif isinstance(kwargs.get("Q_init"), str):
        # is a string
        if kwargs.get("Q_init") == "identity":
            # NOTE: could stuck at local min
            Q_init = np.identity(q_dim).flatten()
        elif kwargs.get("Q_init") == "zeros":
            # NOTE: could stuck at local min.
            Q_init = np.zeros((q_dim, q_dim))
    else:
        # is a np.ndarray
        Q_init = kwargs['Q_init']

    res = SPG(obj_func, proj_func, Q_init, options=spg_options)
    q_opt = res[0].reshape((q_dim, - 1))
    q_fval = - res[1]

    Q = q_opt[:m - 1, :n - 1]

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_array_almost_equal(np.identity(m - 1), Q @ Q.T)
    else:
        np.testing.assert_array_almost_equal(np.identity(n - 1), Q.T @ Q)

    P = 1 / mn_sqrt * em @ en.T + U @ Q @ V.T

    fval = sC * sD / mn + q_fval
    if return_matrix:
        return fval, P
    else:
        return fval


def Qcal_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of Q function, by optimizing Q1, Q2 separately.

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

        \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top) + \\max_{Q_2} tr(E_hat Q_2^\top)
        s.t. Q_1, Q_2 \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
        np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
    """
    # assert C.shape[0] >= D.shape[0]
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    eet = np.ones((m, n))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding(C_hat, D_hat)

    E_hat = 2 / mn_sqrt * U.T @ C @ eet @ D @ V
    _E_hat = squarify(E_hat)
    q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
    l_fval, Q2 = linear_solver(_E_hat, domain="O", return_matrix=True)

    if "verbose" in kwargs:
        print("from gwtil_lb", q_fval, l_fval)

    fval = sC * sD / mn + q_fval + l_fval
    Q1, Q2 = Q1[:m - 1, :n - 1], Q2[:m - 1, :n - 1]

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
        np.testing.assert_almost_equal(np.identity(m - 1), Q2 @ Q2.T)
    else:
        np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)
        np.testing.assert_almost_equal(np.identity(n - 1), Q2.T @ Q2)

    if return_matrix:
        return fval, Q1, Q2
    else:
        return fval


# def Qcal_ub_(C, D, return_matrix=False, **kwargs):
#     """ Upper bound of Q function, by optimizing Q1, Q2 separately.

#     Note: This version does not apply `linear_solver` to solve the padded version E_hat.

#     .. math::
#         \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

#         Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

#         \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top) + \\max_{Q_2} tr(E_hat Q_2^\top)
#         s.t. Q_1, Q_2 \\in \\Ocal_{(m-1) \times (n-1)}

#     Args:
#         C (np.ndarray): Geodesic distance in source domain.
#         D (np.ndarray): Geodesic distance in target domain.
#         return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

#     Returns:
#         float: objective value
#         np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
#         np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

#     Raises:
#         AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
#     """
#     # assert C.shape[0] >= D.shape[0]
#     m = C.shape[0]
#     n = D.shape[0]
#     mn = m * n
#     mn_sqrt = np.sqrt(mn)
#     em = np.ones((m, 1))
#     en = np.ones((n, 1))
#     eet = np.ones((m, n))
#     sC = C.sum()
#     sD = D.sum()

#     U = projection_matrix(m)
#     V = projection_matrix(n)

#     C_hat = U.T @ C @ U
#     D_hat = V.T @ D @ V
#     _C_hat, _D_hat = padding(C_hat, D_hat)

#     q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
#     E_hat = 2 / mn_sqrt * U.T @ C @ eet @ D @ V

#     # Solve Q2 with top s singular vec on both left and right
#     u_mat, sigma, vh_mat = svd(E_hat)
#     l_fval = sigma.sum()
#     s = min(m, n) - 1
#     Q2 = u_mat[:, :s] @ vh_mat[:s, :]

#     # DEBUG: remove the squarify, use truncation method
#     # _E_hat = squarify(E_hat)
#     # l_fval, Q2 = linear_solver(_E_hat, domain="O", return_matrix=True)

#     if "verbose" in kwargs:
#         print("from gwtil_lb", q_fval, l_fval)

#     fval = sC * sD / mn + q_fval + l_fval
#     Q1, Q2 = Q1[:m - 1, :n - 1], Q2[:m - 1, :n - 1]

#     # NOTE: semi-orthogonal defined on the smaller dimension
#     if m < n:
#         np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
#         np.testing.assert_almost_equal(np.identity(m - 1), Q2 @ Q2.T)
#     else:
#         np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)
#         np.testing.assert_almost_equal(np.identity(n - 1), Q2.T @ Q2)

#     if return_matrix:
#         return fval, Q1, Q2
#     else:
#         return fval


def Qcal_ub_v2(C, D, return_matrix=False, **kwargs):
    """ Upper bound of Q function, by optimizing Q1, Q2, Q3 separately.

    NOTE: This is just to split the linear term into separate ones, in order to
    achieve the optimal Q2, Q3 from svd.

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

        \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top) + \\max_{Q_2} tr(E_hat Q_2^\top)
        s.t. Q_1, Q_2 \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
        np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
    """
    # assert C.shape[0] >= D.shape[0]
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    eet = np.ones((m, n))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding(C_hat, D_hat)

    E_hat = 1 / mn_sqrt * U.T @ C @ eet @ D @ V
    _E_hat_q2 = squarify(E_hat)
    _E_hat_q3 = squarify(E_hat.T)

    q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
    l_fval_q2, Q2 = linear_solver(_E_hat_q2, domain="O", return_matrix=True)
    l_fval_q3, Q3 = linear_solver(_E_hat_q3, domain="O", return_matrix=True)

    l_fval = l_fval_q2 + l_fval_q3
    if "verbose" in kwargs:
        print("from gwtil_lb", q_fval, l_fval)

    fval = sC * sD / mn + q_fval + l_fval
    Q1, Q2, Q3 = Q1[:m - 1, :n - 1], Q2[:m - 1, :n - 1], Q3[:m - 1, :n - 1]

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
        np.testing.assert_almost_equal(np.identity(m - 1), Q2 @ Q2.T)
    else:
        np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)
        np.testing.assert_almost_equal(np.identity(n - 1), Q2.T @ Q2)

    if return_matrix:
        return fval, Q1, Q2, Q3
    else:
        return fval


def Qcal_ub_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of Qcal_ub function, by optimizing Q1 only (ignore the linear term in trace).

    .. math::
        \\max_P tr(CPDP^\top)   s.t.  P in \\Ecal \\cap \\Ocal

        Change P = 1/\\sqrt{mn} 11^\top + U Q V^\top  s.t. Q \\in \\Ocal_{(m-1) \times (n-1)}

        \\max_{Q_1} tr(C_hat Q_1 D_hat Q_1^\top)
        s.t. Q_1 \\in \\Ocal_{(m-1) \times (n-1)}

    Args:
        C (np.ndarray): Geodesic distance in source domain.
        D (np.ndarray): Geodesic distance in target domain.
        return_matrix (bool, optional): Return the Q matix if `True`. Defaults to False.

    Returns:
        float: objective value
        np.ndarray, optional: Q1 matrix in semi-orthogonal domain.
        np.ndarray, optional: Q2 matrix in semi-orthogonal domain.

    Raises:
        AssertionError: If Q1 and Q2 matrices are not semi-orthogonal.
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)

    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    _C_hat, _D_hat = padding(C_hat, D_hat)

    # NOTE: remove Q2
    # E_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
    # _E_hat = squarify(E_hat)
    q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
    # l_fval, Q2 = linear_solver(_E_hat, domain="O", return_matrix=True)

    if kwargs.get("verbose"):
        print("from Qcal_ub_ub", q_fval)

    fval = sC * sD / mn + q_fval
    Q1 = Q1[:m - 1, :n - 1]

    # Recover the P matrix from Q1. This is similar to gwtil_lb.
    P = 1 / mn_sqrt * em @ en.T + U @ Q1 @ V.T

    # NOTE: semi-orthogonal defined on the smaller dimension
    if m < n:
        np.testing.assert_almost_equal(np.identity(m - 1), Q1 @ Q1.T)
    else:
        np.testing.assert_almost_equal(np.identity(n - 1), Q1.T @ Q1)

    if return_matrix:
        return fval, P
    else:
        return fval


def gwtil_ub(C, D, return_matrix=False, **kwargs):
    """ Upper bound of gwtil by soling `Qcal_lb`.

    .. math: :
        gwtil = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * \\max_{P} \\Qcal(P)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: gwtil distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_lb`
    """
    m, n = C.shape[0], D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)
    eet = np.ones((m, n))

    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)
    U = projection_matrix(m)
    V = projection_matrix(n)

    # recall Q_init from P_init if no Q_init is specified
    if kwargs.get("P_init") is not None:
        kwargs['Q_init'] = squarify(U.T @ (kwargs['P_init'] - 1 / mn_sqrt * eet) @ V)

    quad_val, P = Qcal_lb(C, D, return_matrix=True, **kwargs)

    fval = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 * quad_val / mn

    if return_matrix:
        return fval, P
    else:
        return fval


def gwtil_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of gwtil by solving `Qcal_ub`.

    .. math: :
        gwtil = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * \\max_{Q1, Q2} \\Qcal(Q1, Q2)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: gwtil distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n

    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)

    qfunc_val, Q1, Q2 = Qcal_ub(C, D, return_matrix=True, **kwargs)
    gwtil_val = C_norm ** 2 / m**2 + D_norm**2 / n**2 - 2 / mn * qfunc_val
    if return_matrix:
        return gwtil_val, Q1, Q2
    else:
        return gwtil_val


def gwtil_o(C, D, return_matrix=False, **kwargs):
    """ Direct Optimize over orthogonal domain.

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix (bool, optional): _description_. Defaults to False.

    Returns:
        float: gwtil distance
        np.ndarray, optional: P matrix

    See also:
        `gwtil_lb`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)

    C_hat, D_hat = padding(C, D)
    qfunc_val, Q = quad_solver(C_hat, D_hat, domain="O", return_matrix=True)
    Q = Q[:m, :n]
    gwtil_val = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 / mn * qfunc_val
    if return_matrix:
        return gwtil_val, Q
    else:
        return gwtil_val
        

def gwtil_lb_lb(C, D, return_matrix=False, **kwargs):
    """ Lower bound of gwtil by solving `Qcal_ub_ub`.

    .. math: :
        gwtil = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * \\max_{Q1, Q2} \\Qcal(Q1, Q2)

    Args:
        C(np.ndarray): Geodesic distance in source domain with dim(m, m).
        D(np.ndarray): Geodesic distance in target domain with dim(n, n).
        return_matrix(bool, optional): Return P matrix if `True`. Defaults to False.

    Returns:
        float: gwtil distance
        np.ndarray, optional: P matrix

    See also:
        `Qcal_ub_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n

    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)

    qfunc_val, P = Qcal_ub_ub(C, D, return_matrix=True, **kwargs)
    gwtil_val = C_norm ** 2 / m**2 + D_norm**2 / n**2 - 2 / mn * qfunc_val
    if return_matrix:
        return gwtil_val, P
    else:
        return gwtil_val


def eval_gwtil_ub(C, D, P):
    """ Evaluate the gwtil given C, D, and optimzed P.

    .. math::
        gwtil = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * tr (C P D P^\top)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        P (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        float: gwtil function value
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)
    gwtil_fval = C_norm**2 / m**2 + D_norm**2 / n**2 - 2 / mn * np.trace(C @ P @ D @ P.T)
    return gwtil_fval


def eval_gwtil_lb(C, D, Q1, Q2):
    """ Evaluate the gwtil given C, D, and optimzed P.

    .. math::
        gwtil = ||C||_F^2 / m^2 + ||D||_F^2 / n^2 - 2 / mn * Qcal(Q1, Q2)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        Q1 (np.ndarray): Optimized Q1 matrix with dim (m-1, n-1).
        Q2 (np.ndarray): Optimized Q2 matrix with dim (m-1, n-1).

    Returns:
        float: gwtil function value

    See Also:
        `Qcal_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = mn ** 0.5
    em = np.ones((m, 1))
    en = np.ones((n, 1))
    sC = C.sum()
    sD = D.sum()

    U = projection_matrix(m)
    V = projection_matrix(n)
    C_norm = np.linalg.norm(C)
    D_norm = np.linalg.norm(D)

    _const = 1 / mn * sC * sD
    _linear = np.trace(V.T @ D @ en @ em.T @ C @ U @ Q2)
    _quad = np.trace(U.T @ C @ U @ Q1 @ V.T @ D @ V @ Q1.T)
    gwtil_fval = C_norm ** 2 / m**2 + D_norm ** 2 / n**2 \
        - 2 / mn * (_const + 2 / mn_sqrt * _linear + _quad)

    return gwtil_fval
