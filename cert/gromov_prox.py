import numpy as np
from numpy.linalg import eigh
from ot.gromov import fused_gromov_wasserstein, gromov_wasserstein
from scipy.linalg import svd
# from fgw.svd import svd
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from scipy.spatial.distance import cdist

from fgw.spg import SPG, default_options
from fgw.utils import padding, squarify, sym


def projection_matrix(n, seed=0):
    """ Set up the projection matrix V with size (n, n-1).

    Args:
        n (int): Dimension.
        seed (int): Seed for random generator.

    Returns:
        (np.ndarray): projection matrix V such that
            $V^\top u = 0, V^\top V = I_{n-1}$

    Reference:
        [1] Hadley, Scott W., Franz Rendl, and Henry Wolkowicz.
        "A new lower bound via projection for the quadratic assignment problem."
        Mathematics of Operations Research 17.3 (1992): 727-739.
    """
    _x = -1 / (n + np.sqrt(n))
    _y = -1 / np.sqrt(n)
    V = np.zeros((n, n - 1))
    V[0, :] = _y * np.ones((1, n - 1))
    V[1:n, :] = _x * np.ones((n - 1, n - 1)) + np.eye(n - 1)
    return V


def eig_decom(A, descending=False):
    """ Eigen decomposition.

    Args:
        A (np.ndarray): Matrix.
        descending (bool, optional): descending the eigvals if `True`.

    Returns:
        (np.array): Array of eigenvalues.
        (np.ndarray): Matrix, each column corresponding to eigenvalues.
    """
    evals, evecs = eigh(A)
    if not descending:
        idx = evals.argsort()
    else:
        idx = evals.argsort()[::-1]
    evals = evals[idx]
    evec_P = evecs[idx]
    return evals, evec_P


def linear_solver(C, domain="O", maximize=True, return_matrix=False):
    """ Solve the linear term.

    .. math::
        \\max_{X: X \\in O} \tr (C X^\top)

    Note:
        The optimal solution Q may not be unique due to its svd decomposition
        and potential ill-conditioned matrix.

    Args:
        C (np.ndarray): Matrix in linear term.
        domain (str, optional): Domain on quadratic term with following options:
            "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
            "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
            "P": permutation domain, i.e., `O \\cap E \\cap N`.
            Default: "O"
        maximize (bool, optional): Set the problem as maximize.
            Default: True.
        return_matrix (bool, optional): Return the closed form matrix.
            Default: False.

    Returns:
        (float):  optimal value from the linear term.
        (np.ndarray): Optimal matrix if `return_matrix` is True.

    Reference:
        * Math proof: https://math.stackexchange.com/a/754476/60983
    """
    # ensure C in a square matrix
    assert C.shape[0] == C.shape[1]
    if domain in ["OE", "O"]:
        # orthonormal and / or row-column sum
        u, s, vh = svd(C)
        f_val = s.sum() if maximize else -s.sum()
        if return_matrix:
            X = u @ vh
            return f_val, X
        else:
            return f_val

    elif domain in ["P"]:
        # permutation domain, Hungarian algorithm
        r_idx, c_idx = linear_sum_assignment(C, maximize=maximize)
        return C[r_idx, c_idx].sum()


def linear_grad(C, domain="O"):
    """ Partial derivative of linear term w.r.t. C.

    .. math::
        $trace (C P^\top)$
        $\\partial trace(CP^\top) / \\partial C = P$

    Args:
        C: (np.ndarray): Matrix in linear term.
        domain: (str, optional): specify the domain.

    Returns:
        (np.ndarray): gradient with respect to linear matrix.
    """
    l_fval, P = linear_solver(C, domain=domain, return_matrix=True)
    return P


def quad_grad(A, B, domain="O"):
    """ Partial derivative of quadratic term w.r.t. A.

    .. math::
                $f(A, B) = trace (A P B P^\top)$
        * domain = "O":
            $\\partial f / \\partial A = P B P^\top$
        * domain = "OE":
            $\\partial f / \\partial A = V @ Q @ \\hat{B} @ Q @ V.T$
    Args:
        A (np.ndarray): Distance matrix from source metric space.
        B (np.ndarray): Distance matrix from target metric space.
        domain (str, optional): Domain on quadratic term.

    Returns:
        (np.ndarray):
            Return gradient wrt to A if domain = "O".
            Return (grad_Q, grad_L) gradient wrt to A if domain = "OE".
    """
    n = A.shape[0]
    if domain == "O":
        # get the optimal P
        q_fval, P = quad_solver(A, B, domain=domain, return_matrix=True)
        return P @ B.T @ P.T

    elif domain == "OE":
        # replace P with 1/n ee.T + VQV.T, and split the quad into two parts.
        V = projection_matrix(n)
        e = np.ones((n, 1))

        A_hat = V.T @ A @ V
        B_hat = V.T @ B @ V
        q_grad = V @ quad_grad(A_hat, B_hat, domain="O") @ V.T

        E_hat = 2 / n * V.T @ A @ e @ e.T @ B @ V
        l_grad = linear_grad(E_hat, domain="O")
        return q_grad, l_grad


def quad_solver(A, B, domain="O", maximize=True, return_matrix=False):
    """ Solve the quadratic term in GW under constraints.

    .. math::
        \\max_{X: X \\in S} \tr (AXBX^\top)

    Args:
        A (np.ndarray): Distance matrix from source metric space.
        B (np.ndarray): Distance matrix from target metric space.
        domain (str, optional): Domain on quadratic term with following options:
            "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
            "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
            "P": permutation domain, i.e., `O \\cap E \\cap N`.
            Default: "O"
        maximize (bool, optional): Set the problem as maximize.
            Default: True.
        return_matrix (bool, optional): Return the closed form matrix.
            Default: False.

    Returns:
        (float): optimal value from the quadratic term.
        (np.ndarray): optimal matrix (matrices) if `return_matrix` is True.
            Return P if domain = "O",
            Return Q1 and Q2 if domain = "OE".
    """
    if domain not in ["O", "OE", "P"]:
        print("Please specify the domain in ['O', 'OE', 'P']")
        exit()
    n = A.shape[0]
    A, B = sym(A), sym(B)
    # ensure A and B are in same size
    assert A.shape[0] == B.shape[0]
    if domain == "O":
        evals_A, evecs_A = eig_decom(A)
        evals_B, evecs_B = eig_decom(B)

        if maximize:
            f_val = np.multiply(evals_A, evals_B).sum()
            P = evecs_A @ evecs_B.T
        else:
            f_val = np.multiply(evals_A[::-1], evals_B).sum()
            P = -evecs_A @ evecs_B.T

        if return_matrix:
            return f_val, P
        else:
            return f_val

    elif domain == "OE":

        # set up the projection matrix V with size (n, n-1)
        V = projection_matrix(n)

        one = np.ones((n, 1))
        A_hat = V.T @ A @ V
        B_hat = V.T @ B @ V

        # handle quadratic term
        if return_matrix:
            q_fval, Q1 = quad_solver(A_hat, B_hat, domain="O", return_matrix=True, maximize=maximize)
        else:
            q_fval = quad_solver(A_hat, B_hat, domain="O", maximize=maximize)

        # handle linear term
        C = V.T @ A @ one @ one.T @ B @ V

        if return_matrix:
            l_fval, Q2 = linear_solver(C, domain=domain, maximize=maximize, return_matrix=True)
        else:
            l_fval = linear_solver(C, domain=domain, maximize=maximize)

        sA = np.sum(A)
        sB = np.sum(B)

        f_val = q_fval + 2 / n * l_fval + 1 / n**2 * sA * sB
        if return_matrix:
            return f_val, Q1, Q2
        else:
            return f_val

    elif domain == "P":
        # GW on permutation domain
        res = quadratic_assignment(A, B)
        if return_matrix:
            perm = res['col_ind']
            P = np.eye(n, dtype=int)[perm]
            return res['fun'], P
        else:
            return res['fun']


def fused_quad_solver(A, B, M, alpha=0.5, maximize=True, domain="O", return_matrix=False):
    """ Solve the quadratic term in FGW under constraints.

    Args:
        A (np.ndarray): Distance matrix from source metric space.
        B (np.ndarray): Distance matrix from target metric space.
        M (np.ndarray): Feature distance matrix.
        domain (str, optional): Domain on quadratic term with following options:
            "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
            "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
            "P": permutation domain, i.e., `O \\cap E \\cap N`.
            Default: "O"
        maximize (bool, optional): Set the problem as maximize.
            Default: True.
        return_matrix (bool, optional): Return optimal matrices.

    Returns:
        (float): optimal value from the quadratic term.
    """
    if domain not in ["O", "OE", "P"]:
        print("Please specify the domain in ['O', 'OE', 'P']")
        exit()
    n = A.shape[0]
    # ensure A and B are in same size
    assert A.shape[0] == B.shape[0]
    n = A.shape[0]

    if domain == "O":
        q_fval, Q1 = quad_solver(A, B, domain="O", return_matrix=True)
        l_fval, Q2 = linear_solver(M, domain="O", return_matrix=True)

        total_fval = 2 * alpha * q_fval - (1 - alpha) * l_fval
        if return_matrix:
            return total_fval, Q1, Q2
        else:
            return total_fval

        # return 2 * alpha * quad_solver(A, B, domain="O") - (1 - alpha) * linear_solver(M, domain="O")

    elif domain == "OE":

        # set up the projection matrix V with size (n, n-1)
        V = projection_matrix(n)

        one = np.ones((n, 1))
        A_hat = V.T @ A @ V
        B_hat = V.T @ B @ V

        # handle quadratic term
        q_fval = quad_solver(A_hat, B_hat, domain="O")

        # REVIEW: additional terms for linear
        C1 = V.T @ A @ one @ one.T @ B @ V
        C2 = V.T @ M @ V
        C = 4 * alpha / n * C1 - (1 - alpha) * C2
        l_fval = linear_solver(C, domain=domain, maximize=maximize)

        sA = np.sum(A)
        sB = np.sum(B)
        sM = np.sum(M)
        return 2 * alpha * q_fval + l_fval + 2 * alpha / n**2 * sA * sB - (1 - alpha) * sM / n

    elif domain == "P":
        """ Approximate solve the problem in permutation domain """
        q_fval = quad_solver(A, B, domain="P")
        l_fval = linear_solver(M, domain="P")


@PendingDeprecationWarning
def fused_quad_solver_rec(C, D, M, alpha=1, maximize=True, domain="OE", return_matrix=False):
    # DEBUG: correct the scalar
    """
    .. math::

    tr(C P D P.T)
    = 1/mn sC * sD + 2/\\sqrt(mn) tr [(V.T D 11.T C U) @ Q] + tr [(U.T C U) Q (V.T D V) Q.T]

    P = 1/\\sqrt(mn) 11.T + U Q V.T

    let E:= V.T D 11.T C U

    """
    if domain not in ["O", "OE", "P"]:
        print("Please specify the domain in ['O', 'OE', 'P']")
        exit()
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(m * n)

    if domain == "OE":
        # set up the projection matrix U with size (m, m-1)
        U = projection_matrix(m)

        # set up the projection matrix V with size (n, n-1)
        V = projection_matrix(n)

        em = np.ones((m, 1))
        en = np.ones((n, 1))

        C_hat = U.T @ C @ U
        D_hat = V.T @ D @ V

        _C_hat, _D_hat = padding(C_hat, D_hat)

        # REVIEW: tr (C_hat Q D_hat Q.T)

        # handle quadratic term
        if return_matrix:
            q_fval, Q1 = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=True)
            Q1 = Q1[:m - 1, :n - 1]
        else:
            q_fval = quad_solver(_C_hat, _D_hat, domain="O", return_matrix=False)

        # F1 = C @ em @ en.T @ D
        # F2 = D @ en @ em.T @ C
        # F_hat = alpha * ((U.T @ F1 @ V) + (V.T @ F2 @ U).T) / np.sqrt(m * n) - (1 - alpha) / 2 * (V.T @ M.T @ U).T

        # REVIEW:
        # F_hat = 2 * alpha * (V.T @ D @ en @ em.T @ C @ U) / mn_sqrt - (1 - alpha) / 2 * (V.T @ M.T @ U)
        # F_hat = F_hat.T
        # F_hat = squarify(F_hat)
        # F_hat = 2 * alpha / mn_sqrt * (U.T @ C @ em @ en.T @ D @ V) - (1 - alpha) / 2 * (U.T @ M @ V)
        # DEBUG:
        #       2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
        #       is different from
        #       2 / mn_sqrt * (U.T @ C @ em @ en.T @ D @ V)
        F_hat = 2 / mn_sqrt * U.T @ C @ em @ en.T @ D @ V
        F_hat = squarify(F_hat)
        # print(2 * alpha * (V.T @ D @ en @ em.T @ C @ U) / mn_sqrt)
        # print(squarify_v2(2 * alpha * (V.T @ D @ en @ em.T @ C @ U) / mn_sqrt))
        # print(F_hat)
        # import pickle
        # E_hat = pickle.load(open("Ehat.pk", "rb"))

        # pickle.dump({"E_hat": E_hat, "F_hat": F_hat}, open("linear_trace_term.pkl", "wb"))
        if return_matrix:
            l_fval, Q2 = linear_solver(F_hat, domain="O", maximize=True, return_matrix=True)
            Q2 = Q2[:m - 1, :n - 1]
        else:
            l_fval = linear_solver(F_hat, domain="O", maximize=True, return_matrix=False)

        # print("from fused_quad_solver", q_fval, l_fval)
        # P = np.zeros([m, n])

        sC = np.sum(C)
        sD = np.sum(D)
        sM = np.sum(M)

        # DEBUG: objective fval
        # fval = alpha * (sC / m**2 + sD / n**2) - 2 * alpha * q_fval / m / n + (1 - alpha) * l_fval / mn_sqrt
        # fval = alpha * (sC / m**2 + sD / n**2) - 2 * alpha * (1 / mn * sC * sD + l_fval + q_fval) / mn
        fval = alpha * sC * sD / mn + l_fval + alpha * q_fval
        if return_matrix:
            return fval, Q1, Q2
        else:
            # DEBUG
            # return alpha * q_fval + l_fval + alpha * sC * sD / (m * n) - (1 - alpha) / 2 * sM / np.sqrt(m * n)
            return fval
    else:
        raise("Not Implemented")


@PendingDeprecationWarning
def fused_quad_solver2(M, A, B, gamma=0.5, maximize=True, domain="O"):
    """ Solve the quadratic term in FGW under constraints.

    Notes:
        This is the version using scalar gamma on feature matrix.

    Args:
        M (np.ndarray): Feature distance matrix.
        A (np.ndarray): Distance matrix from source metric space.
        B (np.ndarray): Distance matrix from target metric space.
        domain (str, optional): Domain on quadratic term with following options:
            "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
            "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
            "P": permutation domain, i.e., `O \\cap E \\cap N`.
            Default: "O"
        maximize (bool, optional): Set the problem as maximize.
            Default: True.

    Returns:
        (float): optimal value from the quadratic term.
    """
    if domain not in ["O", "OE", "P"]:
        print("Please specify the domain in ['O', 'OE', 'P']")
        exit()
    n = A.shape[0]
    # ensure A and B are in same size
    assert A.shape[0] == B.shape[0]
    n = A.shape[0]
    if domain == "O":
        # solve the quad term and linear term separately
        # eqref: eq (27)
        return quad_solver(A, B, domain="O") - linear_solver(M, domain="O")

    elif domain == "OE":
        # eqref: eq (28)

        # set up the projection matrix V with size (n, n-1)
        V = projection_matrix(n)

        one = np.ones((n, 1))
        A_hat = V.T @ A @ V
        B_hat = V.T @ B @ V

        # handle quadratic term
        q_fval = quad_solver(A_hat, B_hat, domain="O")

        # REVIEW: additional terms for linear
        C1 = V.T @ A @ one @ one.T @ B @ V
        C2 = V.T @ M @ V
        C = 2 / n * C1 - C2
        l_fval = linear_solver(C, domain="O", maximize=True)

        sA = np.sum(A)
        sB = np.sum(B)
        sM = np.sum(M)
        return q_fval + l_fval + 1 / n**2 * sA * sB - sM / n

    elif domain == "P":
        """ Approximate solve the problem in permutation domain """
        q_fval = quad_solver(A, B, domain="P")
        l_fval = linear_solver(gamma * M, domain="P")
        return q_fval + l_fval


def quad_solver_lower_bound(C_hat, D_hat, F_hat):
    """ Joint optimize over the quadratic and linear term in eq (8).
    .. math::
        $\\max_{Q \\in \\Ocal} tr (C_hat Q D_hat Q.T) + tr (F_hat Q.T)$

    Args:
        C_hat (np.ndarray): Projected matrix of C with dim (n-1, n-1).
        D_hat (np.ndarray): Projected matrix of D with dim (n-1, n-1).
        F_hat (np.ndarray): Projected matrix of F with dim (n-1, n-1).

    Returns:
        (float): Local optimal function value of Q.
        (np.ndarray): Local optimal Q.
    """
    n = C_hat.shape[0] + 1

    def q_func(Q):
        """ Trace of quadratic term + linear term. Ref eq (8).
        """
        return - np.trace(C_hat @ Q @ D_hat @ Q.T) - np.trace(F_hat @ Q)

    def q_grad(Q):
        """ Gradient w.r.t to Q """
        # Q = Q.reshape((n, -1))
        return -(C_hat @ Q @ D_hat * 2 + F_hat).flatten()

    def obj_func(Q):
        """ Return the function value and its gradient """
        Q = Q.reshape((n - 1, -1))
        return q_func(Q), q_grad(Q)

    def proj_func(Q):
        Q = Q.reshape((n - 1, -1))
        u, s, vh = svd(Q)
        return (u @ vh).flatten()

    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.testOpt = False
    spg_options.verbose = 0
    Q_init = np.identity(n - 1).flatten()
    # Q_init = np.random.rand(n - 1, n - 1).flatten()
    res = SPG(obj_func, proj_func, Q_init, options=spg_options)
    q_opt = res[0].reshape((n - 1, n - 1))
    q_fval = - res[1]
    return q_fval, q_opt


def fgwtil_ub(M, C, D, gamma=0., domain="OE", return_matrix=False):
    """ Calculate the upper bound of FGW.
        We will use SPG to solve the joint term.

    .. math::
        $\\max_{Q} trace (C_hat Q D_hat Q.T) + trace (F_hat Q.T)$
        where   $C_hat = V.T C V$
                $D_hat = V.T D V$
                $F_hat = 2/n V.T C 1 1.T D V - V.T \\gamma M V$

    Args:
        M (np.ndarray):
        C (np.ndarray):
        D (np.ndarray):
        gamma (float, optional):
        domain (str, optional):
        return_matrix (bool, optional):

    Returns
        (float): \bar{Omega}, upper bound of `FGW_{O \\cap E}`
        (np.ndarray): optimal local Q matrix.

    Notes:
        By setting gamma = 0, we recover the upper bound of GW_{O \\cap E}.
    """
    n = C.shape[0]
    assert C.shape[0] == D.shape[0]

    try:
        np.testing.assert_array_equal(C, C.T)
        np.testing.assert_array_equal(D, D.T)
    except BaseException:
        print("Input matrices are not symmetric")

    try:
        assert domain == "OE"
    except BaseException:
        print("The upper bound only apply to the domain OE.")

    def q_func(Q):
        """ Trace of quadratic term + linear term. Ref eq (8).
        """
        return - np.trace(C_hat @ Q @ D_hat @ Q.T) - np.trace(F_hat @ Q)

    def q_grad(Q):
        """ Gradient w.r.t to Q """
        # Q = Q.reshape((n, -1))
        return -(C_hat @ Q @ D_hat * 2 + F_hat).flatten()

    def obj_func(Q):
        """ Return the function value and its gradient """
        Q = Q.reshape((n - 1, -1))
        return q_func(Q), q_grad(Q)

    def proj_func(Q):
        """ Project to the orthonormal domain. """
        Q = Q.reshape((n - 1, -1))
        u, s, vh = svd(Q)
        return (u @ vh).flatten()

    V = projection_matrix(n)
    e = np.ones((n, 1))
    C_hat = V.T @ C @ V
    D_hat = V.T @ D @ V
    E = 2 / n * C @ e @ e.T @ D
    F = E - gamma * M
    # E_hat = V.T @ E @ V
    F_hat = V.T @ F @ V

    # Solve the lower bound of \tilde{Q} func in eq (8)
    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.testOpt = False
    spg_options.verbose = 0
    # REVIEW: Q_init
    Q_init = np.identity(n - 1).flatten()
    # Q_init = np.random.rand(n - 1, n - 1).flatten()
    res = SPG(obj_func, proj_func, Q_init, options=spg_options)
    q_opt = res[0].reshape((n - 1, n - 1))
    q_fval = - res[1]

    # sanity check of orthonormal projection
    try:
        np.testing.assert_almost_equal(q_opt @ q_opt.T, np.identity(n - 1))
    except BaseException:
        print("Not orthonormal")

    sM = M.sum()
    sF = F.sum()
    sC = C.sum()
    sD = D.sum()
    nC = np.linalg.norm(C)**2
    nD = np.linalg.norm(D)**2
    # eq (17)
    omega = (nC + nD - 2 * q_fval - 1 / n * sF + gamma / n * sM) / n**2
    # omega = (nC + nD - 2 * q_fval - 2 / n**2 * C.sum() * D.sum()) / n**2
    if return_matrix:
        return omega, q_opt
    else:
        return omega


def gromov_bound(C1, C2, domain="O", return_matrix=False):
    """ Calculate GW with domain specified.

    Notes:
        \\min_P GW is equivalent to \\max_{P} trace(quadratic term)

    Args:
        C1 (np.ndarray): Distance matrix from source metric space.
        C2 (np.ndarray): Distance matrix from target metric space.
        domain (str, optional):
            Domain on quadratic term with following options:
                "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
                "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
                "NE": positive and column / row sum domain, i.e. `P \\in R_+, P^\top 1 = P 1 = p`
                "P": permutation domain, i.e., `O \\cap E \\cap N`.
                Default: "O"

    Returns:
        (float): Gromov Wasserstein distance under different domains.
        (np.ndarray): optimal matrix (matrices) if `return_matrix` is True.
    """
    n = C1.shape[0]
    assert C1.shape[0] == C2.shape[0]
    if domain in ["O", "P"]:
        # return single matrix
        if return_matrix:
            q_fval, P = quad_solver(C1, C2, domain=domain, return_matrix=True)
            gw_bound = np.linalg.norm(C1) ** 2 + np.linalg.norm(C2)**2 - 2 * q_fval
            gw_bound = gw_bound / n**2
            return gw_bound, P
        else:
            q_fval = quad_solver(C1, C2, domain=domain)
            gw_bound = np.linalg.norm(C1) ** 2 + np.linalg.norm(C2)**2 - 2 * q_fval
            gw_bound = gw_bound / n**2
            return gw_bound

    elif domain == "OE":
        # return decoupled matrices
        if return_matrix:
            q_fval, Q1, Q2 = quad_solver(C1, C2, domain=domain, return_matrix=True)
            gw_bound = np.linalg.norm(C1) ** 2 + np.linalg.norm(C2)**2 - 2 * q_fval
            gw_bound = gw_bound / n**2
            return gw_bound, Q1, Q2
        else:
            q_fval = quad_solver(C1, C2, domain=domain)
            gw_bound = np.linalg.norm(C1) ** 2 + np.linalg.norm(C2)**2 - 2 * q_fval
            gw_bound = gw_bound / n**2
            return gw_bound

    elif domain == "NE":
        # exact gromov wasserstein
        _p = np.array([1. / n] * n)
        T, gw_log = gromov_wasserstein(C1, C2, _p, _p, loss_fun="square_loss", log=True)
        if return_matrix:
            return gw_log['gw_dist'], T
        else:
            return gw_log['gw_dist']


def fused_gromov_bound(M, C1, C2, alpha=0.5, domain="O", return_matrix=False):
    """ Calculate FGW with domain specified.

    Args:
        C1 (np.ndarray): Distance matrix from source metric space.
        C2 (np.ndarray): Distance matrix from target metric space.
        alpha (float, optional): Trade-off between feature metric and topological metrics.
            Default: 0.5.
        domain (str, optional):
            Domain on quadratic term with following options:
                "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
                "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
                "NE": positive and column / row sum domain, i.e. `P \\in R_+, P^\top 1 = P 1 = p`
                "P": permutation domain, i.e., `O \\cap E \\cap N`.
                Default: "O"

    Returns:
        (float): Fused Gromov Wasserstein distance under different domains.
    """
    n = C1.shape[0]
    assert C1.shape[0] == C2.shape[0]
    if domain == "P":
        # split the quad + linear in Q1 and Q2
        if return_matrix:
            q_fval, Q1 = quad_solver(C1, C2, domain=domain, return_matrix=return_matrix)
            l_fval, Q2 = linear_solver(M, domain=domain, return_matrix=return_matrix)
            quad_term = alpha * q_fval + (1 - alpha) * l_fval
            fgw_bound = alpha * np.linalg.norm(C1)**2 + alpha * np.linalg.norm(C2)**2 - 2 * quad_term
            return fgw_bound / n**2, Q1, Q2
        else:
            quad_term = alpha * quad_solver(C1, C2, domain=domain) - (1 - alpha) * linear_solver(M, domain=domain)
            fgw_bound = alpha * np.linalg.norm(C1)**2 + alpha * np.linalg.norm(C2)**2 - 2 * quad_term
            return fgw_bound / n**2

    elif domain in ["O", "OE"]:

        # joint optimize over quad + linear
        if return_matrix:
            quad_linear_term, Q1, Q2 = fused_quad_solver(
                C1, C2, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            fgw_bound = alpha * np.linalg.norm(C1)**2 + alpha * np.linalg.norm(C2)**2 - quad_linear_term
            return fgw_bound / n**2, Q1, Q2
        else:
            quad_linear_term = fused_quad_solver(
                C1, C2, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            fgw_bound = alpha * np.linalg.norm(C1)**2 + alpha * np.linalg.norm(C2)**2 - quad_linear_term
            return fgw_bound / n**2
    elif domain == "NE":
        # exact fused gromov wasserstein distance
        _p = np.array([1. / n] * n)
        T, fgw_log = fused_gromov_wasserstein(M, C1, C2, _p, _p, loss_fun="square_loss", alpha=alpha, log=True)
        return fgw_log["fgw_dist"]


def fused_gromov_upper_bound_rec(M, C1, C2, alpha=0.5, domain="O", return_matrix=False):
    """
    .. math::

    """
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    if n1 >= n2:
        m = n1
        n = n2
    else:
        C1, C2 = C2, C1
        m = n2
        n = n1
        M = M.T

    def sym(X):
        return X.T / 2 + X / 2

    U = projection_matrix(m)
    V = projection_matrix(n)

    em = np.ones((m, 1))
    en = np.ones((n, 1))

    C_hat = U.T @ C1 @ U
    D_hat = V.T @ C2 @ V

    _C_hat, _D_hat = padding(C_hat, D_hat)

    F1 = C1 @ em @ en.T @ C2
    F2 = C2 @ en @ em.T @ C1
    F_hat = alpha * ((U.T @ F1 @ V) + (V.T @ F2 @ U).T) / np.sqrt(m * n) - (1 - alpha) / 2 * (V.T @ M.T @ U).T
    _F_hat = squarify(F_hat)

    def obj_func(Q):
        Q = Q.reshape([m - 1, m - 1])
        fval = alpha * np.trace(_C_hat @ Q @ _D_hat @ Q.T) + np.trace(_F_hat.T @ Q)
        grad = alpha * (_C_hat.T @ Q @ _D_hat.T + _C_hat @ Q @ _D_hat) + _F_hat
        return -fval, -grad.flatten()

    def proj_func(Q):
        Q = Q.reshape((m - 1, -1))
        u, s, vh = svd(Q)
        return (u @ vh).flatten()

    spg_options = default_options
    spg_options.curvilinear = 1
    spg_options.interp = 2
    spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
    spg_options.testOpt = False
    spg_options.verbose = 0
    spg_options.maxIter = 100
    # REVIEW: Q_init
    Q_init = np.identity(m - 1).flatten()
    res = SPG(obj_func, proj_func, Q_init, options=spg_options)
    q_opt = res[0].reshape((m - 1, m - 1))
    q_fval = - res[1]

    Q = q_opt[:m - 1, :n - 1]
    P = 1 / np.sqrt(m * n) * em @ en.T + U @ Q @ V.T
    # P = P/np.sqrt(m*n)

    sC = np.sum(C1)
    sD = np.sum(C2)
    sM = np.sum(M)
    quad_linear_term = q_fval + alpha * sC * sD / (m * n) - (1 - alpha) / 2 * sM / np.sqrt(m * n)
    fgw_bound = alpha * np.linalg.norm(C1)**2 + alpha * np.linalg.norm(C2)**2 - 2 * quad_linear_term

    if not return_matrix:
        return fgw_bound / m / n
    else:
        if n2 > n1:
            return fgw_bound / m / n, P.T / np.sqrt(m * n)
        else:
            return fgw_bound / m / n, P / np.sqrt(m * n)


def fgwtil_lb(M, C1, C2, alpha=1, domain="OE", return_matrix=False):
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    if n1 >= n2:
        C = C1
        D = C2
        m = n1
        n = n2
    else:
        C = C2
        D = C1
        m = n2
        n = n1
        M = M.T

    C_norm = np.linalg.norm(C)**2
    D_norm = np.linalg.norm(D)**2
    C_size = C.shape[0]
    D_size = D.shape[0]

    if domain in ["O", "OE"]:
        # joint optimize over quad + linear
        if return_matrix:
            quad_linear_term, Q1, Q2 = fused_quad_solver_rec(
                C, D, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            # DEBUG:
            fgw_bound = alpha * np.linalg.norm(C)**2 / C_size ** 2 \
                + alpha * np.linalg.norm(D)**2 / D_size ** 2 \
                - 2 * quad_linear_term / C_size / D_size
            return fgw_bound, Q1, Q2
        else:
            quad_linear_term = fused_quad_solver_rec(
                C, D, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            # DEBUG:
            fgw_bound = alpha * np.linalg.norm(C)**2 / C_size ** 2 \
                + alpha * np.linalg.norm(D)**2 / D_size ** 2 \
                - 2 * quad_linear_term / C_size / D_size
            return fgw_bound
    else:
        raise("Not Implemented!")


def fused_gromov_bound_rec(M, C1, C2, alpha=1, domain="O", return_matrix=False):
    """
    .. math::
    """
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    if n1 >= n2:
        C = C1
        D = C2
        m = n1
        n = n2
    else:
        C = C2
        D = C1
        m = n2
        n = n1
        M = M.T

    C_norm = np.linalg.norm(C)**2
    D_norm = np.linalg.norm(D)**2
    C_size = C.shape[0]
    D_size = D.shape[0]
    if domain in ["O", "OE"]:
        # joint optimize over quad + linear
        if return_matrix:

            quad_linear_term, Q1, Q2 = fused_quad_solver_rec(
                C, D, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            # DEBUG:
            fgw_bound = alpha * np.linalg.norm(C)**2 / C_size ** 2 \
                + alpha * np.linalg.norm(D)**2 / D_size ** 2 \
                - 2 * quad_linear_term / C_size / D_size
            return fgw_bound, Q1, Q2
        else:
            quad_linear_term = fused_quad_solver_rec(
                C, D, M, alpha=alpha, domain=domain, return_matrix=return_matrix)
            # DEBUG:
            fgw_bound = alpha * np.linalg.norm(C)**2 / C_size ** 2 \
                + alpha * np.linalg.norm(D)**2 / D_size ** 2 \
                - 2 * quad_linear_term / C_size / D_size
            return fgw_bound
    else:
        raise("Not Implemented!")


@PendingDeprecationWarning
def fused_gromov_bound2(M, C1, C2, gamma=0.5, domain="O"):
    """ Calculate FGW with domain specified.

    Notes:
        This is the version using scalar gamma on feature matrix.

    Args:
        C1 (np.ndarray): Distance matrix from source metric space.
        C2 (np.ndarray): Distance matrix from target metric space.
        alpha (float, optional): Trade-off between feature metric and topological metrics.
            Default: 0.5.
        domain (str, optional):
            Domain on quadratic term with following options:
                "O": othonormal domain, i.e., `P^\top P = P P^\top = I`
                "OE": othonormal and column / row sum domain, i.e., `P^\top 1 = P 1 = p`
                "NE": positive and column / row sum domain, i.e. `P \\in R_+, P^\top 1 = P 1 = p`
                "P": permutation domain, i.e., `O \\cap E \\cap N`.
                Default: "O"

    Returns:
        (float): Fused Gromov Wasserstein distance under different domains.
    """
    n = C1.shape[0]
    assert C1.shape[0] == C2.shape[0]
    # REVIEW: this is the factor applied to recover exact FGW
    alpha = 1 / (1 + 2 * gamma)
    if domain == "P":
        # split the quad + linear in Q1 and Q2
        quad_term = quad_solver(C1, C2, domain=domain) + linear_solver(gamma * M, domain=domain)
        fgw_bound = np.linalg.norm(C1)**2 + np.linalg.norm(C2)**2 - 2 * quad_term
        return fgw_bound / n**2
    elif domain in ["O", "OE"]:
        # joint optimize over quad + linear
        quad_linear_term = fused_quad_solver2(gamma * M, C1, C2, domain=domain)
        fgw_bound = np.linalg.norm(C1)**2 + np.linalg.norm(C2)**2 - 2 * quad_linear_term
        return fgw_bound / n**2
    elif domain == "NE":
        # exact fused gromov wasserstein distance
        _p = np.array([1. / n] * n)
        T, fgw_log = fused_gromov_wasserstein(M, C1, C2, _p, _p, loss_fun="square_loss", alpha=alpha, log=True)
        return fgw_log["fgw_dist"] * 1 / alpha


@PendingDeprecationWarning
def pb(A, B, maximize=True):
    """ Calculate the projection bound of QAP, i.e., PQAP.
    Details in paper [1].

    Args:
        A (np.ndarray): Symmetric matrix with dim (N, N).
        B (np.ndarray): Symmetric matrix with dim (N, N).
        maximize (bool): Maximize the quadratic term if `True`.

    Returns:
        lbd (float): lower bound of tr(AXBX^T)
        X (np.ndarray): The permutation matrix attains the lower bound.
            Take the reference of Theorem 2.1 from paper [1]

    Reference:
        [1] Hadley, Scott W., Franz Rendl, and Henry Wolkowicz.
        "A new lower bound via projection for the quadratic assignment problem."
        Mathematics of Operations Research 17.3 (1992): 727-739.
        [2] Rendl, Franz, and Henry Wolkowicz.
        "Applications of parametric programming and eigenvalue maximization to
        the quadratic assignment problem."
        Mathematical Programming 53.1 (1992): 63-78.

    Also see:
        __scipy.optimize.quadratic_assignment__
        [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.quadratic_assignment.html]
    """
    # get the size n of the matrices
    m, n = A.shape
    # assert on the symmetric square matrix
    np.testing.assert_equal(m, n)
    np.testing.assert_almost_equal(A.T, A)

    # sort row sums of A and B
    # ra is ordered nondecreasingly, rb is ordered nonincreasingly
    ra = np.sort(np.sum(A, axis=0))
    if maximize:
        rb = np.sort(np.sum(B, axis=0))
    else:
        rb = np.sort(np.sum(B, axis=0))[::-1]
    # set up the projection matrix V with size (n, n-1)
    V = projection_matrix(n)

    # sort eigenvalues of V' * A * V and V' * B * V
    a1 = V.T @ A @ V
    # make sure a1 is numerically symmetric
    a1 = (a1 + a1.T) / 2
    b1 = V.T @ B @ V
    # make sure b1 is numerically symmetric
    b1 = (b1 + b1.T) / 2

    a1_eig_w, a1_eig_v = np.linalg.eig(a1)
    b1_eig_w, b1_eig_v = np.linalg.eig(b1)
    # sort l1 in ascending order, sort l2 in descending order
    l1 = np.sort(a1_eig_w)
    if maximize:
        l2 = np.sort(b1_eig_w)
    else:
        l2 = np.sort(b1_eig_w)[::-1]

    # return lower bound, Corollary 4.2 of paper [1]
    lbd = l1.T @ l2 + ra.T @ rb * 2 / n - np.sum(A) * np.sum(B) / n / n
    return lbd


@DeprecationWarning
def qap_faq(A, B):
    """ Solve the QAP from scipy.optimize.
    ..math::
        $ \\min_{X} trace(AXBX^T) $

    Args:
        A (np.ndarray): A matrix with dim (n, n).
        B (np.ndarray): A matrix with dim (n, n).

    Returns:
        X (np.ndarray): permutation matrix.
        fun_val (float): approximate optimal function value.
    """
    n = A.shape[0]
    assert A.shape[0] == B.shape[0]
    res = quadratic_assignment(A, B)
    idx = res['col_ind']
    fun_val = res['fun']
    X = np.identity(n)[idx]
    return X, fun_val


@DeprecationWarning
def gwoe_pb(A, B):
    """ using gwo """
    n = A.shape[0]
    gwo_dist = (np.linalg.norm(A)**2 + np.linalg.norm(B)**2 - 2 * pb(A, B)) / n / n
    return gwo_dist.real


@DeprecationWarning
def gwp_faq(A, B):
    """ using quadratic assignment solver """
    n = A.shape[0]
    gwo_dist = (np.linalg.norm(A)**2 + np.linalg.norm(B)**2 - 2 * quadratic_assignment(A, B)['fun']) / n / n
    return gwo_dist.real


@DeprecationWarning
def lap(D, maximize=False, orth_only=False):
    """ Using the LAP solver from scipy """
    n = D.shape[0]
    if not orth_only:
        # NOTE: https://en.wikipedia.org/wiki/Hungarian_algorithm
        r_idx, c_idx = linear_sum_assignment(D, maximize=maximize)
        Q = np.identity(n)[c_idx] @ np.identity(n)[r_idx]
        f_val = D[r_idx, c_idx].sum()
        # assert the f_val
        np.testing.assert_almost_equal(np.trace(Q @ D), f_val)
    else:
        # NOTE: https://en.wikipedia.org/wiki/Trace_inequality
        D = -D if maximize else D
        U, s, Vh = svd(D)
        Q = U @ Vh
        f_val = s.sum()
    return Q, f_val


@DeprecationWarning
def fgwo_pb(C1, C2, x1, x2, alpha=0.5):
    n = C1.shape[0]
    F1 = cdist(x1, x1, metric='sqeuclidean')
    F2 = cdist(x2, x2, metric='sqeuclidean')

    CF1 = alpha * C1 + (1 - alpha) * F1 / n / n
    CF2 = alpha * C2 + (2 - alpha) * F2 / n / n

    return gwoe_pb(CF1, CF2)


@DeprecationWarning
def fgwo_faq(C1, C2, x1, x2, alpha=0.5):
    n = C1.shape[0]
    F1 = cdist(x1, x1, metric='sqeuclidean')
    F2 = cdist(x2, x2, metric='sqeuclidean')

    CF1 = alpha * C1 + (1 - alpha) * F1 / n / n
    CF2 = alpha * C2 + (2 - alpha) * F2 / n / n

    return gwp_faq(CF1, CF2)


# if __name__ == "__main__":
#     np.random.seed(0)
#     n = 10
#     A = np.random.rand(n, n)
#     A += A.T
#     B = np.random.rand(n, n)
#     B += B.T
#     p = np.array([1. / n] * n)
#     print(pb(A, B, maximize=True))
#     print(quad_solver(A, B, domain="OE"))
#     print(quad_solver(A, B, domain="O"))
#     T, gw_log = gromov_wasserstein(A, B, p, p, loss_fun="square_loss", log=True)
#     print(np.trace(A @ T @ B @ T.T))
#     print(gw_log['gw_dist'])
