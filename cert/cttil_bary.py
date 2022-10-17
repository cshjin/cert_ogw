""" Barycentering using CTtil as dist_func """
import numpy as np
from ot.gromov import gromov_wasserstein
from scipy.optimize import minimize

from fgw.dist import calc_T, cttil_Z


def grad_D_A(A):
    """ Partial derivative of diagonal matrix w.r.t. adjacency matrix

    Args:
        A (np.ndarray): Adjacency matrix with dim (N, N).

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N)
    """
    N = A.shape[0]
    grad_ = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if i == j and i == k and k != l:
                        grad_[i, j, k, l] = 1 / 2
                    if i == j and j == l and k != l:
                        grad_[i, j, k, l] = 1 / 2
                    if i == j and i == k and j == l:
                        grad_[i, j, k, l] = 1
    return grad_


def grad_L_A(A):
    """ Partial derivative of Laplacian matrix w.r.t. adjacency matrix

    .. math::
        \\partial L / \\parital A =  \\partial D / \\parital A - \\partial A / \\partial A
        \\partial A / \\parital A = I \\otimes I

    Args:
        A (np.ndarray): Adjacency matrix with dim (N, N).

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N).
    """
    N = A.shape[0]
    grad_ = grad_D_A(A) - np.kron(np.eye(N), np.eye(N)).reshape((N, N, N, N))
    return grad_


def grad_Z_L(L):
    """ Partial derivative of Z matrix w.r.t. Laplacian matrix

    .. math::
        \\partial Z / \\partial L = \\partial Z / \\partial L_inv : \\partial L_inv / \\partial L

    Args:
        L (np.ndarray): Laplacian matrix with dim (N, N)

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N).
    """
    N = L.shape[0]
    _L = L + 1 / N * np.ones((N, N))
    _L_inv = np.linalg.pinv(_L)
    grad_Z_X = - np.kron(_L_inv, _L_inv).reshape((N, N, N, N))
    # NOTE: alternative implementation
    # grad_X_L = np.zeros((N, N, N, N))
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             for l in range(N):
    #                 if i == k and j == l:
    #                     grad_X_L[i, j, k, l] = 1
    grad_X_L = np.kron(np.eye(N), np.eye(N)).reshape((N, N, N, N))
    grad_ = np.einsum("ijkl,klpq->ijpq", grad_Z_X, grad_X_L)
    return grad_


def grad_T_Z(Z):
    """ Partial derivative of T matrix w.r.t. Z matrix.

    .. math::
        \\partial T_ij      1    if i = j and i = k OR i != j and i = k OR i != j and j = l
        ------------  =    -2       i !=j and i = k and j = l
        \\partial Z_kl      0    otherwise

    Args:
        Z (np.ndarray): Z matrix with dim (N, N)

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N)
    """
    N = Z.shape[0]
    grad_ = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if i == j and i == k:
                        grad_[i, j, k, k] = 1
                    if i != j and i == k:
                        grad_[i, j, k, k] = 1
                    if i != j and j == l:
                        grad_[i, j, l, l] = 1
                    if i != j and i == k and j == l:
                        grad_[i, j, k, l] = -2
    return grad_


def grad_C_T(T):
    """ Partial derivative of CTtil w.r.t. T matrix.

    .. math::
        \\partial C_ij      1    if i != j and i = k and j = l
        ------------  =
        \\partial T_kl      0    otherwise

    Args:
        T (np.ndarray): T matrix with dim (N, N).

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N).
    """
    N = T.shape[0]
    grad_ = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if i == k and j == l and i != j:
                        grad_[i, j, k, l] = 1
    return grad_


def grad_C_A(A):
    """ Partial derivative of CTtil w.r.t. adjacency matrix.

    Args:
        A (np.ndarray): Adjacency matrix with dim (N, N).

    Returns:
        np.ndarray: Gradient tensor with dim (N, N, N, N).
    """
    N = A.shape[0]
    L = np.diag(A.sum(0)) - A
    Z = np.linalg.pinv(L + 1 / N * np.ones((N, N)))
    T = calc_T(Z)
    # grad_ = np.einsum("ijkl,klpq,pqmn,mnxy->ijxy",
    #                   grad_C_T(T),
    #                   grad_T_Z(Z),
    #                   grad_Z_L(L),
    #                   grad_L_A(A))
    # NOTE: tensordot is faster than einsum
    grad_ = np.tensordot(np.tensordot(np.tensordot(grad_C_T(T), grad_T_Z(Z)), grad_Z_L(L)), grad_L_A(A))
    return grad_


def optimize_Z(N, Ds, Zs, ps, p, lambdas, Z_init=None):
    """ Optimize over Z in barycenter problem using Gromov-Wasserstein distance.

    Args:
        N (int): Size of barycenter.
        Ds (list): List of distance matrices, each in `np.ndarray`.
        ps (list): List of node distributions, each in `np.array`.
        p (np.array): Node distrbution of barycenter with dim (N, ).
        lambdas (np.array): Sample weights with dim (S, ).
        Z_init (np.ndarray, optional): Initial Z if provided, otherwise use one matrix. Defaults to None.

    Returns:
        np.ndarray: Optimized Z with dim (N, N)
    """
    S = len(Ds)
    debug = False
    # debug = True

    def obj(Z):
        # barycenter objective function
        Z = Z.reshape((N, N))
        C = cttil_Z(Z)
        T_mat = calc_T(Z)

        Ts = []
        logs = []
        for i in range(S):
            T, log = gromov_wasserstein(C, Ds[i], p, ps[i], loss_fun="square_loss", log=True)
            Ts.append(T)
            logs.append(log)
        fval = sum([logs[i]['gw_dist'] for i in range(S)])

        grad_ = 0
        for i in range(S):
            grad_ += lambdas[i] * np.einsum("ij,ijkl,klpq->pq",
                                            (2 * C / N**2 - 2 * Ts[i] @ Ds[i] @ Ts[i].T),
                                            grad_C_T(T_mat),
                                            grad_T_Z(Z))

        return fval, grad_.flatten()

    def callback(Z):
        fval, grad = obj(Z)
        print(f"obj {fval:.4f}", f"||g|| {np.linalg.norm(grad):.4f}")
        pass

    if Z_init is None:
        # REVIEW: improve the initialization
        # Z_init = np.ones((N, N)).flatten()
        Z_init = np.zeros((N, N)).flatten()

    res = minimize(obj, Z_init, method="BFGS", jac=True, callback=callback)
    Z_opt = res['x'].reshape((N, N))
    # print("obj value", res['fun'])
    if debug:
        import matplotlib.pyplot as plt
        C_opt = cttil_Z(Z_opt)
        T_, log_ = gromov_wasserstein(C_opt, Ds[0], p, ps[0], loss_fun="square_loss", log=True)
        plt.subplot(2, 4, 1)
        plt.imshow(T_)
        plt.title("transportation matrix")
        plt.subplot(2, 4, 2)
        plt.imshow(Zs[0])
        plt.title("Z from sample")
        plt.subplot(2, 4, 3)
        plt.title("Z_opt")
        plt.imshow(Z_opt)
        plt.subplot(2, 4, 4)
        plt.imshow(T_.T @ Z_opt @ T_ * N * N)
        plt.title("Z_opt after permutation")

        plt.subplot(2, 4, 5)
        plt.imshow(cttil_Z(T_.T @ Z_opt @ T_ * N * N))
        plt.title("C from Z_opt pert")
        # plt.imshow(C_opt)
        # plt.show()
        plt.subplot(2, 4, 6)
        plt.imshow(Ds[0])
        plt.title("C from sample")
        plt.subplot(2, 4, 7)
        plt.imshow(C_opt)
        plt.title("C from Z_opt")
        plt.subplot(2, 4, 8)
        plt.imshow(T_.T @ C_opt @ T_ * N * N)
        plt.title("C_opt after permutation")

        plt.tight_layout()
        plt.show()

    return Z_opt


def optimize_Z_v2(N, Ds, Zs, ps, p, lambdas, Z_init=None):
    """ Optimize over Z in barycenter problem using Gromov-Wasserstein distance.

    Args:
        N (int): Size of barycenter.
        Ds (list): List of distance matrices, each in `np.ndarray`.
        ps (list): List of node distributions, each in `np.array`.
        p (np.array): Node distrbution of barycenter with dim (N, ).
        lambdas (np.array): Sample weights with dim (S, ).
        Z_init (np.ndarray, optional): Initial Z if provided, otherwise use one matrix. Defaults to None.

    Returns:
        np.ndarray: Optimized Z with dim (N, N)
    """
    S = len(Ds)
    # debug = False
    debug = True

    Ps = [np.zeros(D.shape) for D in Ds]

    def obj(Z):
        # barycenter objective function
        Z = Z.reshape((N, N))
        C = cttil_Z(Z)
        T_mat = calc_T(Z)

        # Ps = []
        logs = []
        for i in range(S):
            # NOTE: tr(DPCP.T)
            P, log = gromov_wasserstein(Ds[i], C, ps[i], p, loss_fun="square_loss", log=True)
            # Ps.append(P)
            Ps[i] = P
            logs.append(log)

        fval = sum([logs[i]['gw_dist'] * lambdas[i] for i in range(S)])

        grad_ = 0
        for i in range(S):
            grad_ += lambdas[i] * np.einsum("ij,ijkl,klpq->pq",
                                            (2 * C / N**2 - 2 * Ps[i].T @ Ds[i] @ Ps[i]),
                                            grad_C_T(T_mat),
                                            grad_T_Z(Z))

        return fval, grad_.flatten()

    def callback(Z):
        fval, grad = obj(Z)
        print(f"obj {fval:.4f}", f"||g|| {np.linalg.norm(grad):.4f}")
        pass

    if Z_init is None:
        # REVIEW: improve the initialization
        Z_init = np.ones((N, N)).flatten()
        # Z_init = np.zeros((N, N)).flatten()

    res = minimize(obj, Z_init, method="BFGS", jac=True, callback=callback)
    Z_opt = res['x'].reshape((N, N))
    # Z_opt = update_square_loss(p, lambdas, Ps, Zs)
    # print("obj value", res['fun'])
    if debug:
        import matplotlib.pyplot as plt
        C_opt = cttil_Z(Z_opt)
        T_, log_ = gromov_wasserstein(C_opt, Ds[0], p, ps[0], loss_fun="square_loss", log=True)
        plt.subplot(2, 4, 1)
        plt.imshow(T_)
        plt.title("transportation matrix")
        plt.subplot(2, 4, 2)
        plt.imshow(Zs[0])
        plt.title("Z from sample")
        plt.subplot(2, 4, 3)
        plt.title("Z_opt")
        plt.imshow(Z_opt)
        plt.subplot(2, 4, 4)
        plt.imshow(T_.T @ Z_opt @ T_ * N * N)
        plt.title("Z_opt after permutation")

        plt.subplot(2, 4, 5)
        plt.imshow(cttil_Z(T_.T @ Z_opt @ T_ * N * N))
        plt.title("C from Z_opt pert")
        # plt.imshow(C_opt)
        # plt.show()
        plt.subplot(2, 4, 6)
        plt.imshow(Ds[0])
        plt.title("C from sample")
        plt.subplot(2, 4, 7)
        plt.imshow(C_opt)
        plt.title("C from Z_opt")
        plt.subplot(2, 4, 8)
        plt.imshow(T_.T @ C_opt @ T_ * N * N)
        plt.title("C_opt after permutation")

        plt.tight_layout()
        plt.show()

    return Z_opt


def solve_A_from_Z(Z):
    """ Recover A from Z matrix

    .. math::
        Z = (L + 11^\top / n)^-1  where  L = D - A
        A = offdiag(-(Z^-1 - 11^\top / n))

    Args:
        Z (np.ndarray): Z matrix with dim (N, N)

    Returns:
        np.ndarray: recovered A matrix with dim (N, N)

    Note:
        It won't handle the discrete constraint on A within the function.
    """
    N = Z.shape[0]
    Z = np.asarray(Z)
    L = np.linalg.pinv(Z) - np.ones((N, N)) / N

    A = solve_A_from_L_cvx(L)
    return A


def solve_A_from_L_cvx(L):
    """ Given L, solve A in cvx

    .. math::
        \\min_A = ||L - diag(A.sum(1)) + A||_F
        s.t. A \\in {0, 1}

    Args:
        L (np.ndarray): (Approx.) Laplacian matrix.

    Returns:
        np.ndarray: Adjacency matrix.
    """
    import cvxpy as cp
    A = cp.Variable(L.shape, boolean=True)
    loss = cp.norm(L - cp.diag(cp.sum(A, 1)) + A, p="fro")
    problem = cp.Problem(cp.Minimize(loss), [A >= A.T, A.T >= A, cp.diag(A) == 0])
    problem.solve(solver=cp.CPLEX)
    return np.fix(A.value).astype(int)


def _solve_A_from_Z(Z):
    """ DEPRECATED """
    # DEBUG: check the correctness of L
    N = Z.shape[0]
    Z = np.asarray(Z)
    L = np.linalg.pinv(Z) - np.ones((N, N)) / N
    L = L - L.sum() / N**2
    L_ = -L
    # offset = L.sum(1) / N
    A_ = L_
    D_ = np.diag(np.diag(L_))
    np.fill_diagonal(A_, 0)
    return A_


def _solve_A_scipy(L):
    """ solve A in scipy with const """
    # TOFIX
    from scipy.optimize import NonlinearConstraint

    N = L.shape[0]

    def cons_func(x):
        """ x(1-x) = 0, A[i, i] = 0 """
        cons = []
        for i in range(N**2):
            cons.append(x[i] * (1 - x[i]))
        A = x.reshape(L.shape)
        # for i in range(N):
        #     cons.append(A[i, i])
        return cons

    nlc = NonlinearConstraint(cons_func, 0, 0)
    cons = [{'type': "eq", "fun": cons_func}]

    def grad_D_A(A):
        N = A.shape[0]

        grad_ = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if i == j and k == i:
                            grad_[i, j, k, l] = 1
        return grad_

    def obj(x):
        A = x.reshape(L.shape)
        A = A + A.T
        A = A / 2
        D = np.diag(A.sum(1))
        fval = np.linalg.norm(L - D + A) ** 2
        grad = np.einsum("ij,ijkl->kl", 2 * (L - D + A), grad_D_A(A))
        return fval, grad.flatten()

    x_init = np.random.randn(N**2)
    # x_init = np.ones(N**2)
    # G = nx.cycle_graph(N)
    # x_init = nx.adjacency_matrix(G).toarray().flatten()
    # SLSQP
    res = minimize(obj, x_init, jac=True, constraints=cons)
    A_opt = res['x'].reshape(L.shape)
    return A_opt


def _solve_A_quasi_proj(L):
    """ solve A in proj update """
    # TOFIX
    from fgw.spg import SPG, default_options

    N = L.shape[0]

    def grad_D_A(A):
        N = A.shape[0]
        grad_ = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if i == j and k == i:
                            grad_[i, j, k, l] = 1
        return grad_

    def obj(A):
        A = A.reshape(L.shape)
        D = np.diag(A.sum(1))
        fval = np.linalg.norm(L - D + A) ** 2
        grad = np.einsum("ij,ijkl->kl", 2 * (L - D + A), grad_D_A(A))
        return fval, grad.flatten()

    def proj(A):
        A = A.reshape(L.shape)
        A += A.T
        A = A / 2
        A = np.round(A)
        A = np.maximum(np.minimum(A, 1), 0)
        np.fill_diagonal(A, 0)
        return A.flatten()

    A_init = np.random.randn(N**2)
    # A_init = np.ones(N**2)
    # G = nx.cycle_graph(N)
    # A_init = nx.adjacency_matrix(G).toarray().flatten()
    default_options.verbose = 2
    res = SPG(obj, proj, A_init, options=default_options)
    A_opt = res[0].reshape(L.shape)
    return A_opt


def _solve_A_ls(L):
    """ solve the A based on line search """
    # A = np.copy(L - L.min())
    # np.fill_diagonal(A, 0)
    # np.fill_diagonal(A, 0)
    search = np.linspace(L.min(), L.max(), 100)
    errs = []
    As = []
    for s in search:
        A_ = np.copy(L)
        np.fill_diagonal(A_, 0)
        A_[A_ >= s] = 1
        A_[A_ < s] = 0
        D_ = np.diag(A_.sum(1))
        errs.append(np.linalg.norm(L - L.min() - D_ + A_))
        As.append(A_)
        # print(np.linalg.norm(L - D_ + A_), np.linalg.norm(A_))
    A = As[np.argmin(errs)]
    return A
