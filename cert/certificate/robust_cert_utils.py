from inspect import Parameter
import cvxpy as cp
import numpy as np
from torch import relu
from fgw.certificate.dp import exact_solver_wrapper
from fgw.gromov_prox import projection_matrix
from fgw.utils import sym
from scipy.linalg import pinvh
from scipy.optimize import minimize

norm = np.linalg.norm
inv = pinvh


def logdet(Z):
    """ Log determinant function.

    .. math::
        f(Z) = \\log \\det (Z)

    Args:
        Z (np.ndarray): Input matrix with dim (n, n).

    Returns:
        float: log det fval.

    Warnings:
        NaN in log function.
    """
    fval = np.log(np.linalg.det(Z))
    return fval


def grad_logdet(Z):
    """ Gradient of logdet function.

    .. math::
        \\grad f(Z) = (Z^\\dagger)^\top

    Args:
        Z (np.ndarray): Input matrix with dim (n, n).

    Returns:
        np.ndarray: Gradient matrix with dim (n, n).

    Notes:
        Check `Z` is an symmetric matrix.
    """
    np.testing.assert_array_almost_equal(Z, Z.T)
    grad = inv(Z)
    return grad


def margin_star(R, A_org, XW, U, nG, delta_l, delta_g, act="linear", **kwargs):
    """ Given R, solve the X in M^* function.

    .. math::
        M^*(R) = \\max_W (tr(R^\top (W)) - \\sum_i f_i (W_i:)) - tr(R^\top A_org)
               = - \\min_W \\sum_i f_i (W_i:)) - tr(R^\top W) + tr(R^\top A_org)

        Where ||W - A_org||_1 <= \\delta_g

    Args:
        R (np.ndarray): Parameter R matrix with dim (n, n).
        A_org (np.ndarray): Original Adjacency matrix.
        XW (np.ndarray): Parameter from GNN with dim (d, h).
        U (np.ndarray): Parameter from GNN with dim (h, c).
        nG (int): Size of the graph.
        delta_l (int, or np.array): Row budgets. If it is an integer, expand it to list with same values.
        delta_g (int): Global budget.
        act (string, optional): Activation function. Defaults to be `linear`.

    Returns:
        tuple:
            float: optimal function value.
            np.ndarray: gradient w.r.t. R with dim (n, n).

    See Also:
        `fgw.certificate.dp.exact_solver_wrapper`
    """
    if act == "linear":
        # exact_solver finds the optimal A in the domain_g
        # returns -\tr(R.T X) + M(X)
        dp_sol = exact_solver_wrapper(A_org, np.tile(XW @ U, (nG, 1)), np.zeros(nG), -R, delta_l, 2 * delta_g, '1+2')
        fval_0, fval, A_opt = dp_sol

        # subtract from tr(R.T A_org)
        fval = -fval - np.trace(R.T @ A_org)

        # return the fval and grad in term of X, instead of A
        grad = A_opt - A_org

        # check the budget
        # assert np.abs(grad).sum() <= 2 * delta_g
        return fval, grad
    elif act == "relu":
        from gdro.optim.convex_relaxation import ConvexRelaxation
        cvx_relax = ConvexRelaxation(A_org, XW, U, delta_l, delta_g,
                                     activation="relu",
                                     relaxation="envelope",
                                     relu_relaxation="doubleL")
        cvx_sol = exact_solver_wrapper(A_org, cvx_relax.Q, cvx_relax.p, -R, delta_l, delta_g, '1+2')
        _, fval, A_opt = cvx_sol
        fval = -fval - np.trace(R.T @ A_org)
        grad = A_opt - A_org
        return fval, grad


def margin_star_v2(R, A_org, XWU, nG, delta_l, delta_g):
    """ Given R, solve the A in M^* function.
    NOTE: this is returns the optimal A as min problem, instead of X.

    .. math::
        M^*(R) = \\max_W (tr(R^\top (W)) - \\sum_i f_i (W_i:)) - tr(R^\top A_org)
               = - \\min_W \\sum_i f_i (W_i:)) - tr(R^\top W) + tr(R^\top A_org)

        Where ||W - A_org||_1 <= \\delta_g

    Args:
        R (np.ndarray): Parameter R matrix with dim (n, n).
        A_org (np.ndarray): Original Adjacency matrix.
        XWU (np.ndarray): Parameter from GNN with dim (d, c).
        nG (int): Size of the graph.
        delta_l (int, or np.array): Row budgets. If it is an integer, expand it to list with same values.
        delta_g (int): Global budget.

    Returns:
        tuple:
            float: optimal function value.
            np.ndarray: gradient w.r.t. R with dim (n, n).

    See Also:
        `fgw.certificate.dp.exact_solver_wrapper`
        `margin_star`
    """
    # exact_solver finds the optimal A in the domain_g
    # \min_A  M(A) - tr(R^T A)
    dp_sol = exact_solver_wrapper(A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), -R, delta_l, 2 * delta_g, '1+2')
    fval_0, fval, A_opt = dp_sol
    # NOTE: return the negation
    return -fval, A_opt


def margin_bidual(A_org, XWU, nG, delta_l, delta_g):
    """ Optimize the margin bidual.

    .. math::
        M**(X) = \\max_{R} tr(R^\top A) + \\min_W \\sum_i f_i (W_i:)) - tr(R^\top X)

        \\min_{X \\in co X} M**(A)
            = \\min_{X \\in co X} \\max_{R} \\min_W ..
            = \\max_{R} \\min_{X \\in co X} \\min_W ..

    Args:
        A_org (np.ndarray): Original adjacency matrix.
        XWU (np.ndarray): Model parameters.
        nG (int): Number of nodes.
        delta_l (np.array): Local budget.
        delta_g (int): Global budget.

    Returns:
        scipy.optimize.OptimizeResult: Optimal results from scipy.
    """
    def obj_bidual(R):
        R = R.reshape((nG, -1))
        R = sym(R)
        # solve M*
        fval_0, fval_opt, W_opt = exact_solver_wrapper(A_org, np.tile(
            XWU, (nG, 1)), np.zeros(nG), -R, delta_l, 2 * delta_g)
        # solve trace
        fval_trace, X_opt = solve_coX(R, delta_g, minimize=True)

        fval = np.trace(R.T @ X_opt) + fval_opt + np.trace(R.T @ A_org)
        grad = X_opt - (W_opt - A_org)
        return -fval, -grad.flatten()

    R_init = np.random.rand(nG**2)
    res = minimize(obj_bidual, R_init, jac=True)
    return res


def margin_bidual_eval(X, A_org, XWU, nG, delta_l, delta_g):
    """ Eval the margin bidual without optimizing over X.

    .. math::
        M**(X) = \\max_{R} tr(R^\top A) + \\min_W \\sum_i f_i (W_i:)) - tr(R^\top X)

    Args:
        X (np.ndarray): Input X matrix.
        A_org (np.ndarray): Original adjacency matrix.
        XWU (np.ndarray): Model parameters.
        nG (int): Number of nodes.
        delta_l (np.array): Local budget.
        delta_g (int): Global budget.

    Returns:
        scipy.optimize.OptimizeResult: Optimal results from scipy.
    """
    np.fill_diagonal(X, 0)
    n = X.shape[0]

    def obj_bidual_eval(R):
        R = R.reshape((n, -1))
        fval_0, fval_opt, W_opt = exact_solver_wrapper(
            A_org, np.tile(XWU, (nG, 1)), np.zeros(nG), -R, delta_l, 2 * delta_g)
        # NOTE: no optimizing over X
        fval = np.trace(R.T @ X) + fval_opt + np.trace(R.T @ A_org)
        grad = X - (W_opt - A_org)
        return -fval, -grad.flatten()
    R_init = np.random.rand(n**2)
    res = minimize(obj_bidual_eval, R_init, jac=True)
    if np.isnan(res['fun']):
        print("debug nan")
    return res


def solve_coX(W, delta_g, delta_l=None, minimize=False, **kwargs):
    """ Solve X in co X with global budget (and local budget).

    .. math::
        \\min_{X} tr(W^\top X)  OR  \\max_{X} tr(W^\top X)
            where X \\in co \\Xcal

    Args:
        W (np.ndarray): A symmetric matrix.
        delta_g (int): Global budget.
        delta_l (np.array, optional): Local budget with dim (n, ). Defaults to None.
        minimize (bool, optional): Minimize objective if `True`. Defaults to False.

    Returns:
        tuple:
            (float): fval_opt
            (np.ndarray): X_opt
    """
    np.testing.assert_array_almost_equal(W, W.T)
    X = cp.Variable(W.shape)
    loss = cp.trace(W.T @ X)
    const = [X <= 1, X >= 0, cp.sum(X) <= 2 * delta_g, X >= X.T, X.T >= X, cp.diag(X) == 0]
    if isinstance(delta_l, np.ndarray):
        const += [cp.sum(X, 1) <= delta_l]

    if minimize:
        obj = cp.Minimize(loss)
    else:
        obj = cp.Maximize(loss)
    problem = cp.Problem(obj, const)

    try:
        problem.solve()
        X_opt = X.value
        fval_opt = obj.value
        return fval_opt, X_opt
    except Exception:
        print("Cannot solve the problem")
