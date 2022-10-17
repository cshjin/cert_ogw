import pickle

import cvxpy as cp
import numpy as np
from fgw.bundle import ProximalBundleMethod as PBM
from fgw.certificate.robust_cert_utils import (grad_logdet, logdet,
                                               margin_bidual,
                                               margin_bidual_eval, margin_star)
from fgw.dist import cttil
from fgw.gromov_prox import eig_decom, projection_matrix
from fgw.gwtil import Qcal_ub
from fgw.gwtil_bary import grad_Qcal_ub_C
from fgw.gwtil_cvx import omega_circ_star, omega_circ_star_v2
from fgw.spg import SPG, SPGOptions, default_options
from fgw.utils import sym
from nsopy.loggers import GenericMethodLogger
from nsopy.methods.bundle import BundleMethod as BM
from nsopy.methods.bundle import CuttingPlanesMethod as CPM
from nsopy.methods.quasi_monotone import SGMTripleAveraging as TA
from nsopy.methods.subgradient import SubgradientMethod as SG
from scipy.linalg import pinvh
from scipy.optimize import approx_fprime, check_grad, minimize

norm = np.linalg.norm
inv = pinvh


def process_var(x, n):
    """ Process vectorized variable into separate ones.

        U: n * n
        R: (n-1) * (n-1)
        s: n
        alpha: 1
        gamma: 1

        Total: n^2 + (n-1)^2 + n + 2
    Args:
        x (np.array):
        n (int):

    Returns:
        tuple: splitted 5 variables.
    """
    U_var = x[: n**2].reshape((n, -1))
    R_var = x[n**2: n**2 + (n - 1)**2].reshape((n - 1, -1))
    s_var = x[n**2 + (n - 1)**2: n**2 + (n - 1)**2 + n]
    alpha = x[-2]
    gamma = x[-1]
    return U_var, R_var, s_var, alpha, gamma


def assemble_var(U, R, s, alpha, gamma):
    """ Assemble multiple variables into a single vector.

    Args:
        U (np.ndarray): Variable U with dim (n, n).
        R (np.ndarray): Variable R with dim (n-1, n-1).
        s (np.array): Variable s with dim (n, ).
        alpha (float): Variable alpha.
        gamma (float): Variable gamma.

    Returns:
        np.array: Single vector format.
    """
    return np.concatenate([U.flatten(), R.flatten(), s, [alpha], [gamma]])


def cttil_cvx_env_solver(A_org, D0, XW, U, delta_l, delta_g, delta_omega, **kwargs):
    """ Solve convex env problem from maximizing its dual.

    NOTE: with alpha \\ge 0, gamma \\ge 0.

        .. math::
        \\max_{U, R, s, alpha, gamma} \\min_{X, Z}
            tr(U^T X)
            - M*(U)
            + tr(Q^\top Acal_Z)
            - alpha * \\Omega_\\circ^*(Q/alpha)
            - alpha * delta_omega
            + gamma * F^*(Z)
            + gamma * F(CX + K)
            - gamma * tr(BZ) + gamma
        =
        \\max_{U, R, s, alpha, gamm} \\min_{X}
            tr(U^T X)
            - M*(U)
            - alpha * \\Omega_\\circ^*(Q/alpha)
            - alpha * delta_omega
            - gamma * logdet(CX + K)
            + gamma * logdet(11^T/n + 1/gamma VRRV^T)

        NOTE: project R
        Q         = diag(s) +  M \\circ (\\gamma B - V R R V^T)/2
        Acal^*(Q) = 2(Q - diag(Q1))
        P         = 11^T/n + 1/gamma VRRV^T
        Z_opt     = - P^{-1}

    Args:
        A_org (np.ndarray): Original adjacency matrix.
        D0 (np.ndarray): Original distance matrix.
        XW (np.ndarray): Feature matrix with dim (n, d)
        U (np.ndarray): Weight from linear layer with dim (d, c).
        delta_l (int or np.array): Local budget.
        delta_g (int): Global budget.
        delta_omega (float): GWtil distance matrix.

    Returns:
        (dict): {"fun": fval_opt, "x": x_opt}
    """
    AUTOGRAD = kwargs.get("autograd", False)
    VERBOSE = kwargs.get("verbose", False)
    CHECK_GRAD = kwargs.get("check_grad", False)
    DEBUG = kwargs.get("debug", False)
    SOLVER = kwargs.get("solver", "joint")
    MAXITER = kwargs.get("maxiter", 15000)

    n = A_org.shape[0]
    # projection matrix
    V = projection_matrix(n)
    e = np.ones(n)
    eet = np.ones((n, n))
    # Laplacian matrix
    L = np.diag(A_org.sum(1)) - A_org

    # Z = -inv(L + eet)
    # T = np.zeros([n, n])
    # for i in range(n):
    #     for j in range(n):
    #         T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2

    # eq 113:
    K = L + 1 / n * eet

    XWU = XW @ U

    Mcal = np.ones((n, n))
    np.fill_diagonal(Mcal, 0)

    # bound of Beta
    # B = np.zeros((n, n))
    # B = Mcal * (L + eet)
    # FIXED
    B = L - 2 * (np.eye(n) * L)
    # setting the upper bound of B
    # B = T * Mcal

    # NEW: CVXPY problem setup, for efficiency
    A_var = cp.Variable(A_org.shape, symmetric=True)
    Phi_A = - A_var + cp.diag(cp.sum(A_var, 1)) + eet / n
    const = [cp.diag(A_var) == 0,
             cp.sum(A_var - A_org) <= 2 * delta_g,
             A_var - A_org >= 0,
             A_var - A_org <= 1]
    const += [cp.sum(A_var - A_org, 0) <= delta_l]

    def obj(x, debug=False, mask_all=True,
            mask_U=False, mask_R=False,
            mask_s=False, mask_alpha=False, mask_gamma=False, xk=None):
        """ Objective function """
        # in case of block update
        if mask_all:
            U_var, R_var, s_var, alpha, gamma = process_var(x, n)
        elif mask_U:
            U_var = x.reshape((n, -1))
            _, R_var, s_var, alpha, gamma = process_var(xk, n)
        elif mask_R:
            R_var = x.reshape((n - 1, -1))
            U_var, _, s_var, alpha, gamma = process_var(xk, n)
        elif mask_s:
            s_var = x
            U_var, R_var, _, alpha, gamma = process_var(xk, n)
        elif mask_alpha:
            alpha = x
            U_var, R_var, s_var, _, gamma = process_var(xk, n)
        elif mask_gamma:
            gamma = x
            U_var, R_var, s_var, alpha, _ = process_var(xk, n)

        U_var = sym(U_var)

        ''' solve A with SDP solver '''
        try:
            loss_A = -gamma * cp.log_det(Phi_A) + cp.trace(U_var.T @ A_var)
            obj_A = cp.Minimize(loss_A)
            problem = cp.Problem(obj_A, const)
            problem.solve(cp.MOSEK)
            A_opt = A_var.value
        except Exception as error:
            print(error)
            print("Set A_opt as the A_org")
            A_opt = A_org

        if problem.status in ['infeasible', 'unbounded']:
            A_opt = A_org

        ''' recover X_opt '''
        X_opt = A_opt - A_org
        CX = -X_opt + np.diag(X_opt.sum(1))

        # REVIEW: B = 0 OR B = L
        J = V @ R_var @ R_var.T @ V.T
        Q = np.diag(s_var) + Mcal * (gamma * B - J) / 2
        P = eet / n + J / gamma
        P_inv = inv(P)
        Z_opt = -P_inv

        # REVIEW:
        # T = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         T[i, j] = -(Z_opt[i, i] + Z_opt[j, j] - 2 * Z_opt[i, j]) if i != j else -Z_opt[i, i]
        # Acal_Z = T * (eet - np.eye(n))

        fval_margin_star, grad_margin_star = margin_star(
            U_var, A_org, XW, U, n, delta_l, delta_g, act=kwargs.get("act"))
        fval_omega_circ_star, grad_omega_circ_star = omega_circ_star_v2(Q / alpha, D0)

        fval_dual = \
            + np.trace(U_var.T @ X_opt) \
            - fval_margin_star \
            - alpha * fval_omega_circ_star \
            - alpha * delta_omega \
            + gamma * logdet(P) \
            - gamma * logdet(CX + K)

        """ check feasible """
        # # if debug:
        # Acal_star_Q = 2 * (Q - np.diag(Q.sum(1)))
        # z = np.diag(Z_opt)
        # Acal_Z = 2 * Z_opt - np.outer(e, z) - np.outer(z, e)

        # # fval_primal = margin_bidual(A_org, XWU, n, delta_l, delta_g)['fun']
        # fval_primal = - margin_bidual_eval(X_opt, A_org, XWU, n, delta_l, delta_g)['fun']

        # fval_c1 = np.trace(Q.T @ Acal_Z) - alpha * fval_omega_circ_star - alpha * delta_omega
        # fval_c2 = -n - logdet(-Z_opt) - logdet(CX + K) + 1 - np.trace(B @ Z_opt)

        # # fval_omega_circ_Z, _ = omega_circ_star(Acal_Z, D0)
        # # fval_omega_circ_X, _ = omega_circ_star(cttil(A_opt), D0)

        # print(f"{'d_fval:'} {fval_dual:<8.2e} ",
        #       f"{'p_fval:'} {fval_primal:<8.2e} ",
        #       f"{'  diff:'} {fval_primal - fval_dual:<8.2e} ",
        #       f"{'   d<P:'} {fval_dual <= fval_primal:<4} ",
        #       f"{'   c_1:'} {fval_c1:<8.2e} ",
        #       #   f"{'  Ωo_Z:'} {fval_omega_circ_Z:<8.2e} ",
        #       #   f"{'  Ωo_X:'} {fval_omega_circ_X:<8.2e} ",
        #       f"{'   c_2:'} {fval_c2:<8.2e} ",
        #       f"{' alpha:'} {alpha:>9.2e}",
        #       f"{' gamma:'} {gamma:>9.2e}")

        ''' gradient information '''
        if not AUTOGRAD:

            grad_U = X_opt - grad_margin_star

            grad_Q = - grad_omega_circ_star
            grad_R = - V.T @ (grad_Q * Mcal) @ V @ R_var \
                + V.T @ P_inv @ V @ R_var * 2

            grad_s = np.diag(grad_Q)

            grad_alpha = - delta_omega - fval_omega_circ_star \
                + 1 / alpha * np.trace(grad_omega_circ_star.T @ Q)

            grad_gamma = \
                + 1 * logdet(P) \
                - 1 * logdet(CX + K) \
                + np.trace((grad_Q * Mcal) @ B) / 2 \
                - 1 / gamma * np.trace(P_inv.T @ J)

            # combine gradient
            if mask_all:
                _grad = np.concatenate([grad_U.flatten(),
                                        grad_R.flatten(),
                                        grad_s,
                                        [grad_alpha],
                                        [grad_gamma]])
            elif mask_U:
                _grad = grad_U.flatten()
            elif mask_R:
                _grad = grad_R.flatten()
            elif mask_s:
                _grad = grad_s
            elif mask_alpha:
                _grad = grad_alpha
            elif mask_gamma:
                _grad = grad_gamma

        else:
            _grad = np.zeros(len(x))

        return -fval_dual, -_grad

    def callback(x):
        """ callback within solvers """
        fval, grad = obj(x)

        if VERBOSE:
            print(f"callback BFGS: fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}",
                  f"alpha {x[-2]:<10.2e}",
                  f"gamma {x[-1]:<10.2e}")

        # NOTE: early stop: customized in _minimize_lbfgsb
        if fval <= 0:
            return True

    def proj(x, eps=1e-4):
        """ Projection function """
        # Projection of alpha, gamma
        x[-2] = max(eps, x[-2])
        x[-1] = max(eps, x[-1])
        return x

    ''' start optimization '''
    if kwargs.get("x0") is None:
        x0 = np.random.randn(n**2 + (n - 1)**2 + n + 2) * 0.01
        x0[-1] = abs(x0[-1] * 100)
        x0[-2] = abs(x0[-2] * 100)

        if DEBUG:
            res = pickle.load(open("debug_x.pkl", "rb"))
            x0 = res['x']
            obj(x0, debug=True)
            exit()
    else:
        x0 = kwargs.get("x0")

    if SOLVER == "bfgs":

        ''' check grad '''
        if CHECK_GRAD:
            grad_approx = approx_fprime(x0, lambda x: obj(x)[0], 1e-8)
            grad_true = obj(x0)[1]
            print(f"{'all':<15s}",
                  f"||g||_approx {norm(grad_approx):>10.4f}",
                  f"||g||_true {norm(grad_true):>10.4f}",
                  f"||g||_diff {norm(grad_approx - grad_true):>10.4f}")
            exit()
        ''' solve in bfgs with explicit grad '''

        bnd = [(None, None)] * len(x0)
        bnd[-1] = (1e-3, None)
        bnd[-2] = (1e-3, None)
        res = minimize(obj, x0,
                       method="L-BFGS-B",
                       jac=True,
                       callback=callback if VERBOSE else None,
                       bounds=bnd,
                       options={'early_stopping': True}
                       )

        if VERBOSE:
            print(f"{'fval':<10s} {-res['fun']:>10.2e} status {res['status']}")
            _, _ = obj(res['x'], debug=True)

        if DEBUG:
            pickle.dump(res, open("debug_x.pkl", "wb"))
            exit()

        return res

    elif SOLVER == "joint":
        ''' Start with L-BFGS-B and then run bundle method '''
        bnd = [(None, None)] * len(x0)
        bnd[-1] = (1e-3, None)
        bnd[-2] = (1e-3, None)

        xk = x0
        # REVIEW: alpha, gamma in PBM?
        # p = PBM(n=len(x0), alpha=1000, gamma=0.005, sense=min)
        # p = PBM(n=len(x0), alpha=1, gamma=0.005, sense=min)
        p = PBM(n=len(x0), alpha=kwargs.get("alpha", 100), gamma=kwargs.get("gamma", 0.001), sense=min)
        p.custom_constraints = [p.x[-1] >= 1e-3, p.x[-2] >= 1e-3]

        res = minimize(obj, xk,
                       method="L-BFGS-B",
                       jac=True,
                       callback=callback if VERBOSE else None,
                       bounds=bnd,
                       options={'early_stopping': True, 'maxiter': MAXITER}
                       )
        if MAXITER == 1:
            return res
            
        if res['fun'] < 0:
            return res

        xk = res['x']

        for i in range(1000):
            fval, grad = obj(xk)
            if fval < 0:
                # early stopping
                return {"fun": fval, "x": xk}
            xk = p.step(fval, xk, grad)
            if VERBOSE:
                if i % 10 == 0:
                    print(f"iter {i:04d} d_fval {-fval:6.2e} alpha {xk[-2]:6.2e} gamma {xk[-1]:6.2e}")

        return {"fun": fval, "x": xk}

    elif SOLVER == "spg":
        ''' solve in spg '''
        spg_options = default_options
        spg_options.curvilinear = 1
        spg_options.interp = 2
        spg_options.numdiff = 0  # 0 to use gradients, 1 for numerical diff
        spg_options.testOpt = True
        spg_options.verbose = 2 if VERBOSE else 0
        spg_options.maxIter = 10000
        spg_options.optTol = 1e-15
        spg_options.progTol = 1e-15
        # spg_options.early_stopping = False  # new in optimizer, return if f < 0

        res = SPG(obj, lambda x: proj(x), x0, options=spg_options)
        return {'fun': res[1], 'x': res[0]}

    elif SOLVER in ["ta", "sg", "bundle"]:
        ''' solve in non-smooth solvers '''
        if SOLVER == "ta":
            method = TA(lambda x: (0, -obj(x)[0], -obj(x)[1]),
                        # lambda x: x,
                        lambda x: proj(x),
                        gamma=10000,
                        variant=1,
                        dimension=len(x0),
                        sense="max",
                        x_init=x0
                        )
        elif SOLVER == "sg":
            method = SG(lambda x: (0, -obj(x)[0], -obj(x)[1]),
                        lambda x: proj(x),
                        # proj,
                        dimension=len(x0),
                        stepsize_0=1e-4,
                        stepsize_rule="1/k",
                        # stepsize_rule="1/sqrt(k)",
                        # stepsize_rule="constant",
                        sense="max",
                        x_init=x0
                        )
        elif SOLVER == "bundle":
            method = BM(lambda x: (0, obj(x)[0], obj(x)[1]),
                        lambda x: proj(x),
                        epsilon=0.001,
                        mu=1000,
                        dimension=len(x0),
                        sense="min",
                        x_init=x0
                        )

        logger = GenericMethodLogger(method)
        for iter in range(100000):
            method.step()
            x_k = logger.x_k_iterates[-1]
            fval_k, grad_k = obj(x_k)

            if VERBOSE:
                if iter % 1 == 0:
                    print(f"iter {iter:06d}",
                          f"fval {-fval_k:.2e}",
                          f"||g|| {norm(grad_k):.2e}")
            if fval_k <= 0:
                return {"fun": fval_k, "x": x_k}
        return {"fun": fval_k, "x": x_k}

    elif SOLVER == "bca":
        ''' block update on max variables'''
        U_var, R_var, s_var, alpha_var, gamma_var = process_var(x0, n)

        xk = x0

        def callback_U(x):
            fval, grad = obj(x, mask_all=False, mask_U=True, xk=xk)
            print(f"callback in U fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}")

        def callback_R(x):
            fval, grad = obj(x, mask_all=False, mask_R=True, xk=xk)
            print(f"callback in R fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}")

        def callback_s(x):
            fval, grad = obj(x, mask_all=False, mask_s=True, xk=xk)
            print(f"callback in s fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}")

        def callback_alpha(x):
            if not isinstance(x, float):
                x = x.item()
            fval, grad = obj(x, mask_all=False, mask_alpha=True, xk=xk)
            print(f"callback in alpha fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}")

        def callback_gamma(x):
            if not isinstance(x, float):
                x = x.item()
            fval, grad = obj(x, mask_all=False, mask_gamma=True, xk=xk)
            print(f"callback in gamma fval {-fval:<10.2e}",
                  f"||g|| {norm(grad):<10.2e}")

        for iter in range(1, 51):
            print(f"iter {iter:>03d} =======")

            ''' Optimize U'''
            try:
                if CHECK_GRAD:
                    grad_approx = approx_fprime(U_var.flatten(),
                                                lambda x: obj(x, mask_all=False, mask_U=True, xk=xk)[0],
                                                1e-8)
                    grad_true = obj(U_var, mask_all=False, mask_U=True, xk=xk)[1]
                    print(f"{'U:':<10s} ||g||_approx {norm(grad_approx):<10.2e}",
                          f"||g||_true {norm(grad_true):<10.2e}",
                          f"||g||_diff {norm(grad_approx - grad_true):<10.2e}")
                    exit()

                res_U = minimize(obj, U_var,
                                 args=(False, False, True, False, False, False, False, xk),
                                 callback=callback_U if VERBOSE else None,
                                 jac=True,
                                 options={"maxiter": 20})

                U_var = res_U['x'].reshape((n, -1))
                xk = assemble_var(U_var, R_var, s_var, alpha_var, gamma_var)
                if VERBOSE:
                    print(f"{'fval_U':<10s} {-res_U['fun']:>10.2e} status {res_U['status']}")
                if res_U['fun'] < 0:
                    return {'fun': res_U['fun'], 'x': xk}
            except Exception:
                print("error in opt_U")

            ''' Optimize R'''
            try:
                if CHECK_GRAD:
                    # NOTE: passed
                    grad_approx = approx_fprime(R_var.flatten(),
                                                lambda x: obj(x, mask_all=False, mask_R=True, xk=xk)[0],
                                                1e-8)
                    grad_true = obj(R_var, mask_all=False, mask_R=True, xk=xk)[1]
                    print(f"{'R:':<10s} ||g||_approx {norm(grad_approx):<10.2e}",
                          f"||g||_true {norm(grad_true):<10.2e}",
                          f"||g||_diff {norm(grad_approx - grad_true):<10.2e}")
                    exit()
                res_R = minimize(obj, R_var,
                                 args=(False, False, False, True, False, False, False, xk),
                                 callback=callback_R if VERBOSE else None,
                                 jac=True,
                                 options={"maxiter": 20})

                R_var = res_R['x'].reshape((n - 1, -1))
                xk = assemble_var(U_var, R_var, s_var, alpha_var, gamma_var)
                if VERBOSE:
                    print(f"{'fval_R':<10s} {-res_R['fun']:>10.2e} status {res_R['status']}")
                if res_R['fun'] < 0:
                    return {'fun': res_R['fun'], 'x': xk}
            except Exception:
                print("error in opt_R")

            ''' Optimize s'''
            try:
                if CHECK_GRAD:
                    # NOTE: passed
                    grad_approx = approx_fprime(s_var.flatten(),
                                                lambda x: obj(x, mask_all=False, mask_s=True, xk=xk)[0],
                                                1e-8)
                    grad_true = obj(s_var, mask_all=False, mask_s=True, xk=xk)[1]
                    print(f"{'s:':<10s} ||g||_approx {norm(grad_approx):<10.2e}",
                          f"||g||_true {norm(grad_true):<10.2e}",
                          f"||g||_diff {norm(grad_approx - grad_true):<10.2e}")
                    exit()

                res_s = minimize(obj, s_var,
                                 args=(False, False, False, False, True, False, False, xk),
                                 callback=callback_s if VERBOSE else None,
                                 jac=True,
                                 options={"maxiter": 20})

                s_var = res_s['x']
                xk = assemble_var(U_var, R_var, s_var, alpha_var, gamma_var)
                if VERBOSE:
                    print(f"{'fval_s':<10s} {-res_s['fun']:>10.2e} status {res_s['status']}")
                if res_s['fun'] < 0:
                    return {'fun': res_s['fun'], 'x': xk}
            except Exception:
                print("error in opt_s")

            ''' Optimize alpha'''
            try:
                if CHECK_GRAD:
                    # NOTE: passed
                    grad_approx = approx_fprime(alpha_var,
                                                lambda x: obj(x, mask_all=False, mask_alpha=True, xk=xk)[0],
                                                1e-8)
                    grad_true = obj(alpha_var, mask_all=False, mask_alpha=True, xk=xk)[1]
                    print(f"{'alpha:':<10s} ||g||_approx {norm(grad_approx):<10.2e}",
                          f"||g||_true {norm(grad_true):<10.2e}",
                          f"||g||_diff {norm(grad_approx - grad_true):<10.2e}")
                    exit()

                res_alpha = minimize(obj, alpha_var,
                                     method="L-BFGS-B",
                                     bounds=[(1e-3, None)],
                                     args=(False, False, False, False, False, True, False, xk),
                                     callback=callback_alpha if VERBOSE else None,
                                     jac=True,
                                     options={"maxiter": 20})

                alpha_var = res_alpha['x'].item()
                xk = assemble_var(U_var, R_var, s_var, alpha_var, gamma_var)
                if VERBOSE:
                    if isinstance(res_alpha['fun'], np.ndarray):
                        fval_alpha = -res_alpha['fun'].item()
                    else:
                        fval_alpha = -res_alpha['fun']
                    print(f"{'fval_alpha':<10s} {-fval_alpha:>10.2e} status {res_alpha['status']}")
                if res_alpha['fun'] < 0:
                    return {'fun': res_alpha['fun'], 'x': xk}
            except Exception:
                print("error in opt_alpha")

            ''' Optimize gamma'''
            try:
                if CHECK_GRAD:
                    # NOTE: passed
                    grad_approx = approx_fprime(gamma_var,
                                                lambda x: obj(x, mask_all=False, mask_gamma=True, xk=xk)[0],
                                                1e-8)
                    grad_true = obj(gamma_var, mask_all=False, mask_gamma=True, xk=xk)[1]
                    print(f"{'gamma:':<10s} ||g||_approx {norm(grad_approx):<10.2e}",
                          f"||g||_true {norm(grad_true):<10.2e}",
                          f"||g||_diff {norm(grad_approx - grad_true):<10.2e}")
                    exit()

                res_gamma = minimize(obj, gamma_var,
                                     method="L-BFGS-B",
                                     bounds=[(1e-3, None)],
                                     args=(False, False, False, False, False, False, True, xk),
                                     callback=callback_gamma if VERBOSE else None,
                                     jac=True,
                                     options={"maxiter": 20})

                gamma_var = res_gamma['x'].item()
                xk = assemble_var(U_var, R_var, s_var, alpha_var, gamma_var)
                if VERBOSE:
                    if isinstance(res_gamma['fun'], np.ndarray):
                        fval_gamma = -res_gamma['fun'].item()
                    else:
                        fval_gamma = -res_gamma['fun']
                    print(f"{'fval_gamma':<10s} {fval_gamma:>10.2e} status {res_gamma['status']}")
                if res_gamma['fun'] < 0:
                    return {'fun': res_gamma['fun'], 'x': xk}
            except Exception:
                print("error in opt_gamma")

            ''' end of alternating '''
            res = {'fun': res_gamma['fun'], 'x': xk}
    else:
        raise Exception(f"unknown solver '{SOLVER}'")

    return res
