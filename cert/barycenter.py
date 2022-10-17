import numpy as np
from ot.gromov import gromov_wasserstein
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from fgw.cttil_bary import grad_C_A, grad_C_T, grad_T_Z, solve_A_from_L_cvx
from fgw.dist import calc_T, cttil, cttil_Z
from fgw.graph import find_thresh, sp_to_adjacency
from fgw.gw_bary import (grad_gw_C, gromov_wasserstein_v2, optim_C_gw,
                         optim_C_gw_v2)
from fgw.gwtil import gwtil_lb, gwtil_ub
from fgw.gwtil_bary import (grad_gwtil_lb_C, grad_gwtil_ub_C, optim_C_gwtil_lb,
                            optim_C_gwtil_lb_lb, optim_C_gwtil_lb_lb_v2,
                            optim_C_gwtil_lb_v2, optim_C_gwtil_ub,
                            optim_C_gwtil_ub_v2)


class MethodError(Exception):
    pass


class Barycenter(object):
    def __init__(self, topo_metric="gw", dist_func="sp") -> None:
        super().__init__()
        self.topo_metric = topo_metric
        self.dist_func = dist_func

    def optim_C(self, N, Ds, ps, p, lambdas, method="BFGS", **kwargs):
        """ Optimize barycenter regardless of dist_func """
        assert len(Ds) == len(ps) and len(Ds) == len(lambdas)
        assert p.shape[0] == N

        if method == "BFGS":
            if self.topo_metric == "gw":
                res = optim_C_gw(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_ub":
                res = optim_C_gwtil_ub(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_lb":
                res = optim_C_gwtil_lb(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_lb_lb":
                res = optim_C_gwtil_lb_lb(N, Ds, ps, p, lambdas, **kwargs)

        elif method == "closed-form":
            if self.topo_metric == "gw":
                res = optim_C_gw_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_ub":
                res = optim_C_gwtil_ub_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_lb":
                res = optim_C_gwtil_lb_v2(N, Ds, ps, p, lambdas, **kwargs)

            elif self.topo_metric == "gwtil_lb_lb":
                res = optim_C_gwtil_lb_lb_v2(N, Ds, ps, p, lambdas, **kwargs)
        else:
            raise MethodError

        if isinstance(res, tuple):
            self.C = res[0]
        else:
            self.C = res
        return res

    def optim_A(self, N, Ds, As, ps, p, lambdas, log=False, **kwargs):
        """ only optimize when `dist_func=cttil` """
        assert self.dist_func == "cttil"

        S = len(Ds)
        Ts = [None] * S
        Ps = [None] * S
        Q1s = [None] * S
        Q2s = [None] * S
        gw_fval = [0] * S

        def obj(A):
            A = A.reshape((N, N))
            # A = sym(A)
            C = cttil(A)
            # C = cttil_Z(Z)
            # T_mat = calc_T(Z)

            for i in range(S):
                if self.topo_metric == "gw":
                    T, gwlog = gromov_wasserstein_v2(C, Ds[i], p, ps[i], log=True, T_init=Ts[i], **kwargs)
                    # T, gwlog = gromov_wasserstein(Ds[i], C, ps[i], p, loss_fun="square_loss", log=True)
                    Ts[i] = T
                    gw_fval[i] = gwlog['gw_dist'] * lambdas[i]
                # elif self.topo_metric == "gwtil_lb":
                #     _fval, Q1, Q2 = gwtil_lb(C, Ds[i], return_matrix=True, **kwargs)
                #     Q1s[i] = Q1
                #     Q2s[i] = Q2
                #     gw_fval[i] = _fval * lambdas[i]
                # elif self.topo_metric == "gwtil_ub":
                #     _fval, P = gwtil_ub(C, Ds[i], return_matrix=True, P_init=Ps[i], **kwargs)
                #     Ps[i] = P
                #     gw_fval[i] = _fval * lambdas[i]
            fval = sum(gw_fval)
            grad_ = 0
            for i in range(S):
                if self.topo_metric == "gw":
                    grad_ += lambdas[i] * np.einsum("ij, ijkl -> kl",
                                                    grad_gw_C(C, Ds[i], Ts[i]),
                                                    grad_C_A(A))
                # elif self.topo_metric == "gwtil_lb":
                #     grad_ += lambdas[i] * np.einsum("ij, ijkl, klpq -> pq",
                #                                     grad_gwtil_lb_C(C, Ds[i], Q1s[i], Q2s[i]),
                #                                     grad_C_T(T_mat),
                #                                     grad_T_Z(Z))
                # elif self.topo_metric == "gwtil_ub":
                #     grad_ += lambdas[i] * np.einsum("ij, ijkl, klpq -> pq",
                #                                     grad_gwtil_ub_C(C, Ds[i], Ps[i]),
                #                                     grad_C_T(T_mat),
                #                                     grad_T_Z(Z))
            return fval, grad_.flatten()

        def callback(A):
            fval, grad = obj(A)
            print(f"obj {fval:.4f} ||g|| {np.linalg.norm(grad):.4f}")

        if "A_init" in kwargs:
            A_init = kwargs['A_init']
        else:
            # ''' init with random P.S.D '''
            import networkx as nx

            # _A = np.random.rand(N, 2)
            # A_init = cdist(_A, _A)
            # A_init = np.ones((N, N)).flatten()
            # A_init = Zs[0]

            G = nx.path_graph(N)
            A_init = nx.adjacency_matrix(G).toarray()
        bnd = [(0, 1) for _ in range(N**2)]
        res = minimize(obj, A_init,
                       #    method="BFGS",
                       method="L-BFGS-B",
                       jac=True,
                       bounds=bnd,
                       callback=callback)
        A_opt = res['x'].reshape((N, N))
        return A_opt

    def optim_Z(self, N, Ds, Zs, ps, p, lambdas, log=False, **kwargs):
        """ only optimize when `dist_func=cttil` """
        assert self.dist_func == "cttil"

        S = len(Ds)
        Ts = [None] * S
        Ps = [None] * S
        Q1s = [None] * S
        Q2s = [None] * S
        gw_fval = [0] * S

        def obj(Z):
            Z = Z.reshape((N, N))
            C = cttil_Z(Z)
            T_mat = calc_T(Z)

            for i in range(S):
                if self.topo_metric == "gw":
                    # T, gwlog = gromov_wasserstein_v2(C, Ds[i], p, ps[i], log=True, T_init=Ts[i], **kwargs)
                    T, gwlog = gromov_wasserstein(Ds[i], C, ps[i], p, loss_fun="square_loss", log=True)
                    Ts[i] = T
                    gw_fval[i] = gwlog['gw_dist']
                elif self.topo_metric == "gwtil_lb":
                    _fval, Q1, Q2 = gwtil_lb(C, Ds[i], return_matrix=True, **kwargs)
                    Q1s[i] = Q1
                    Q2s[i] = Q2
                    gw_fval[i] = _fval * lambdas[i]
                elif self.topo_metric == "gwtil_ub":
                    _fval, P = gwtil_ub(C, Ds[i], return_matrix=True, P_init=Ps[i], **kwargs)
                    Ps[i] = P
                    gw_fval[i] = _fval * lambdas[i]
            fval = sum(gw_fval)

            grad_ = 0
            for i in range(S):
                if self.topo_metric == "gw":
                    grad_ += lambdas[i] * np.einsum("ij, ijkl, klpq -> pq",
                                                    # grad_gw_C(C, Ds[i], Ts[i]),
                                                    (2 * C / N**2 - 2 * Ts[i].T @ Ds[i] @ Ts[i]),
                                                    grad_C_T(T_mat),
                                                    grad_T_Z(Z))
                elif self.topo_metric == "gwtil_lb":
                    grad_ += lambdas[i] * np.einsum("ij, ijkl, klpq -> pq",
                                                    grad_gwtil_lb_C(C, Ds[i], Q1s[i], Q2s[i]),
                                                    grad_C_T(T_mat),
                                                    grad_T_Z(Z))
                elif self.topo_metric == "gwtil_ub":
                    grad_ += lambdas[i] * np.einsum("ij, ijkl, klpq -> pq",
                                                    grad_gwtil_ub_C(C, Ds[i], Ps[i]),
                                                    grad_C_T(T_mat),
                                                    grad_T_Z(Z))
            return fval, grad_.flatten()

        def callback(Z):
            fval, grad = obj(Z)
            print(f"obj {fval:.4f} ||g|| {np.linalg.norm(grad):.4f}")

        if "Z_init" in kwargs:
            Z_init = kwargs['Z_init']
        else:
            # ''' init with random P.S.D '''
            # _Z = np.random.rand(N, 2)
            # Z_init = cdist(_Z, _Z)

            Z_init = np.ones((N, N)).flatten()
            Z_init = Zs[0]

        res = minimize(obj, Z_init,
                       method="BFGS",
                       jac=True,
                       callback=callback)
        Z_opt = res['x'].reshape((N, N))

        # if self.topo_metric == "gw":
        #     # Z_opt = update_square_loss_gw(p, lambdas, Ts, Zs)
        #     Z_opt = update_square_loss(p, lambdas, Ts, Zs)
        # elif self.topo_metric == "gwtil_ub":
        #     Z_opt = update_square_loss_gwtil_ub(p, lambdas, Ps, Zs)
        # elif self.topo_metric == "gwtil_lb":
        #     Z_opt = update_square_loss_gwtil_lb(p, lambdas, Q1s, Q2s, Zs)

        if False:
            import matplotlib.pyplot as plt
            C_opt = cttil_Z(Z_opt)
            T_, log_ = gromov_wasserstein_v2(C_opt, Ds[0], p, ps[0], loss_fun="square_loss", log=True)
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
            plt.imshow(T_ @ Z_opt @ T_.T * N * N)
            plt.title("Z_opt after permutation")

            plt.subplot(2, 4, 5)
            plt.imshow(cttil_Z(T_ @ Z_opt @ T_.T * N * N))
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
            plt.imshow(T_ @ C_opt @ T_.T * N * N)
            plt.title("C_opt after permutation")

            plt.tight_layout()
            plt.show()

        if log:
            return Z_opt, {"fvals": None}
        else:
            return Z_opt

    def optim_A_from_Z(self, Z):
        N = Z.shape[0]
        Z = np.asarray(Z)
        L = np.linalg.pinv(Z) - np.ones((N, N)) / N

        A = solve_A_from_L_cvx(L)
        return A

    def optim_A_from_C(self, C):
        """ Solve A from barycenter using heuristic method """
        if self.dist_func == "sp":
            _up_bound = find_thresh(C, sup=C.max(), step=100, metric=self.dist_func)[0]
            A = sp_to_adjacency(C, threshinf=0, threshsup=_up_bound)
            return A
        elif self.dist_func == "cttil":
            _up_bound = find_thresh(C, inf=C.min(), sup=C.max() + 1e-7, step=100, metric=self.dist_func)[0]
            A = sp_to_adjacency(C, threshinf=0, threshsup=_up_bound)
            return A
