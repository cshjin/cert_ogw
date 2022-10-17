import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from fgw.barycenter import Barycenter
from fgw.data_loader import build_noisy_circular_graph
from fgw.dist import cttil, shortest_path
from fgw.gwtil_bary import eigen_projection
import argparse

if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo_metric", "-t", type=str, default="gw")
    parser.add_argument("--dist_func", "-d", type=str, default="sp")
    parser.add_argument("--size", "-s", type=int, default=30)
    parser.add_argument("--noise", action="store_true", dest="noise")
    parser.add_argument("--single", action="store_true", dest="single")
    parser.add_argument("--verbose", action="store_true", dest="verbose")
    args = parser.parse_args()
    args = vars(args)

    N = args['size']
    ''' single graph '''
    if args["single"]:
        S = 1
        Gs = [build_noisy_circular_graph(
            9,
            with_noise=False,
            structure_noise=args['noise'],
            p=3) for _ in range(S)]
    else:
        ''' multiple graphs '''
        S = 8
        Gs = [build_noisy_circular_graph(np.random.randint(15, 25),
                                         with_noise=False,
                                         structure_noise=args["noise"],
                                         p=4) for _ in range(S)]

    ''' plot samples '''
    fig = plt.figure(figsize=(4, 2), tight_layout=True)
    for i in range(8):
        plt.subplot(2, 4, 1 + i)
        nx.draw(Gs[i].nx_graph, pos=nx.kamada_kawai_layout(Gs[i].nx_graph), node_size=30)
    plt.show()
    # exit()

    As = [nx.adjacency_matrix(G.nx_graph).toarray() for G in Gs]
    Ls = [nx.laplacian_matrix(G.nx_graph) for G in Gs]
    # Zs = [np.linalg.inv(Ls[i] + np.ones(Ls[i].shape) / Ls[i].shape[0]) for i in range(len(Ls))]

    Ds = [shortest_path(A) for A in As]
    # if args['dist_func'] == "sp":
    #     Ds = [shortest_path(A) for A in As]
    # elif args['dist_func'] == "cttil":
    #     Ds = [cttil(A) for A in As]

    ps = [np.ones(Ls[i].shape[0]) / Ls[i].shape[0] for i in range(len(Ls))]
    lambdas = np.ones(len(Gs)) / len(Gs)
    p = np.ones(N) / N
    # bary = Barycenter(topo_metric="gw", dist_func="sp")
    # bary = Barycenter(topo_metric="gwtil_lb_lb", dist_func="sp")
    # bary = Barycenter(topo_metric="gwtil_ub", dist_func="sp")

    bary = Barycenter(topo_metric=args["topo_metric"], dist_func=args["dist_func"])

    C, log = bary.optim_C(N, Ds, ps, p, lambdas, log=True, method="closed-form", verbose=args['verbose'])

    # NOTE: addition projection for gwtil
    if bary.topo_metric != "gw":
        C = sum([eigen_projection(C, Ds[i]) * lambdas[i] for i in range(len(Ds))])

    ''' plotting '''
    # fig = plt.figure(figsize=(3 * S, 6))
    # for i in range(S):
    #     plt.subplot(2, S, i + 1)
    #     plt.imshow(Ds[i])
    #     plt.colorbar()
    #     plt.subplot(2, S, i + 1 + S)
    #     nx.draw(Gs[i].nx_graph, pos=nx.kamada_kawai_layout(Gs[i].nx_graph), node_size=50)
    # # plt.savefig("graphs.png")
    # plt.show()

    # exit()

    # exit()
    fig = plt.figure(figsize=(4, 2), tight_layout=True)
    A = bary.optim_A_from_C(C)
    ''' for cttil only '''
    # A_opt = bary.optim_A(N, Ds, As, ps, p, lambdas, log=True, verbose=False)
    # C = shortest_path(A)
    # print(log['fun'], gwtil_lb(C, Ds[0]))
    G = nx.from_numpy_array(A)
    plt.subplot(1, 2, 1)
    plt.imshow(C)
    plt.colorbar()
    plt.title("Barycenter")
    # plt.subplot(2, 2, 2)
    # plt.imshow(Ds[0])
    # plt.colorbar()
    # plt.title("Sample 0")

    plt.subplot(1, 2, 2)
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_size=50)
    # plt.subplot(2, 2, 4)
    # nx.draw(Gs[0].nx_graph, pos=nx.kamada_kawai_layout(Gs[0].nx_graph), node_size=50)
    # plt.savefig("bary.png")
    plt.show()
