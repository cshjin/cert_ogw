import argparse
import os
import os.path as osp
import pickle
from tkinter.filedialog import SaveFileDialog
import warnings

import numpy as np
import torch
import torch_geometric.transforms as T
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.special import softmax
from torch_geometric.loader import DataLoader
# from torch_geometric.data.makedirs import makedirs
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from tqdm import tqdm

from gdro.attacker.greedy_gw import IterAttacker
from gdro.model.gnn import GNN, eval, train
from gdro.optim.omega import Omega_solver
from gdro.utils import cal_logits, margin, process_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    # for reproducing
    torch.manual_seed(0)
    np.random.seed(0)

    ACTS = ['linear', 'relu']

    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='MUTAG', help='Specify Dataset')
    parser.add_argument('--act', '-a', default='linear', help='Specify activation', choices=ACTS)
    parser.add_argument('--hidden', default=64, type=int, help='Specify hidden layer size')
    parser.add_argument('--train_size', default=0.3, type=float, help='Specify train size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Specify learning rate')
    parser.add_argument('--epoch', '-e', default=200, type=int, help='Specify number of epochs')
    parser.add_argument('--dropout', default=0.5, type=float, help='Specify dropout rate')
    parser.add_argument('--verbose', '-v', action='store_true', help='Toggle verbose output')
    parser.add_argument('--clean', '-c', action='store_true', help='Toggle clean training')
    parser.add_argument('--gamma', default=0.5, type=float, help="Specify the gamma in FGW")
    parser.add_argument('--delta_l', '-l', default=1, type=int, help="Specify local budget")
    parser.add_argument('--delta_g', '-g', default=10, type=int, help='Specify global budget')
    parser.add_argument('--delta_omega', default=1, type=float, help='Specify fgw budget')

    # REVIEW robust training
    parser.add_argument('--robust', action='store_true', help='Specify robust training')
    parser.add_argument('--plot', action='store_true', help='Plot graphs')
    # parser.add_argument('--min', default=0, help='Minimum local budget')
    # parser.add_argument('--train_gamma', default=0.01, type=float, help='Specify the weight the regularizer')
    # parser.add_argument('--adv_gamma', default=0.1, type=float, help='Specify the adv gamma value')

    # process args
    args = vars(parser.parse_args())

    ds_name = args['dataset']
    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'TUDataset')
    SAVED_PATH = osp.join(ROOT, ds_name, 'saved')
    if not osp.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)
    SAVED_FILE = osp.join(SAVED_PATH, "result_robust.pkl") if args['robust'] else osp.join(SAVED_PATH, "result.pkl")
    # SAVED_FILE_v2 = osp.join(SAVED_PATH, "result.pt")
    # scalers = pickle.load(open("scalers.pkl", "rb"))

    # ==========================================================================
    # prepare dataset
    max_deg = {"IMDB-BINARY": 135, "IMDB-MULTI": 88}
    if ds_name in ["IMDB-BINARY", "IMDB-MULTI"]:
        dataset = TUDataset(ROOT, name=ds_name,
                            transform=T.OneHotDegree(max_deg[ds_name]))
    else:
        dataset = TUDataset(ROOT, name=ds_name, use_node_attr=True)
    dataset = dataset.shuffle()

    train_size = int(len(dataset) * args['train_size'])
    val_size = int(len(dataset) * 0.2)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    if args['verbose']:
        edges = [0] * len(dataset)
        nodes = [0] * len(dataset)
        labels = [0] * len(dataset)
        for i, data in enumerate(dataset):
            edges[i] = data.num_edges
            nodes[i] = data.num_nodes
            labels[i] = data.y.item()
        print("-" * 80)
        print(f"dataset                 {args['dataset']}")
        print(f"# of total graphs       {len(dataset)}")
        print(f'# of train graphs:      {train_size}')
        print(f'# of valid graphs:      {val_size}')
        print(f'# of test graphs:       {len(test_dataset)}')
        print(f"# of features           {data.num_features}")
        print(f"# of labels             {dataset.num_classes}")
        print(f"avg edges               {np.median(edges):.1f}")
        print(f"min edges               {np.min(edges)}")
        print(f"max edges               {np.max(edges)}")
        print(f"avg node                {np.median(nodes):.1f}")
        print(f"min node                {np.min(nodes)}")
        print(f"max node                {np.max(nodes)}")
        print("-" * 80)
    # exit()
    # create model
    # model = GNN(hidden=args['hidden'],
    #             n_features=dataset.num_features,
    #             n_classes=dataset.num_classes,
    #             act=args['act'],
    #             pool='avg',
    #             dropout=args['dropout'])

    # Acals = pickle.load(open(osp.join(SAVED_PATH, "{}_Acal.pkl".format(ds_name)), 'rb'))

    DELTA_G = args['delta_g']
    DELTA_OMEGA = args['delta_omega']
    checkpoint = torch.load(SAVED_FILE)
    # model = torch.load(SAVED_FILE_v2)
    W = checkpoint['conv.weight'].numpy()
    U = checkpoint['lin.weight'].numpy()

    corr_count = 0
    succ_count = 0
    offset = train_size + val_size
    pbar = tqdm(test_dataset[:100], desc="Attack {} with δg {} δΩ {}".format(ds_name, DELTA_G, DELTA_OMEGA))
    gamma = args['gamma']
    record_FC = {}
    record_GW = {}
    for gid, g in enumerate(pbar):
        gid = gid + offset
        graph = dataset[gid]
        y = graph.y
        # Acal = Acals[gid]
        X = graph.x.numpy()
        M = cdist(X, X, metric="sqeuclidean")
        XW = (X @ W)
        # XW = torch.matmul(graph.x, W).numpy()
        A = to_scipy_sparse_matrix(graph.edge_index)
        nG = A.shape[0]
        DELTA_L = np.ones(nG) * args['delta_l']
        # REVIEW:
        # Acal.data = np.nan_to_num(Acal.data, posinf=nG, neginf=-nG)
        logits_0 = softmax(cal_logits(A, XW, U))
        margin_0 = margin(logits_0, y)
        # D0 = shortest_path(A.toarray())
        P = np.identity(nG)
        # only attack graph which classified correctly

        # plot #
        # g_att = IterAttacker(A, XW, U, y, DELTA_L, DELTA_G, DELTA_OMEGA, M, gamma,
        #                      domain="OE", verbose=True)
        # data = pickle.load(open("attack_res.pkl", "rb"))
        # g_att.draw_curve(142, [0.35] + data["FC"][142], [0] + data["GW"][142], SAVED_PATH="./visualize/attack/")
        # exit()
        if np.argmax(logits_0) == y:
            # g_att.draw_curve(gid, data["FC"][gid], data["GW"][gid], SAVED_PATH="./visualize/attack/")
            # continue
            corr_count += 1
            g_att = IterAttacker(A, XW, U, y, DELTA_L, DELTA_G, DELTA_OMEGA, M, gamma,
                                 domain="OE", verbose=True)
            ATs, FCs, GWs = g_att.attack_parallel()
            # DEBUG:
            # ATs, FCs, GWs = g_att.attack()
            # logits_1 = softmax(cal_logits(A_new, XW, U))
            # print(GWs)
            X = ATs[-1] - A
            # AX = (Acal @ X.reshape((-1, 1))).reshape((nG, -1)).toarray()
            # omega_circ = Omega_solver(D0, AX, M, gamma=1)
            # print(omega_circ)
            # if margin(logits_1, y) < 0:
            #     succ_count += 1
            if min(FCs) < 0:
                succ_count += 1
                # # if FCs[-1] < 0:
                # #     succ_count += 1
                if args['plot']:
                    idx = np.argmin(FCs)
                    A_new = ATs[idx]
                    logits_1 = softmax(cal_logits(A_new, XW, U))
                    g_att.draw_curve(gid, [margin_0] + FCs, [0] + GWs, SAVED_PATH)
                    g_att.draw_graph(gid, A_new, logits_0, logits_1, SAVED_PATH)
            record_FC[gid] = FCs
            record_GW[gid] = GWs
        pbar.set_postfix({"succ": succ_count, "corr": corr_count})

    # print("DELTA_G {:02d} DELTA_OMEGA {:.2f} GAMMA {} Certified Non-robust {:.4f}".format(DELTA_G,
    #                                                                                       DELTA_OMEGA, gamma, succ_count / corr_count))
    # REVIEW: output attack results from 1 to delta_g
    pickle.dump({"FC": record_FC, "GW": record_GW}, open("attack_res.pkl", "wb"))
    for i in range(1, DELTA_G + 1):
        succ_count = 0
        for x in record_FC:
            if min(record_FC[x][:i]) < 0:
                succ_count += 1

        print(f"Dataset {args['dataset']}",
              f"Robust {args['robust']}",
              f"DELTA_L {args['delta_l']:d}",
              f"DELTA_G {i:02d}",
              f"DELTA_OMEGA {DELTA_OMEGA:.2f}",
              f"att_num {succ_count:d}",
              #   f"corr_num {corr_count:d}",
              f"att_rate {succ_count / corr_count:.4f}"
              )
