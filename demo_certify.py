import argparse
import os
import os.path as osp
import pickle
import tempfile
import warnings
from email.policy import default

import numpy as np
import torch
from gdro.optim.cvx_env_solver import cvx_env_solver
from gdro.utils import cal_logits, margin, process_data
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.special import softmax
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from tqdm import tqdm

from fgw.certificate.robust_cert import cttil_cvx_env_solver
from fgw.dist import cttil

# from fgw.certificate.robust_cert_v4 import cttil_cvx_env_solver


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    np.set_printoptions(3)

    # DATASETS = ['MUTAG', '']
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
    parser.add_argument('--solver', default='bfgs', type=str, help='Specify the solver.')
    parser.add_argument('--gamma', default=0, type=float, help='Specify the gamma in FGW')
    parser.add_argument('--delta_l', '-l', default=1, type=int, help='Specify local budget')
    parser.add_argument('--delta_g', '-g', default=10, type=int, help='Specify global budget')
    parser.add_argument('--delta_omega', default=.1, type=float, help='Specify fgw budget')
    parser.add_argument('--autograd', action='store_true', help='Toggle to use autograd in bidual.')
    parser.add_argument('--margin_solver', default='bfgs', type=str, help='Spcify the margin_star solver.')
    parser.add_argument('--mag', default=0.0001, type=float, help='scalar of init.')
    parser.add_argument('--debug', action='store_true', help='Toggle to debug mode.')
    parser.add_argument('--debug_ns', action='store_true', help='Toggle to debug ns conjugate function')
    parser.add_argument('--check_grad', action='store_true', help='Toggle to check grad.')
    # REVIEW robust training
    parser.add_argument('--robust', action='store_true', help='Specify robust training')
    # parser.add_argument('--min', default=0, help='Minimum local budget')
    # parser.add_argument('--train_gamma', default=0.01, type=float, help='Specify the weight the regularizer')
    # parser.add_argument('--adv_gamma', default=0.1, type=float, help='Specify the adv gamma value')

    # process args
    args = parser.parse_args()
    args = vars(args)

    ds_name = args['dataset']
    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'TUDataset')
    SAVED_PATH = osp.join(ROOT, ds_name, 'saved')
    if not osp.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)
    SAVED_FILE = osp.join(SAVED_PATH, "result_robust.pkl") if args['robust'] else osp.join(SAVED_PATH, "result.pkl")

    # ==========================================================================
    # prepare dataset
    max_deg = {"IMDB-BINARY": 135, "IMDB-MULTI": 88}
    # no node feature
    if ds_name in ['IMDB-BINARY', 'IMDB-MULTI']:
        # add one hot feature
        dataset = TUDataset(ROOT, name=ds_name,
                            transform=T.OneHotDegree(max_deg[ds_name]))
    else:
        dataset = TUDataset(ROOT, name=ds_name, use_node_attr=True)
    dataset = dataset.shuffle()

    # REVIEW:
    # dataset = dataset.shuffle()
    train_size = int(len(dataset) * args['train_size'])
    val_size = int(len(dataset) * 0.2)
    # tracking the graph
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
    W = checkpoint['conv.weight'].numpy().astype(np.float64)
    U = checkpoint['lin.weight'].numpy().astype(np.float64)

    corr_count = 0
    succ_count = 0
    cert_count = 0
    offset = train_size + val_size
    # REVIEW: order of the dataset affect the offline Acal generation

    pbar = tqdm(test_dataset[:100], desc="Certify {} with δg {} δΩ {}".format(ds_name, DELTA_G, DELTA_OMEGA))
    gamma = args['gamma']
    for gid, g in enumerate(pbar):
        # for gid, g, in enumerate(test_dataset[:100]):

        gid = gid + offset
        graph = dataset[gid]

    # for _ in range(1):
    #     graph = dataset[117]
        X = graph.x.numpy()
        M = cdist(X, X, metric="sqeuclidean")
        XW = (graph.x @ W).numpy()
        A = to_scipy_sparse_matrix(graph.edge_index)
        A = A.astype(np.float64)
        deg = A.toarray().sum(1)
        strength = 3
        nG = A.shape[0]

        # Acal.data = np.nan_to_num(Acal.data, posinf=nG, neginf=-nG)
        logits_0 = softmax(cal_logits(A, XW, U))
        y = graph.y
        # D0 = shortest_path(A.toarray())
        D0 = cttil(A.toarray())

        if args['delta_l'] == -1:
            delta_l = np.array([nG] * nG)
        else:
            delta_l = np.array([args['delta_l']] * nG)
        DELTA_G = min(args['delta_g'], int((nG - 1) * nG / 2))
        # delta_l = np.array([min(nG, DELTA_G)] * nG)

        # TODO: iter all labels
        # only attack graph which classified correctly
        if np.argmax(logits_0) == y:
            corr_count += 1

            cvx_env_params = dict(
                autograd=args['autograd'],
                iter=300,
                lr=0.3,
                verbose=args['verbose'],
                constr='2',
                act='linear',
                algo='swapping',
                solver=args['solver'],
                x_solver='lp',
                margin_solver=args['margin_solver'],
                mag=args['mag'],
                debug=args['debug'],
                debug_ns=args['debug_ns'],
                check_grad=args['check_grad'],
                nonsmooth_init='subgrad')

            cvx_env_sol = cttil_cvx_env_solver(A.toarray(), D0, XW, (U[y] - U[1 - y]) / nG,
                                               delta_l, DELTA_G, DELTA_OMEGA, **cvx_env_params)

            if -cvx_env_sol['fun'] >= 0:
                cert_count += 1

        pbar.set_postfix({"cert": cert_count, "corr": corr_count})

    print(f"Dataset {args['dataset']}",
          f"Robust {args['robust']}",
          f"DELTA_L {args['delta_l']:d}",
          f"DELTA_G {DELTA_G:02d}",
          f"DELTA_OMEGA {DELTA_OMEGA:.2f}",
          f"cert_num {cert_count:d}",
          f"cert_rate {cert_count / corr_count:.4f}")
    # print("DELTA_G {:02d} DELTA_OMEGA {:.2f} cert_num {:d} cert_rate {:.4f}".format(
    #     args['delta_g'], args['delta_omega'], cert_count, cert_count / corr_count))
