""" Train the base GNN model for graph classification.
"""

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T
import torch_geometric.utils as pygutils
from fgw.mnist_2d_bary import process_graph
from gdro.model.gnn import GNN, eval, train
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from gdro.attacker.greedy_gw import IterAttacker


def train_model():
    """ Train the graph classification model. """
    # prepare dataloader
    best = 0
    pbar = tqdm(range(args['epoch']), desc="Train")
    for epoch in pbar:
        if epoch <= args['epoch'] // 2:
            train_loss = train(model, loader=train_loader)
        # REVIEW: attack the graphs
        if args['robust'] and epoch > args['epoch'] // 2:
            _W = model.conv.weight.detach().cpu().numpy()
            _U = model.lin.weight.detach().cpu().numpy()

            for graph_idx, loader in enumerate(train_loader):
                # attack one graph
                idx = np.random.choice(len(train_dataset))
                if idx == graph_idx:
                    XW = Xs[graph_idx] @ _W
                    g_att = IterAttacker(As[graph_idx], XW, _U, y=ys[graph_idx],
                                         delta_l=np.ones(Xs[graph_idx].shape[0]),
                                         delta_g=args['delta_g'],
                                         delta_omega=args['delta_omega'],
                                         M=np.ones((Ss[graph_idx], Ss[graph_idx])),
                                         gamma=1,
                                         domain="OE",
                                         verbose=True)
                    ATs, _, _ = g_att.attack_parallel()
                    A = ATs[-1]
                    # update dataloader with new adj
                    loader.edge_index = torch.tensor([list(A.nonzero()[0]), list(A.nonzero()[1])], dtype=torch.int64)
            # train with updated data_loader
            train_loss = train(model, loader=train_loader)
        train_acc = eval(model, train_loader)
        val_acc = eval(model, val_loader)
        if val_acc >= best:
            best = val_acc
            torch.save(model.state_dict(), SAVED_FILE)
            torch.save(model, SAVED_FILE_v2)

        pbar.set_postfix({'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
    test_acc = eval(model, test_loader, testing=True, save_path=SAVED_PATH, robust=args['robust'])

    print("Testing accuracy {:.4f}".format(test_acc))


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

    # REVIEW robust training
    parser.add_argument('--robust', action='store_true', help='Specify robust training')
    # parser.add_argument('--min', default=0, help='Minimum local budget')
    parser.add_argument('--delta_g', '-g', default=10, type=int, help='Specify global budget')
    parser.add_argument('--delta_omega', default=1, type=float, help='Specify gwtil_ub budget')
    # parser.add_argument('--train_gamma', default=0.01, type=float, help='Specify the weight the regularizer')
    # parser.add_argument('--adv_gamma', default=0.1, type=float, help='Specify the adv gamma value')

    # process args
    args = vars(parser.parse_args())

    ''' process invalid args '''
    if not args['robust']:
        args['delta_g'] = -1
        # args['strength'] = -1
        # args['min'] = -1
        # args['delta_omega'] = -1
        # args['train_gamma'] = -1
        # args['adv_gamma'] = -1

    ds_name = args['dataset']
    ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', "TUDataset")
    SAVED_PATH = osp.join(ROOT, ds_name, 'saved')
    if not osp.exists(SAVED_PATH):
        os.makedirs(SAVED_PATH)

    # file to save model_dict()
    if not args['robust']:
        SAVED_FILE = osp.join(SAVED_PATH, "result.pkl")
    else:
        SAVED_FILE = osp.join(SAVED_PATH, "result_robust.pkl")

    # file to save model
    if not args['robust']:
        SAVED_FILE_v2 = osp.join(SAVED_PATH, "result.pt")
    else:
        SAVED_FILE_v2 = osp.join(SAVED_PATH, "result_robust.pt")

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

    ''' NOTE: check max degree of IMDB-BINARY dataset '''
    # max_deg = 0
    # for data in dataset:
    #     G = pygutils.to_networkx(data, to_undirected=True)
    #     _deg = max(dict(G.degree).values())
    #     if _deg > max_deg:
    #         max_deg = _deg
    # print(max_deg)
    # exit()

    train_size = int(len(dataset) * args['train_size'])
    val_size = int(len(dataset) * 0.2)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size: train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # verbose output of information from dataset
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

    # create model
    model = GNN(hidden=args['hidden'],
                n_features=dataset.num_features,
                n_classes=dataset.num_classes,
                act=args['act'],
                pool='avg',
                dropout=args['dropout'])
    # build data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # prepare for attackers
    # Gs = [pygutils.to_networkx(data, to_undirected=True) for data in train_dataset]
    As = [pygutils.to_scipy_sparse_matrix(data.edge_index) for data in train_dataset]
    Xs = [data.x.numpy() for data in train_dataset]
    ys = [data.y.item() for data in train_dataset]
    Ss = [X.shape[0] for X in Xs]
    # attack
    train_model()
