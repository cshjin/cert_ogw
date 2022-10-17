import os
import os.path as osp
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.utils as pygutils
from torch_geometric.data import dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from fgw.dist import commute_time, cttil, shortest_path
from fgw.gwtil import gwtil_lb, gwtil_ub
from fgw.gwtil_cvx import omega_circ
from fgw.utils import random_perturb
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
ROOT = osp.join(osp.expanduser('~'), 'tmp', 'data', 'TUDataset')
# SAVED_PATH = osp.join(ROOT, "COX2", 'saved')
# if not osp.exists(SAVED_PATH):
# os.makedirs(SAVED_PATH)
# SAVED_FILE = osp.join(SAVED_PATH, "result.pkl")
# SAVED_FILE_v2 = osp.join(SAVED_PATH, "result.pt")
dataset = TUDataset(ROOT, "MUTAG")

# model = torch.load(SAVED_FILE_v2)
# dataloader = DataLoader(dataset[:1])

data = dataset[0]
nG = data.num_nodes
y = data.y.item()

G = pygutils.to_networkx(data, to_undirected=True)
A = pygutils.to_scipy_sparse_matrix(data.edge_index).toarray()
# dist = shortest_path
dist = cttil
C = dist(A)
# C = C / C.max()
p = np.ones(nG) / nG
lbs = defaultdict(list)
ubs = defaultdict(list)
for n in tqdm(range(0, 11), desc="budget"):
    for i in tqdm(range(20), desc="round", leave=False):
        A_ = random_perturb(A, n=n, seed=i)
        C_ = dist(A_)
        # C_ = C_ / C_.max()
        gwtil_lb_fval = gwtil_lb(C_, C)
        # omega = omega_circ(C_, C)
        # gwtil_ub_fval = omega_circ(C_, C)
        # print(f"gwtil_lb {gwtil_lb_fval:.4f} gwtil_ub {gwtil_ub_fval:.4f}")
        lbs[n].append(gwtil_lb_fval)
        # ubs[n].append(gwtil_ub_fval)
# exit()
fit = plt.figure(figsize=(4, 3))
xs = list(lbs.keys())
# ub_means = np.mean(list(ubs.values()), axis=1)
# ub_stds = np.std(list(ubs.values()), axis=1)
lb_means = np.mean(list(lbs.values()), axis=1)
lb_stds = np.std(list(lbs.values()), axis=1)
# NOTE: prevent negative in plots
# plt.fill_between(xs, np.maximum(0, ub_means - ub_stds), ub_means + ub_stds, alpha=0.2, color='#377eb8')
# plt.plot(xs, ub_means, color='#377eb8', label=r"$\widetilde{GW}_{ub}$")
plt.fill_between(xs, np.maximum(0, lb_means - lb_stds), lb_means + lb_stds, alpha=0.2, color='#e41a1c')
plt.plot(xs, lb_means, color='#e41a1c')
# plt.xticks(range(0, len(xs) + 1), xs)
plt.xlabel(r"$\delta_g$")
plt.ylabel(r"OGW")
plt.margins(0, 0)
# plt.legend()
plt.tight_layout()
plt.savefig("tmp.png")
# plt.show()

# plt.savefig("attack_cttil_COX2.png")
