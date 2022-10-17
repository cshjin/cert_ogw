import argparse

import numpy as np
from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath
from grakel.kernels.graphlet_sampling import GraphletSampling
from grakel.kernels.random_walk import RandomWalk
from grakel.kernels.weisfeiler_lehman import WeisfeilerLehman
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from grakel import Graph

if __name__ == "__main__":
    np.random.seed(0)
    H2O_adjacency = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
    H2O_node_labels = {0: 'O', 1: 'H', 2: 'H'}
    H2O = Graph(initialization_object=H2O_adjacency, node_labels=H2O_node_labels)

    H3O_adjacency = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    H3O_node_labels = {0: 'O', 1: 'H', 2: 'H', 3: 'H'}
    H3O = Graph(initialization_object=H3O_adjacency, node_labels=H3O_node_labels)

    # kernel
    sp_kernel = ShortestPath(normalize=True)
    print(sp_kernel.fit_transform([H2O, H3O]))
    print(sp_kernel.transform([H3O]))
