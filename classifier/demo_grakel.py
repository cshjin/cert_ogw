""" Graph classification with graph kernels. """
import argparse
from grakel import WeisfeilerLehmanOptimalAssignment

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

if __name__ == "__main__":
    # np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="PTC_MR", help="dataset")
    parser.add_argument("--kernel", "-k", type=str, default="gk", help="graph kernel name")
    parser.add_argument("--debug", dest="debug", action="store_true")

    args = parser.parse_args()
    args = vars(args)
    ds_name = args['dataset']
    print("=" * 20)
    print(f"dataset {args['dataset']} kernel {args['kernel']}")

    dataset = fetch_dataset(args['dataset'], verbose=False)
    G, y = dataset.data, dataset.target

    ''' pipeline '''
    if args['debug']:
        if args['kernel'] == 'sp':
            gk = ShortestPath(normalize=True)
        elif args['kernel'] == 'rw':
            gk = RandomWalk(normalize=True)
        elif args['kernel'] == 'gk':
            gk = GraphletSampling(normalize=True, k=3)
        elif args['kernel'] == 'wl':
            gk = WeisfeilerLehman(normalize=True)
        # G_train, G_test, y_train, y_test = train_test_split(G, y, random_state=0)
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2)
        # pipe = Pipeline([("sp", ShortestPath(normalize=True)), ("svc", SVC(kernel="precomputed"))])
        pipe = Pipeline([(args['kernel'], gk), ("svc", SVC(kernel="precomputed"))])
        pipe.fit(G_train, y_train)
        y_pred = pipe.predict(G_test)
        print("accuracy ", accuracy_score(y_test, y_pred))
    else:
        ''' no pipeline '''
        # if args['kernel'] == 'sp':
        #     gk = ShortestPath(normalize=True, with_labels=False)
        # elif args['kernel'] == 'rw':
        #     gk = RandomWalk(normalize=True)
        # elif args['kernel'] == 'gk':
        #     gk = GraphletSampling(normalize=True, k=3)
        # elif args['kernel'] == 'wl':
        #     gk = WeisfeilerLehman(normalize=True)

        # DEBUG: handle the dataset without node label / feature
        gk = WeisfeilerLehman(normalize=True)

        K_trans = gk.fit_transform(G)
        clf = SVC(kernel="precomputed")
        scores = cross_val_score(clf, K_trans, y, cv=5)
        print(f"mean / std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
