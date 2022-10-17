# Certifying Robust Graph Classification under Orthogonal Gromov-Wasserstein Threats

## Preparation

* create virtual env (restrict to python=3.7, which is compatible with CPLEX 12.10)
  `conda create -n gdro python=3.7`

* install pytorch / pytorch-geometric
  `conda install pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`
  `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html`
  `pip install torch-geometric pot`

* install supplimentary libraries
  `conda install matplotlib joblib networkx numba`
  `conda install gurobi -c gurobi`
  `pip install qpsolvers nsopy`

* install the package as develop mode
  `python setup.py develop`

* (optional) install additional solver if available.
  `conda install docplex cplex -c imbdecisionoptimization`
  `conda install mosek -c mosek`
  `conda install cvxopt -c conda-forge`

## dataset

| dataset | # graphs | # lables | # features | ave. edge | min edge | max edge | avg. node | min node | max node |
| ------- | -------- | -------- | ---------- | --------- | -------- | -------- | --------- | -------- | -------- |
| MUTAG   | 188      | 2        | 7          | 38        | 20       | 66       | 17.5      | 10       | 28       |
| PTC_MR  | 344      | 2        | 18         | 25.0      | 2        | 142      | 13.0      | 2        | 64       |
| COX2    | 467      | 2        | 38         | 86.0      | 68       | 118      | 41.0      | 32       | 56       |
| BZR     | 405      | 2        | 56         | 74.0      | 26       | 120      | 35.0      | 13       | 57       |


## Experiments

We provided a comprehensive notebook `demo.ipynb` to show the idea of

* tractable bounds of FGW
* convex extension of FGW
* certification and attack for task of graph classification

Other files includes:

* `demo_train.py`: build the model for certificate and attack.
* `demo_spla.py`: preprocess to generate the linear mapping matrix $\mathcal{A}$.
* `demo_certify.py`: complete setup for experiments of robust certifications.
* `demo_attack.py`: complete setup for experiments of attack.

## License

The project is under [MIT](./LICENSE) license.

## NOTE:

`scipy==1.7.3` in case the changes in source code made differences.
