#!/usr/bin/bash

# virtual env
conda create -n tmp python=3.7 -y
conda activate tmp

# install packages
conda install pytorch=1.8.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric pot

# install subplementary
conda install matplotlib seaborn joblib networkx numba -y
pip install qpsolvers nsopy cvxpy

python setup.py develop