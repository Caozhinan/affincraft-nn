#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install requirements
pip install 'pip<24' ninja
conda install numpy==1.20.3 pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
# fix AttributeError: module 'numpy' has no attribute 'float'
pip install torch-geometric==1.7.2 \
            tensorboardX==2.4.1 ogb==1.3.2 \
            rdkit-pypi==2021.9.3 icecream \
            torchmetrics==0.9.2 tensorboard \
            setuptools==59.5.0 protobuf==3.20.1 \
            wandb Cython \
            numpy==1.20.3 scipy pandas
cd fairseq
# if fairseq submodule has not been checkouted, run:
# git submodule update --init --recursive
pip install .  # --use-feature=in-tree-build
python setup.py build_ext --inplace
cd ..
python setup_cython.py build_ext --inplace

