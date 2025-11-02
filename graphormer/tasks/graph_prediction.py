# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from importlib.metadata import metadata
import logging
import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
import importlib
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion
import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
    FairseqDataset
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
# from ..data.affincraft_dataset import LMDBAffinCraftDataset 
from ..data.dataset import EpochShuffleDataset


from ..data.affincraft_dataset import (
    AffinCraftDataset,
    OptimizedBatchedLazyAffinCraftDataset,
    affincraft_collator,
    LMDBAffinCraftDataset,
)

import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
import math

from ..data import DATASET_REGISTRY
import sys
import os
import glob
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    dataset_name: str = field(
        default="pcqm4m",
        metadata={"help": "name of the dataset"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=128,
        metadata={"help": "max nodes per graph"},
    )

    dataset_source: str = field(
        default="pyg",
        metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles, affincraft"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    seed: int = II("common.seed")

    pretrained_model_name: str = field(
        default="none",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    train_epoch_shuffle: bool = field(
        default=True,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )

    user_data_dir: str = field(
        default="",
        metadata={"help": "path to the module of user-defined dataset"},
    )

    data_path: str = field(
        default="",
        metadata={"help": "Custom place to store data"}
    )

    train_pkl_pattern: str = field(
        default="",
        metadata={"help": "glob pattern for training PKL files (e.g., /path/to/train/*.pkl)"}
    )

    valid_pkl_pattern: str = field(
        default="",
        metadata={"help": "glob pattern for validation PKL files (e.g., /path/to/valid/*.pkl)"}
    )

    test_pkl_pattern: str = field(
        default="",
        metadata={"help": "glob pattern for test PKL files (e.g., /path/to/test/*.pkl)"}
    )

    merged_pkl_file: str = field(
        default="",
        metadata={"help": "Path to single PKL file containing all complexes"}
    )

    train_pkl_objects: int = field(
        default=10000,
        metadata={"help": "Number of objects in training PKL file"}
    )

    valid_pkl_objects: int = field(
        default=1000,
        metadata={"help": "Number of objects in validation PKL file"}
    )

    test_pkl_objects: int = field(
        default=0,
        metadata={"help": "Number of objects in test PKL file"}
    )

    train_pkl_index: str = field(
        default="",
        metadata={"help": "Path to training PKL index file"}
    )

    valid_pkl_index: str = field(
        default="",
        metadata={"help": "Path to validation PKL index file"}
    )

    test_pkl_index: str = field(
        default="",
        metadata={"help": "Path to test PKL index file"}
    )


class AffinCraftDatasetWrapper:
    """包装AffinCraftDataset以兼容GraphormerDataset接口"""

    def __init__(self, train_pkl_files=None, valid_pkl_files=None, test_pkl_files=None,
                 train_complexes=None, valid_complexes=None, test_complexes=None,
                 is_merged=False, seed=0):
        self.seed = seed

        if is_merged:
            self.train_complexes = train_complexes or []
            self.valid_complexes = valid_complexes or []
            self.test_complexes = test_complexes or []

            self.dataset_train = AffinCraftDataset(self.train_complexes, is_merged=True) if self.train_complexes else None
            self.dataset_val = AffinCraftDataset(self.valid_complexes, is_merged=True) if self.valid_complexes else None
            self.dataset_test = AffinCraftDataset(self.test_complexes, is_merged=True) if self.test_complexes else None
        else:
            self.train_pkl_files = train_pkl_files or []
            self.valid_pkl_files = valid_pkl_files or []
            self.test_pkl_files = test_pkl_files or []

            self.dataset_train = AffinCraftDataset(self.train_pkl_files) if self.train_pkl_files else None
            self.dataset_val = AffinCraftDataset(self.valid_pkl_files) if self.valid_pkl_files else None
            self.dataset_test = AffinCraftDataset(self.test_pkl_files) if self.test_pkl_files else None

        self.train_idx = list(range(len(self.dataset_train))) if self.dataset_train else []
        self.valid_idx = list(range(len(self.dataset_val))) if self.dataset_val else []
        self.test_idx = list(range(len(self.dataset_test))) if self.dataset_test else []


class AffinCraftBatchedDataDataset(FairseqDataset):
    """专门为AffinCraft数据设计的批处理数据集"""

    def __init__(self, dataset, max_node=512):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node

    def __getitem__(self, index):
        item = self.dataset[index]
        if item is None:
            return None
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        result = affincraft_collator(samples, max_node=self.max_node)
        return result


class AffinCraftTargetDataset(FairseqDataset):  
    """专门为AffinCraft数据设计的目标数据集"""  
  
    def __init__(self, dataset):  
        super().__init__()  
        self.dataset = dataset  
  
    def __getitem__(self, index):  
        item = self.dataset[index]  
        # 处理 None 或标记为跳过的样本  
        if item is None or item.get('_skip', False):  
            return None  
        # 检查 'pk' 键是否存在  
        if 'pk' not in item:  
            return None  
        return torch.tensor([item['pk']], dtype=torch.float)  
  
    def __len__(self):  
        return len(self.dataset)  
  
    def collater(self, samples):  
        # 过滤掉 None 样本  
        samples = [s for s in samples if s is not None]  
        if not samples:  
            return torch.tensor([], dtype=torch.float)  
        return torch.stack(samples, dim=0)


@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """Graph prediction (classification or regression) task."""

    def __init__(self, cfg):
        super().__init__(cfg)

        if cfg.dataset_source == "affincraft":
            self._setup_affincraft_dataset(cfg)
        elif cfg.user_data_dir != "":
            self.__import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = GraphormerDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    data_path=dataset_dict["data_path"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    seed=cfg.seed)
            else:
                raise ValueError(f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}")
        else:
            self.dm = GraphormerDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                data_path=cfg.data_path,
                seed=cfg.seed,
            )

    def _setup_affincraft_dataset(self, cfg):  
        """设置AffinCraft数据集 - 支持LMDB格式"""  
        try:  
            dataset_train = None  
            dataset_val = None  
            dataset_test = None  

            # 检测是LMDB还是PKL格式  
            def is_lmdb_path(path):  
                return path and (path.endswith('.lmdb') or Path(path).is_dir())  

            # 训练集  
            if cfg.train_pkl_pattern:  
                if is_lmdb_path(cfg.train_pkl_pattern):  
                    # 使用LMDB数据集   
                    dataset_train = LMDBAffinCraftDataset(  
                        cfg.train_pkl_pattern,  
                        readonly=True  
                    )  
                    logger.info(f"训练数据集(LMDB)加载完成，{len(dataset_train)} 个样本")  
                elif os.path.exists(cfg.train_pkl_pattern):  
                    # 使用原有的PKL数据集  
                    dataset_train = OptimizedBatchedLazyAffinCraftDataset(  
                        cfg.train_pkl_pattern,  
                        batch_size=16,  
                        total_objects=getattr(cfg, 'train_pkl_objects', None),  
                        index_file_path=getattr(cfg, 'train_pkl_index', None) if getattr(cfg, 'train_pkl_index', '') else None  
                    )  
                    logger.info(f"训练数据集(PKL)加载完成，{len(dataset_train)} 个样本")  

            # 验证集  
            if cfg.valid_pkl_pattern:  
                if is_lmdb_path(cfg.valid_pkl_pattern):  
                    dataset_val = LMDBAffinCraftDataset(  
                        cfg.valid_pkl_pattern,  
                        readonly=True  
                    )  
                    logger.info(f"验证数据集(LMDB)加载完成，{len(dataset_val)} 个样本")  
                elif os.path.exists(cfg.valid_pkl_pattern):  
                    dataset_val = OptimizedBatchedLazyAffinCraftDataset(  
                        cfg.valid_pkl_pattern,  
                        batch_size=16,  
                        total_objects=getattr(cfg, 'valid_pkl_objects', None),  
                        index_file_path=getattr(cfg, 'valid_pkl_index', None) if getattr(cfg, 'valid_pkl_index', '') else None  
                    )  
                    logger.info(f"验证数据集(PKL)加载完成，{len(dataset_val)} 个样本")  

            # 测试集(同理)  
            if cfg.test_pkl_pattern:  
                if is_lmdb_path(cfg.test_pkl_pattern):   
                    dataset_test = LMDBAffinCraftDataset(  
                        cfg.test_pkl_pattern,  
                        readonly=True  
                    )  
                    logger.info(f"测试数据集(LMDB)加载完成，{len(dataset_test)} 个样本")  
                elif os.path.exists(cfg.test_pkl_pattern):  
                    dataset_test = OptimizedBatchedLazyAffinCraftDataset(  
                        cfg.test_pkl_pattern,  
                        batch_size=16,  
                        total_objects=getattr(cfg, 'test_pkl_objects', None),  
                        index_file_path=getattr(cfg, 'test_pkl_index', None) if getattr(cfg, 'test_pkl_index', '') else None  
                    )  
                    logger.info(f"测试数据集(PKL)加载完成，{len(dataset_test)} 个样本")  

            if not any([dataset_train, dataset_val, dataset_test]):  
                raise ValueError("未找到任何数据集，请检查PKL/LMDB文件路径")  

            # 创建wrapper  
            self.dm = AffinCraftDatasetWrapper(  
                train_complexes=None,  
                valid_complexes=None,  
                test_complexes=None,  
                is_merged=False,  
                seed=cfg.seed  
            )  

            self.dm.dataset_train = dataset_train  
            self.dm.dataset_val = dataset_val  
            self.dm.dataset_test = dataset_test  

        except Exception as e:  
            logger.error(f"设置AffinCraft数据集时出错: {str(e)}")  
            raise e

    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        if batched_data is None:
            raise ValueError(f"No data available for split: {split}")

        if self.cfg.dataset_source == "affincraft":
            batched_data = AffinCraftBatchedDataDataset(
                batched_data,
                max_node=self.max_nodes()
            )
            target = AffinCraftTargetDataset(batched_data.dataset)
        else:
            batched_data = BatchedDataDataset(
                batched_data,
                max_node=self.max_nodes(),
                multi_hop_max_dist=self.cfg.multi_hop_max_dist,
                spatial_pos_max=self.cfg.spatial_pos_max,
            )
            target = TargetDataset(batched_data)

        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset, len(dataset), seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models
        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes
        model = models.build_model(cfg, self)
        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None

    def get_targets(self, sample, net_output):
        if self.cfg.dataset_source == "affincraft":
            return sample["target"]
        else:
            return sample["target"]

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


@dataclass
class GraphPredictionWithFlagConfig(GraphPredictionConfig):
    flag_m: int = field(
        default=3,
        metadata={
            "help": "number of iterations to optimize the perturbations with flag objectives"
        },
    )

    flag_step_size: float = field(
        default=1e-3,
        metadata={
            "help": "learning rate of iterations to optimize the perturbations with flag objective"
        },
    )

    flag_mag: float = field(
        default=1e-3,
        metadata={"help": "magnitude bound for perturbations in flag objectives"},
    )


@register_task("graph_prediction_with_flag", dataclass=GraphPredictionWithFlagConfig)
class GraphPredictionWithFlagTask(GraphPredictionTask):
    """Graph prediction task with FLAG support, specifically adapted for AffinCraft datasets."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.flag_m = cfg.flag_m
        self.flag_step_size = cfg.flag_step_size
        self.flag_mag = cfg.flag_mag

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.set_detect_anomaly(False):
            loss, sample_size, logging_output = criterion(model, sample)

        if self.flag_m > 0:
            node_feat = sample["net_input"]["batched_data"]["node_feat"]
            embedding_dim = 1024
            batch_size, num_nodes = node_feat.shape[:2]
            perturb = torch.FloatTensor(batch_size, num_nodes, embedding_dim).uniform_(
                -self.flag_mag, self.flag_mag
            ).to(node_feat.device)
            perturb.requires_grad_()

            for m in range(self.flag_m):
                sample_perturb = {
                    "nsamples": sample["nsamples"],
                    "net_input": sample["net_input"],
                    "target": sample["target"],
                    "perturb": perturb
                }

                loss_perturb, _, _ = criterion(model, sample_perturb)

                grad = torch.autograd.grad(loss_perturb, perturb, only_inputs=True)[0]
                perturb = perturb + self.flag_step_size * grad / (grad.norm() + 1e-8)
                perturb = torch.clamp(perturb, -self.flag_mag, self.flag_mag)
                perturb = perturb.detach().requires_grad_()

            sample["perturb"] = perturb
            loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0

        optimizer.backward(loss)
        return loss, sample_size, logging_output