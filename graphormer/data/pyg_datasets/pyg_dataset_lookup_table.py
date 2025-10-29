# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import torch
import pickle
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
import random
import copy


def process_num_on_node():
    ngpu = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % ngpu
    return local_rank



class MDDataset(InMemoryDataset):
    url = "https://scientificdata.blob.core.windows.net/dynaformer/dataset/mddata/{}.zip"

    def __init__(self, root, set_name, split, seed=None, transform=None, pre_transform=None, pre_filter=None):
        self.path = Path(f"{root}").absolute()
        self.set_name = set_name
        self.split = split
        self.seed = int(seed)
        assert set_name.startswith("md-refined2019")
        assert split in ["train", "valid", "test"]
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.path / f"processed_{self.set_name}_{self.split}.pkl")
    
    @property
    def raw_dir(self):
        return str(self.path)

    @property
    def processed_dir(self):
        return str(self.path)
    
    @property
    def processed_file_names(self):
        if self.split in ["train", "valid"]:
            return [f"processed_{self.set_name}_{s}.pkl" for s in ["train", "valid"]]
        else:
            return [f"processed_{self.set_name}_test.pkl"]

    @property
    def raw_file_names(self):
        if self.split in ["train", "valid"]:
            return [f"{self.set_name}_train_val.pkl"]
        else:
            return [f"{self.set_name}_test.pkl"]

    def download(self):
        if not dist.is_initialized() or process_num_on_node() == 0:
            # print(f"This is rank {dist.get_rank()} / {dist.get_world_size()}")
            shutil.rmtree(self.raw_dir)
            path = download_url(self.url.format(self.set_name), self.root)
            extract_zip(path, self.root)
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        def concat_data(lst):
            tmp = []
            for i in lst:
                tmp += i
            return tmp
        if not dist.is_initialized() or process_num_on_node() == 0:
            print(f"Loading file: {self.raw_paths[0]}, exists? {Path(self.raw_paths[0]).is_file()}")
            with open(self.raw_paths[0], 'rb') as fin:
                data_lists = pickle.load(fin)
            pdbids = [d[0].pdbid for d in data_lists]
            print(f"Loading {self.split} set with {len(data_lists)} complex, {sum([len(i) for i in data_lists])} graphs.")
            if self.split in ["train", "valid"]:
                train_pdbid, valid_pdbid = train_test_split(pdbids, test_size=len(pdbids) // 10, random_state=self.seed)
                train_data = concat_data([d for d in data_lists if d[0].pdbid in train_pdbid])
                valid_data = concat_data([d for d in data_lists if d[0].pdbid in valid_pdbid])
                random.shuffle(train_data)
                random.shuffle(valid_data)
                torch.save(self.collate(train_data), self.processed_paths[0])
                torch.save(self.collate(valid_data), self.processed_paths[1])
            else:
                test_data = concat_data(data_lists)
                torch.save(self.collate(test_data), self.processed_paths[0])

        if dist.is_initialized():
            dist.barrier()


class PDBBind(InMemoryDataset):
    url = "https://scientificdata.blob.core.windows.net/dynaformer/dataset/pdbbind/{}.zip"

    def __init__(self, root, set_name, cutoffs, split, seed, transform=None, pre_transform=None,
                 pre_filter=None):
        self.path = Path(f"{root}").absolute()
        self.set_name = set_name
        self.cutoffs = cutoffs
        self.split = split
        self.seed = int(seed)
        super().__init__(root, transform, pre_transform, pre_filter)
        if split == "train":
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == "valid":
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == "test":
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_dir(self):
        return str(self.path / self.set_name)

    @property
    def processed_dir(self):
        return str(self.path / self.set_name)

    @property
    def processed_file_names(self):
        elem = self.set_name.split('-')
        assert len(elem) == 5
        base = f"{elem[0]}-{elem[1]}-{elem[2]}-{self.cutoffs}"
        return [f"processed_{base}_train.pkl", f"processed_{base}_valid.pkl", f"processed_{base}_test.pkl"]

    @property
    def raw_file_names(self):
        elem = self.set_name.split('-')
        assert len(elem) == 5
        base = f"{elem[0]}-{elem[1]}-{elem[2]}-{self.cutoffs}"
        return [f"{base}_train_val.pkl", f"{base}_test.pkl"]

    def download(self):
        if not dist.is_initialized() or process_num_on_node() == 0:
            shutil.rmtree(self.raw_dir)
            path = download_url(self.url.format(self.set_name), self.root)
            extract_zip(path, self.root)
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or process_num_on_node() == 0:
            # train valid data
            print(f"Loading file: {self.raw_paths[0]}, exists? {Path(self.raw_paths[0]).is_file()}")
            with open(self.raw_paths[0], 'rb') as fin:
                data_list = pickle.load(fin)

            train_data, valid_data = train_test_split(data_list, test_size=len(data_list) // 10, random_state=self.seed)
            random.shuffle(train_data)
            random.shuffle(valid_data)
            torch.save(self.collate(train_data), self.processed_paths[0])
            torch.save(self.collate(valid_data), self.processed_paths[1])
            # test data
            print(f"Loading file: {self.raw_paths[1]}, exists? {Path(self.raw_paths[1]).is_file()}")
            with open(self.raw_paths[1], 'rb') as fin:
                data_list = pickle.load(fin)
            torch.save(self.collate(data_list), self.processed_paths[2])

        if dist.is_initialized():
            dist.barrier()


class PDBBindWrapper(Dataset):
    def __init__(self, data_list):
        super(PDBBindWrapper, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return copy.copy(self.data_list[idx])


def pdbbind_helper(root, set_name, *args, **kwargs):
    if set_name.startswith("refined-set"):
        return PDBBind(root, set_name, *args, **kwargs)
    elif set_name.startswith("general-set"):
        data_refined = PDBBind(root, set_name.replace("general-set", "refined-set"), *args, **kwargs)
        data_general = PDBBind(root, set_name, *args, **kwargs)
        data_list = [data_refined.get(i) for i in range(len(data_refined))] + [data_general.get(i) for i in range(len(data_general))]
        del data_refined
        del data_general
        return PDBBindWrapper(data_list)


class HybridData(InMemoryDataset):
    def __init__(self, root, set_name, cutoffs, split, seed=None, transform=None, pre_transform=None, pre_filter=None):
        self.md_set_name, self.pdbbind_set_name = set_name.split("+")
        print(f"Loading hybrid data from {self.md_set_name}, {self.pdbbind_set_name}")
        self.path = Path(f"{root}").absolute()
        self.split = split
        if split == "train":
            self.md_data = MDDataset(root, self.md_set_name, split, seed)
            self.pdbbind_data = pdbbind_helper(root, self.pdbbind_set_name, cutoffs, split, seed)
        elif split == "valid":
            self.md_data = MDDataset(root, self.md_set_name, split, seed)
            self.pdbbind_data = pdbbind_helper(root, self.pdbbind_set_name, cutoffs, split, seed)
        elif split == "test":
            self.md_data = MDDataset(root, self.md_set_name, split, seed)
            self.pdbbind_data = pdbbind_helper(root, self.pdbbind_set_name, cutoffs, split, seed)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return str(self.path / (self.md_set_name + "__" + self.pdbbind_set_name))

    @property
    def processed_dir(self):
        return str(self.path / (self.md_set_name + "__" + self.pdbbind_set_name))

    @property
    def processed_file_names(self):
        base = f"{self.md_set_name}__{self.pdbbind_set_name}"
        return [f"processed_hybrid_{base}_{self.split}.pkl"]

    @property
    def raw_file_names(self):
        return []

    def process(self):
        if not dist.is_initialized() or process_num_on_node() == 0:
            data_md = [self.md_data.get(i) for i in range(len(self.md_data))]
            data_pdbbind = [self.pdbbind_data.get(i) for i in range(len(self.pdbbind_data))]
            del self.md_data
            del self.pdbbind_data
            data_list = data_md + data_pdbbind
            random.shuffle(data_list)
            data, slices = self.collate(data_list)
            print(f"Loaded {len(data_list)} from MD data and PDBBind data")
            torch.save((data, slices), self.processed_paths[0])
        if dist.is_initialized():
            dist.barrier()


class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, data_path: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        else:
            raise RuntimeError(f"Dataset name not valid: {dataset_spec}")
        inner_dataset = None

        train_set = None
        valid_set = None
        test_set = None

        root = str(Path(data_path).absolute()) if data_path else str(Path("dataset").absolute())
        print(f"Root at {root}")
        if name == "pdbbind":
            # set_name, cutoffs, split, seed
            set_name, cutoffs, seed = None, None, None
            for param in params:
                key, value = param.split("=")
                if key == "set_name":
                    set_name = value
                if key == "cutoffs":
                    cutoffs = value
                if key == "seed":
                    seed = value
            train_set = pdbbind_helper(root, set_name=set_name, cutoffs=cutoffs, split="train", seed=seed)
            valid_set = pdbbind_helper(root, set_name=set_name, cutoffs=cutoffs, split="valid", seed=seed)
            test_set = pdbbind_helper(root, set_name=set_name, cutoffs=cutoffs, split="test", seed=seed)
        elif name == "mddata":
            set_name, seed = None, None
            for param in params:
                key, value = param.split("=")
                if key == "set_name":
                    set_name = value
                if key == "seed":
                    seed = value
            train_set = MDDataset(root, set_name=set_name, split="train", seed=seed)
            valid_set = MDDataset(root, set_name=set_name, split="valid", seed=seed)
            test_set = MDDataset(root, set_name=set_name, split="test", seed=seed)
        elif name == "hybrid":
            set_name, cutoffs, seed = None, None, None
            for param in params:
                key, value = param.split("=")
                if key == "set_name":
                    set_name = value
                if key == "cutoffs":
                    cutoffs = value
                if key == "seed":
                    seed = value
            train_set = HybridData(root, set_name=set_name, cutoffs=cutoffs, split="train", seed=seed)
            valid_set = HybridData(root, set_name=set_name, cutoffs=cutoffs, split="valid", seed=seed)
            test_set = HybridData(root, set_name=set_name, cutoffs=cutoffs, split="test", seed=seed)

        elif name == "custom":
            train_path, valid_path, test_path = None, None, None
            for param in params:
                key, value = param.split("=")
                if key == "train_path":
                    train_path = value
                if key == "valid_path":
                    valid_path = value
                if key == "test_path":
                    test_path = value
            train_set = valid_set = test_set = None
            if train_path:
                with open(train_path, 'rb') as f:
                    train_set = PDBBindWrapper(pickle.load(f))
            if valid_path:
                with open(valid_path, 'rb') as f:
                    valid_set = PDBBindWrapper(pickle.load(f))
            if test_path:
                with open(test_path, 'rb') as f:
                    test_set = PDBBindWrapper(pickle.load(f))
            if train_set is not None or valid_set is not None or test_set is not None:
                return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                    )       
            else:
                raise ValueError(f"At least one of train/valid/test must be provided for custom pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
