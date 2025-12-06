import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import os
import lmdb
import time
import sys


# =====================================================
# 预处理函数
# =====================================================

def preprocess_affincraft_item(pkl_data):  
    """专门处理AffinCraft PKL文件的预处理函数"""  
      
    # 处理新数据格式：如果pkl_data是列表，取第一个元素（字典）  
    if isinstance(pkl_data, list):  
        if len(pkl_data) == 0:  
            return None  
        pkl_data = pkl_data[0]  
      
    # 确保现在是字典格式  
    if not isinstance(pkl_data, dict):  
        return None  
  
    # 扩展NaN检测到MaSIF特征  
    critical_fields = ['node_feat', 'edge_feat', 'coords', 'masif_desc_straight']  
    masif_fields = ['ligand_masif_feature', 'protein_masif_feature']  
      
    # 检查原有关键字段  
    for field in critical_fields:  
        if field in pkl_data:  
            data = pkl_data[field]  
            if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):  
                if np.isnan(data).any():  
                    print(f"⚠️ 跳过包含NaN的样本，字段: {field}, pdbid={pkl_data.get('pdbid', 'unknown')}")  
                    return None  
      
    # 检查新增的MaSIF特征字段  
    for field in masif_fields:  
        if field in pkl_data:  
            data = pkl_data[field]  
            if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):  
                if np.isnan(data).any():  
                    print(f"⚠️ 跳过包含NaN的MaSIF样本，字段: {field}, pdbid={pkl_data.get('pdbid', 'unknown')}")  
                    return None  
  
    if 'num_node' not in pkl_data or pkl_data['num_node'][0] <= 0:  
        print(f"⚠️ 跳过无效 num_node={pkl_data.get('num_node', [0])[0]}, pdbid={pkl_data.get('pdbid', 'unknown')}")    
        return None   
  
    # 继续原有处理逻辑...  
    node_feat = torch.from_numpy(pkl_data['node_feat']).clone()  
    edge_index = torch.from_numpy(pkl_data['edge_index']).clone()  
    edge_feat = torch.from_numpy(pkl_data['edge_feat']).clone()  
    coords = torch.from_numpy(pkl_data['coords']).clone()  
  
    # 生成角度和距离特征  
    from .wrapper import gen_angle_dist  
    item_data = {'edge_index': edge_index, 'pos': coords}  
    angle, dists = gen_angle_dist(item_data)  
  
    lig_spatial_edges = {  
        'index': torch.from_numpy(pkl_data['lig_spatial_edge_index']).clone(),  
        'attr': torch.from_numpy(pkl_data['lig_spatial_edge_attr']).clone(),  
    }  
    pro_spatial_edges = {  
        'index': torch.from_numpy(pkl_data['pro_spatial_edge_index']).clone(),  
        'attr': torch.from_numpy(pkl_data['pro_spatial_edge_attr']).clone(),  
    }  
  
    # 处理MaSIF特征  
    masif_features = {}  
      
    # 转换 ligand_masif_feature 和 protein_masif_feature  
    if 'ligand_masif_feature' in pkl_data:  
        masif_features['ligand_masif_feature'] = torch.from_numpy(pkl_data['ligand_masif_feature']).clone()  
          
    if 'protein_masif_feature' in pkl_data:  
        masif_features['protein_masif_feature'] = torch.from_numpy(pkl_data['protein_masif_feature']).clone()  
          
    # 保留原有的masif特征处理逻辑  
    if 'masif_desc_straight' in pkl_data:  
        masif_features['desc_straight'] = torch.from_numpy(pkl_data['masif_desc_straight']).clone()  
  
    for key in [  
        'masif_input_feat', 'masif_desc_flipped',  
        'masif_rho_wrt_center', 'masif_theta_wrt_center', 'masif_mask'  
    ]:  
        if key in pkl_data:  
            masif_features[key.replace('masif_', '')] = torch.from_numpy(pkl_data[key]).clone()  
  
    N = node_feat.shape[0]  
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  
    adj = torch.zeros([N, N], dtype=torch.bool)  
    adj[edge_index[0, :], edge_index[1, :]] = True  
    in_degree = adj.long().sum(dim=1)  
    out_degree = in_degree  
  
    return {  
        'node_feat': node_feat,  
        'edge_index': edge_index,  
        'edge_feat': edge_feat,  
        'coords': coords,  
        'lig_spatial_edges': lig_spatial_edges,  
        'pro_spatial_edges': pro_spatial_edges,  
        'masif_features': masif_features,  
        'num_ligand_atoms': pkl_data['num_node'][0],  
        'attn_bias': attn_bias,  
        'in_degree': in_degree,  
        'out_degree': out_degree,  
        'gbscore': torch.from_numpy(pkl_data['gbscore']).clone(),  
        'pdbid': pkl_data['pdbid'],  
        'angle': angle,  
        'dists': dists,  
        'pk': pkl_data['pk'],  
        'smiles': pkl_data['smiles'],  
        'rmsd': pkl_data['rmsd'],  
    }


# =====================================================
# LMDB 数据集
# =====================================================

class LMDBAffinCraftDataset(torch.utils.data.Dataset):
    """基于LMDB的AffinCraft数据集，支持高效随机访问和多进程并发读取"""

    def __init__(self, lmdb_path: str, readonly: bool = True):
        self.lmdb_path = Path(lmdb_path).expanduser().absolute()
        self.readonly = readonly
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB路径不存在: {self.lmdb_path}")

        self.env = None
        self._length = None
        self._pid = None
        self._load_metadata()

        print(f"✅ LMDB 数据集加载完成: {self.lmdb_path}")
        print(f"样本总数: {self._length:,}")

    def _load_metadata(self):
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
        with env.begin() as txn:
            meta_bytes = txn.get(b'__metadata__')
            if meta_bytes is None:
                raise ValueError(f"LMDB数据库缺少元数据: {self.lmdb_path}")
            meta = pickle.loads(meta_bytes)
            self._length = meta['total_samples']
        env.close()

    def _init_db(self):
        pid = os.getpid()
        if self.env is None or self._pid != pid:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            self.env = lmdb.open(str(self.lmdb_path), readonly=self.readonly, lock=False, readahead=False, meminit=False, max_readers=256)
            self._pid = pid

    def __len__(self):
        return self._length

    def __getitem__(self, idx):  
        from datetime import datetime  
        worker_id = os.getpid()  
        log_file = f"/tmp/dataloader_debug_worker_{worker_id}.log"  
    
        try:  
            if idx < 0 or idx >= self._length:  
                raise IndexError(f"索引 {idx} 超出范围 [0, {self._length})")  
    
            with open(log_file, "a") as f:  
                f.write(f"[{datetime.now()}] Worker {worker_id} loading {idx}\n")  
    
            self._init_db()  
    
            with self.env.begin() as txn:  
                key = f"{idx}".encode("ascii")  
                data_bytes = txn.get(key)  
                if data_bytes is None:  
                    raise RuntimeError(f"样本 {idx} 不存在")  
                pkl_data = pickle.loads(data_bytes)  
    
            processed = preprocess_affincraft_item(pkl_data)  
            if processed is None:  
                # 修复：处理pkl_data是list的情况  
                if isinstance(pkl_data, list) and len(pkl_data) > 0:  
                    pdbid = pkl_data[0].get("pdbid", f"sample_{idx}") if isinstance(pkl_data[0], dict) else f"sample_{idx}"  
                else:  
                    pdbid = f"sample_{idx}"  
                return {"_skip": True, "idx": idx, "pdbid": pdbid, "pk": 0.0}  
            return processed  
    
        except Exception as e:  
            import traceback  
            err = f"""  
    [{datetime.now()}] 错误:  
      Index: {idx}  
      Type: {type(e).__name__}  
      Msg: {e}  
    Traceback:  
    {traceback.format_exc()}  
    """  
            print(err, file=sys.stderr)  
            with open(log_file, "a") as f:  
                f.write(err)  
            return {"_skip": True, "idx": idx, "pdbid": f"error_{idx}", "pk": 0.0}


# =====================================================
# 带缓存的LMDB数据集
# =====================================================

class CachedLMDBAffinCraftDataset(torch.utils.data.Dataset):
    """带缓存的LMDB数据集"""
    def __init__(self, lmdb_path: str, cache_size: int = 1000, readonly: bool = True):
        self.lmdb_path = Path(lmdb_path).expanduser().absolute()
        self.cache_size = cache_size
        self.readonly = readonly
        self.env = None
        self._pid = None
        self._cache = {}
        self._length = None
        self._load_metadata()

    def _load_metadata(self):
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            meta_bytes = txn.get(b'__metadata__')
            if meta_bytes is None:
                raise ValueError(f"{self.lmdb_path} 缺少元数据")
            meta = pickle.loads(meta_bytes)
            self._length = meta['total_samples']
        env.close()

    def _init_db(self):
        pid = os.getpid()
        if self.env is None or self._pid != pid:
            if self.env is not None:
                self.env.close()
            self.env = lmdb.open(str(self.lmdb_path),
                                 readonly=self.readonly,
                                 lock=False, readahead=False, meminit=False, max_readers=256)
            self._pid = pid

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        self._init_db()
        with self.env.begin() as txn:
            key = f"{idx}".encode("ascii")
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise RuntimeError(f"索引 {idx} 不存在")
            pkl_data = pickle.loads(data_bytes)

        processed = preprocess_affincraft_item(pkl_data)
        if processed is None:
            return {'_skip': True, 'idx': idx, 'pdbid': f'invalid_{idx}', 'pk': 0.0}

        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = processed
        return processed

    def __del__(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass


# =====================================================
# 懒加载 PKL 数据集
# =====================================================

class OptimizedBatchedLazyAffinCraftDataset(Dataset):
    def __init__(self, pkl_file_path, batch_size=16, total_objects=None, index_file_path=None):
        self.pkl_file_path = pkl_file_path
        self.batch_size = batch_size
        self.total_objects = total_objects
        self.file_size = os.path.getsize(pkl_file_path)
        self._length = None
        self._file_positions = None
        self._batch_cache = {}

        print(f"文件大小: {self.file_size / (1024**3):.2f} GB")

        if index_file_path and os.path.exists(index_file_path):
            self._load_index(index_file_path)
        else:
            print("未提供索引文件，开始构建索引...")
            self._build_index()

    def _load_index(self, path):
        print(f"加载索引文件: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._file_positions = data['positions']
        self._length = data['total_objects']
        if data.get('file_size') != self.file_size:
            print("⚠️ 索引与文件大小不匹配")
        print(f"索引加载完成，共{self._length:,}对象")

    def _build_index(self):
        positions = []
        with open(self.pkl_file_path, 'rb') as f:
            count = 0
            while True:
                pos = f.tell()
                try:
                    pickle.load(f)
                    positions.append(pos)
                    count += 1
                    if count % 10000 == 0:
                        print(f"已索引 {count}...")
                except EOFError:
                    break
                except pickle.UnpicklingError as e:
                    print(f"pickle错误: {e}, 终止")
                    break
        self._file_positions = positions
        self._length = len(positions)
        print(f"索引构建完成，总对象 {self._length}")

    def __len__(self):
        return self._length

    def _get_range(self, idx):
        b = idx // self.batch_size
        s = b * self.batch_size
        e = min(s + self.batch_size, self._length)
        return b, s, e

    def _load_batch(self, batch_id, s, e):
        if batch_id in self._batch_cache:
            return self._batch_cache[batch_id]
        result = []
        with open(self.pkl_file_path, 'rb') as f:
            for i in range(s, e):
                try:
                    f.seek(self._file_positions[i])
                    result.append(pickle.load(f))
                except Exception as ex:
                    print(f"载入 {i} 出错: {ex}")
                    break
        if len(self._batch_cache) < 10:
            self._batch_cache[batch_id] = result
        return result

    def __getitem__(self, idx):
        if idx >= self._length:
            raise IndexError(idx)
        start = time.time()
        b, s, e = self._get_range(idx)
        data = self._load_batch(b, s, e)
        rel = idx - s
        pkl_data = data[rel] if rel < len(data) else None
        if pkl_data is None:
            raise RuntimeError(f"数据为空 idx={idx}")
        print("读取一条耗时:", round(time.time() - start, 3), "s")
        return preprocess_affincraft_item(pkl_data)


# =====================================================
# 数据批量合并函数
# =====================================================

def affincraft_collator(items, max_node=512):  
    """批量组合函数"""  
    items = [i for i in items if i and not i.get('_skip', False)]  
    if not items:  
        return None  
      
    max_node_num = max(i['node_feat'].size(0) for i in items)  
    max_edge_num = max(i['edge_feat'].size(0) for i in items)  
      
    # 修复：从masif_features子字典中检查新格式特征  
    has_new_masif = any(  
        'masif_features' in i and   
        'ligand_masif_feature' in i['masif_features'] and   
        'protein_masif_feature' in i['masif_features']   
        for i in items  
    )  
    has_old_masif = any(  
        'masif_features' in i and   
        'desc_straight' in i['masif_features']   
        for i in items  
    )  
      
    if has_new_masif:  
        # 计算新格式特征的最大尺寸  
        masif_items = [i for i in items if 'masif_features' in i and   
                      'ligand_masif_feature' in i['masif_features'] and   
                      'protein_masif_feature' in i['masif_features']]  
        if masif_items:  
            max_ligand_masif = max(i['masif_features']['ligand_masif_feature'].size(0) for i in masif_items)  
            max_protein_masif = max(i['masif_features']['protein_masif_feature'].size(0) for i in masif_items)  
            ligand_spatial_dim = masif_items[0]['masif_features']['ligand_masif_feature'].size(1)  
            protein_spatial_dim = masif_items[0]['masif_features']['protein_masif_feature'].size(1)  
            feature_dim = 5  
    elif has_old_masif:  
        max_masif = max(i['masif_features']['desc_straight'].size(0)  
                        for i in items if 'masif_features' in i and 'desc_straight' in i['masif_features'])  
    else:  
        max_masif = 0  
  
    # 初始化列表  
    node_feats, edge_feats, coords, edge_indices = [], [], [], []  
    attn_biases, in_degs, out_degs, angles, dists = [], [], [], [], []  
      
    if has_new_masif:  
        ligand_masif_feats = []  
        protein_masif_feats = []  
    elif has_old_masif:  
        masif_feats, masif_masks = [], []  
  
    for i in items:  
        n, e = i['node_feat'].size(0), i['edge_feat'].size(0)  
          
        # 基础特征处理  
        nf = torch.zeros(max_node_num, i['node_feat'].size(1))  
        nf[:n] = i['node_feat']  
        node_feats.append(nf)  
  
        ef = torch.zeros(max_edge_num, i['edge_feat'].size(1))  
        ef[:e] = i['edge_feat']  
        edge_feats.append(ef)  
  
        ei = torch.zeros(2, max_edge_num, dtype=torch.long)  
        ei[:, :e] = i['edge_index']  
        edge_indices.append(ei)  
  
        cf = torch.zeros(max_node_num, 3)  
        cf[:n] = i['coords']  
        coords.append(cf)  
  
        ab = torch.zeros(max_node_num + 1, max_node_num + 1)  
        ab[:n + 1, :n + 1] = i['attn_bias']  
        attn_biases.append(ab)  
  
        indeg = torch.zeros(max_node_num, dtype=torch.long)  
        indeg[:n] = i['in_degree']  
        outdeg = torch.zeros(max_node_num, dtype=torch.long)  
        outdeg[:n] = i['out_degree']  
        in_degs.append(indeg)  
        out_degs.append(outdeg)  
  
        if 'angle' in i and i['angle'] is not None:  
            A = torch.zeros(max_node_num, max_node_num, i['angle'].size(-1))  
            D = torch.zeros_like(A)  
            sz = i['angle'].size(0)  
            A[:sz, :sz] = i['angle']  
            D[:sz, :sz] = i['dists']  
            angles.append(A)  
            dists.append(D)  
  
        # 修复：从masif_features子字典中获取MaSIF特征  
        if has_new_masif:  
            if ('masif_features' in i and   
                'ligand_masif_feature' in i['masif_features'] and   
                'protein_masif_feature' in i['masif_features']):  
                  
                lig_feat = i['masif_features']['ligand_masif_feature']  
                pro_feat = i['masif_features']['protein_masif_feature']  
                  
                lig_padded = torch.zeros(max_ligand_masif, ligand_spatial_dim, feature_dim)  
                lig_padded[:lig_feat.size(0), :lig_feat.size(1), :lig_feat.size(2)] = lig_feat  
                ligand_masif_feats.append(lig_padded)  
                    
                pro_padded = torch.zeros(max_protein_masif, protein_spatial_dim, feature_dim)  
                pro_padded[:pro_feat.size(0), :pro_feat.size(1), :pro_feat.size(2)] = pro_feat  
                protein_masif_feats.append(pro_padded)  
            else:  
                ligand_masif_feats.append(torch.zeros(max_ligand_masif, ligand_spatial_dim, feature_dim))  
                protein_masif_feats.append(torch.zeros(max_protein_masif, protein_spatial_dim, feature_dim))  
                  
        elif has_old_masif:  
            m = i['masif_features'].get('desc_straight')  
            if m is not None:  
                nm = m.size(0)  
                pf = torch.zeros(max_masif, m.size(1))  
                pf[:nm] = m  
                mf = torch.zeros(max_masif, dtype=torch.bool)  
                mf[:nm] = True  
            else:  
                pf = torch.zeros(max_masif, 80)  
                mf = torch.zeros(max_masif, dtype=torch.bool)  
            masif_feats.append(pf)  
            masif_masks.append(mf)  
  
    # 构建返回字典  
    result = {  
        'node_feat': torch.stack(node_feats),  
        'edge_feat': torch.stack(edge_feats),  
        'coords': torch.stack(coords),  
        'edge_index': torch.stack(edge_indices),  
        'attn_bias': torch.stack(attn_biases),  
        'in_degree': torch.stack(in_degs),  
        'out_degree': torch.stack(out_degs),  
        'angle': torch.stack(angles) if angles else None,  
        'dists': torch.stack(dists) if dists else None,  
        'pdbid': [i['pdbid'] for i in items],  
        'pk': torch.tensor([i['pk'] for i in items]),  
        'rmsd': torch.tensor([i['rmsd'] for i in items], dtype=torch.float),  
        'gbscore': torch.stack([i['gbscore'] for i in items]),  
        'num_ligand_atoms': torch.tensor([i['num_ligand_atoms'] for i in items]),  
        'smiles': [i['smiles'] for i in items],  
    }  
      
    # 添加MaSIF特征到返回字典的顶层  
    if has_new_masif:
        ligand = torch.stack(ligand_masif_feats)
        protein = torch.stack(protein_masif_feats)
    
        # Debug: batch 级 NaN 检查
        if torch.isnan(ligand).any():
            print("[COLLATOR] NaN detected in ligand_masif_feature batch")
        if torch.isnan(protein).any():
            print("[COLLATOR] NaN detected in protein_masif_feature batch")
    
        result['ligand_masif_feature'] = ligand
        result['protein_masif_feature'] = protein
    elif has_old_masif:  
        result['masif_desc_straight'] = torch.stack(masif_feats)  
        result['masif_mask'] = torch.stack(masif_masks)  
        
    return result