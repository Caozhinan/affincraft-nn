import pickle    
import torch    
import numpy as np    
from pathlib import Path    
from typing import List, Dict, Any, Optional    
from torch.utils.data import Dataset, DataLoader    
import mmap
import os 
from typing import Optional
import lmdb
import time
# 如果需要使用现有的数据处理工具    
# from .wrapper import preprocess_item    
# from .collator import collator  
  
class AffinCraftDataset(torch.utils.data.Dataset):      
    def __init__(self, data, is_merged=False):    
        if is_merged:    
            # 直接使用复合物数据列表    
            self.complexes = data    
            self.pkl_files = None    
        else:    
            # 原有的PKL文件列表方式    
            self.pkl_files = data    
            self.complexes = None    
              
    def __len__(self):    
        if self.complexes is not None:    
            return len(self.complexes)    
        return len(self.pkl_files)    
          
    def __getitem__(self, idx):      
        if self.complexes is not None:      
            # 直接从复合物列表获取数据（已修复字典访问问题）  
            pkl_data = self.complexes[idx]      
        else:      
            # 原有方式：从PKL文件加载      
            pkl_file = self.pkl_files[idx]        
            with open(pkl_file, 'rb') as f:        
                pkl_data = pickle.load(f)[0]    
  
        return preprocess_affincraft_item(pkl_data)  
  
  
class BatchedLazyAffinCraftDataset(torch.utils.data.Dataset):    
    def __init__(self, pkl_file_path, batch_size=32):    
        self.pkl_file_path = pkl_file_path    
        self.batch_size = batch_size    
        self._length = None    
        self._file_positions = None    
        self._batch_cache = {}  # 缓存已加载的批次    
        self._build_index()    
        
    def _build_index(self):    
        """构建文件位置索引，不加载实际数据"""    
        positions = []    
        with open(self.pkl_file_path, 'rb') as f:    
            try:    
                while True:    
                    pos = f.tell()    
                    pickle.load(f)  # 跳过对象    
                    positions.append(pos)    
            except EOFError:    
                pass    
            
        self._file_positions = positions    
        self._length = len(positions)    
        
    def _get_batch_indices(self, idx):    
        """获取包含指定索引的批次范围"""    
        batch_id = idx // self.batch_size    
        start_idx = batch_id * self.batch_size    
        end_idx = min(start_idx + self.batch_size, self._length)    
        return batch_id, start_idx, end_idx    
        
    def _load_batch(self, batch_id, start_idx, end_idx):    
        """加载一个批次的数据"""    
        if batch_id in self._batch_cache:    
            return self._batch_cache[batch_id]    
            
        batch_data = []    
        with open(self.pkl_file_path, 'rb') as f:    
            for i in range(start_idx, end_idx):    
                f.seek(self._file_positions[i])    
                pkl_data = pickle.load(f)    
                batch_data.append(pkl_data)    
            
        # 缓存批次数据（可选：限制缓存大小）    
        if len(self._batch_cache) < 1000:  # 最多缓存10个批次    
            self._batch_cache[batch_id] = batch_data    
            
        return batch_data    
        
    def __len__(self):    
        return self._length    
        
    def __getitem__(self, idx):    
        batch_id, start_idx, end_idx = self._get_batch_indices(idx)    
        batch_data = self._load_batch(batch_id, start_idx, end_idx)    
            
        # 返回批次中的特定样本    
        local_idx = idx - start_idx    
        pkl_data = batch_data[local_idx]    
            
        return preprocess_affincraft_item(pkl_data)  

  
class LMDBAffinCraftDataset(torch.utils.data.Dataset):  
    """  
    基于LMDB的AffinCraft数据集  
    支持高效的随机访问和多进程并发读取  
    使用延迟初始化避免fork问题  
    """  
      
    def __init__(self, lmdb_path: str, readonly: bool = True):  
        """  
        初始化LMDB数据集  
          
        Args:  
            lmdb_path: LMDB数据库路径  
            readonly: 是否以只读模式打开(训练时应为True)  
        """  
        self.lmdb_path = Path(lmdb_path).expanduser().absolute()  
        self.readonly = readonly  
          
        if not self.lmdb_path.exists():  
            raise FileNotFoundError(f"LMDB路径不存在: {self.lmdb_path}")  
          
        # 🟡 延迟初始化:不在__init__中打开LMDB  
        self.env = None  
        self._length = None  
          
        # 只在主进程中读取元数据获取样本总数  
        self._load_metadata()  
          
        print(f"LMDB数据集加载完成: {self.lmdb_path}")  
        print(f"样本总数: {self._length:,}")  
      
    def _load_metadata(self):  
        """只读取元数据,不保持环境打开"""  
        env = lmdb.open(  
            str(self.lmdb_path),  
            readonly=True,  
            lock=False,  
            readahead=False,  
            meminit=False,  
            max_readers=256  
        )  
          
        with env.begin() as txn:  
            metadata_bytes = txn.get(b'__metadata__')  
            if metadata_bytes is None:  
                raise ValueError(f"LMDB数据库缺少元数据: {self.lmdb_path}")  
              
            metadata = pickle.loads(metadata_bytes)  
            self._length = metadata['total_samples']  
          
        env.close()  # 立即关闭,不保持打开  
      
    def _init_db(self):  
        """延迟初始化:在第一次访问时才打开LMDB"""  
        if self.env is None:  
            self.env = lmdb.open(  
                str(self.lmdb_path),  
                readonly=self.readonly,  
                lock=False,  
                readahead=False,  
                meminit=False,  
                max_readers=256  
            )  
      
    def __len__(self):  
        return self._length  
      
    def __getitem__(self, idx):  
        """  
        获取指定索引的样本,自动跳过包含NaN的样本  
          
        Args:  
            idx: 样本索引  
              
        Returns:  
            预处理后的样本数据,如果包含NaN则返回None  
        """  
        if idx >= self._length or idx < 0:  
            raise IndexError(f"索引 {idx} 超出范围 [0, {self._length})")  
          
        # 🟡 延迟初始化:每个worker进程第一次调用时才打开LMDB  
        self._init_db()  
          
        # 从LMDB读取数据  
        with self.env.begin() as txn:  
            key = f'{idx}'.encode('ascii')  
            data_bytes = txn.get(key)  
              
            if data_bytes is None:  
                raise RuntimeError(f"无法读取索引 {idx} 的数据")  
              
            # 反序列化  
            pkl_data = pickle.loads(data_bytes)  
          
        # 使用现有的预处理函数(已包含NaN检测)  
        processed_data = preprocess_affincraft_item(pkl_data)  
          
        # 如果返回None,说明包含NaN  
        if processed_data is None:  
            pdbid = pkl_data.get('pdbid', f'sample_{idx}')  
            print(f"警告: 跳过包含NaN的样本 {idx} (pdbid: {pdbid})")  
            return None  
          
        return processed_data  
      
    def __del__(self):  
        """清理资源"""  
        if hasattr(self, 'env') and self.env is not None:  
            self.env.close()
  
  
class CachedLMDBAffinCraftDataset(torch.utils.data.Dataset):  
    """  
    带缓存的LMDB数据集  
    在内存中缓存最近访问的样本,进一步提升性能  
    """  
      
    def __init__(self, lmdb_path: str, cache_size: int = 1000, readonly: bool = True):  
        """  
        初始化带缓存的LMDB数据集  
          
        Args:  
            lmdb_path: LMDB数据库路径  
            cache_size: 缓存样本数量  
            readonly: 是否以只读模式打开  
        """  
        self.lmdb_path = Path(lmdb_path).expanduser().absolute()  
        self.cache_size = cache_size  
        self._cache = {}  # 简单的字典缓存  
          
        if not self.lmdb_path.exists():  
            raise FileNotFoundError(f"LMDB路径不存在: {self.lmdb_path}")  
          
        # 打开LMDB环境  
        self.env = lmdb.open(  
            str(self.lmdb_path),  
            readonly=readonly,  
            lock=False,  
            readahead=False,  
            meminit=False,  
            max_readers=256  
        )  
          
        # 读取元数据  
        with self.env.begin() as txn:  
            metadata_bytes = txn.get(b'__metadata__')  
            if metadata_bytes is None:  
                raise ValueError(f"LMDB数据库缺少元数据: {self.lmdb_path}")  
              
            metadata = pickle.loads(metadata_bytes)  
            self._length = metadata['total_samples']  
          
        print(f"带缓存的LMDB数据集加载完成: {self.lmdb_path}")  
        print(f"样本总数: {self._length:,}, 缓存大小: {cache_size}")  
      
    def __len__(self):  
        return self._length  
      
    def __getitem__(self, idx):  
        if idx >= self._length or idx < 0:  
            raise IndexError(f"索引 {idx} 超出范围 [0, {self._length})")  
          
        # 检查缓存  
        if idx in self._cache:  
            return self._cache[idx]  
          
        # 从LMDB读取  
        with self.env.begin() as txn:  
            key = f'{idx}'.encode('ascii')  
            data_bytes = txn.get(key)  
              
            if data_bytes is None:  
                raise RuntimeError(f"无法读取索引 {idx} 的数据")  
              
            pkl_data = pickle.loads(data_bytes)  
          
        # 预处理  
        processed_data = preprocess_affincraft_item(pkl_data)  
          
        # 更新缓存(简单的LRU策略)  
        if len(self._cache) >= self.cache_size:  
            # 删除第一个元素(最旧的)  
            self._cache.pop(next(iter(self._cache)))  
          
        self._cache[idx] = processed_data  
          
        return processed_data  
      
    def __del__(self):  
        if hasattr(self, 'env'):  
            self.env.close()

class OptimizedBatchedLazyAffinCraftDataset(torch.utils.data.Dataset):  
    def __init__(self, pkl_file_path, batch_size=16, total_objects=None, index_file_path=None):  
        """  
        优化的批次懒加载数据集  
          
        Args:  
            pkl_file_path: PKL文件路径  
            batch_size: 批次大小  
            total_objects: 文件中的总对象数量（可选，仅用于进度显示）  
            index_file_path: 预构建索引文件路径（可选，提供后直接使用精确位置）  
        """  
        self.pkl_file_path = pkl_file_path  
        self.batch_size = batch_size  
        self.total_objects = total_objects  
        self.file_size = os.path.getsize(pkl_file_path)  
          
        self._length = None  
        self._file_positions = None  
        self._batch_cache = {}  
          
        print(f"文件大小: {self.file_size / (1024**3):.2f} GB")  
          
        # 优先使用预构建索引，否则构建完整索引  
        if index_file_path and os.path.exists(index_file_path):  
            self._load_prebuilt_index(index_file_path)  
        else:  
            print("未提供索引文件，开始构建完整索引...")  
            self._build_full_index()  
  
    def _load_prebuilt_index(self, index_file_path):  
        """加载预构建的索引文件"""  
        print(f"加载预构建索引文件: {index_file_path}")  
          
        with open(index_file_path, 'rb') as f:  
            index_data = pickle.load(f)  
          
        self._file_positions = index_data['positions']  
        self._length = index_data['total_objects']  
          
        # 验证索引文件与PKL文件的一致性  
        if index_data.get('file_size') != self.file_size:  
            print(f"警告: 索引文件记录的文件大小({index_data.get('file_size')})与实际文件大小({self.file_size})不匹配")  
          
        print(f"成功加载预构建索引：{self._length:,} 个对象")  
  
    def _build_full_index(self):    
        """构建完整的位置索引 - 逐个加载所有对象"""    
        positions = []    
    
        with open(self.pkl_file_path, 'rb') as f:    
            count = 0    
            while True:    
                try:  
                    pos = f.tell()    
                    pickle.load(f)  # 加载对象以移动文件指针    
                    positions.append(pos)    
                    count += 1    
    
                    if count % 10000 == 0:    
                        print(f"已索引 {count:,} 个对象...")    
    
                except EOFError:  
                    # 正常的文件结束  
                    print(f"完整索引构建完成，共 {count:,} 个对象")  
                    break  

                except pickle.UnpicklingError as e:    
                    print(f"警告：在位置 {pos} 遇到pickle错误: {e}")    
                    print(f"可能的文件截断，停止索引构建")    
                    break    
                
        self._file_positions = positions    
        self._length = len(positions)    

        if len(positions) == 0:    
            raise RuntimeError(f"无法从PKL文件中读取任何有效对象: {self.pkl_file_path}")
  
    def _get_batch_indices(self, idx):  
        """获取包含指定索引的批次范围"""  
        batch_id = idx // self.batch_size  
        start_idx = batch_id * self.batch_size  
        end_idx = min(start_idx + self.batch_size, self._length)  
        return batch_id, start_idx, end_idx  
  
    def _load_batch_from_positions(self, batch_id, start_idx, end_idx):  
        """从精确位置加载批次数据"""  
        if batch_id in self._batch_cache:  
            return self._batch_cache[batch_id]  
          
        batch_data = []  
        with open(self.pkl_file_path, 'rb') as f:  
            for i in range(start_idx, end_idx):  
                try:  
                    pos = self._file_positions[i]  # 直接使用精确位置  
                    f.seek(pos)  
                    pkl_data = pickle.load(f)  
                    batch_data.append(pkl_data)  
                except (EOFError, pickle.UnpicklingError, OSError) as e:  
                    print(f"Warning: Failed to load object at index {i}: {str(e)}")  
                    if isinstance(e, EOFError):  
                        break  
                    continue  
                except Exception as e:  
                    print(f"Unexpected error loading object at index {i}: {str(e)}")  
                    continue  
          
        # 缓存批次数据（限制缓存大小）  
        if len(self._batch_cache) < 10 and batch_data:  
            self._batch_cache[batch_id] = batch_data  
          
        return batch_data  
  
    def __len__(self):  
        return self._length  
  
    def __getitem__(self, idx):  
        if idx >= self._length:  
            raise IndexError(f"Index {idx} out of range for dataset of size {self._length}")  
          
        try:
            start = time.time()  
            batch_id, start_idx, end_idx = self._get_batch_indices(idx)  
            batch_data = self._load_batch_from_positions(batch_id, start_idx, end_idx)  
              
            local_idx = idx - start_idx  
            if local_idx < len(batch_data) and batch_data[local_idx] is not None:  
                pkl_data = batch_data[local_idx]  
            elif batch_data:  
                pkl_data = batch_data[-1]  
            else:  
                raise RuntimeError(f"Batch data is empty for index {idx}, batch_id {batch_id}")  
              
            if pkl_data is None:  
                raise RuntimeError(f"pkl_data is None at index {idx}")  
            print("one data time is ",time.time()-start )  
            return preprocess_affincraft_item(pkl_data)  
              
        except Exception as e:  
            print(f"Error loading data at index {idx}: {str(e)}")  
            raise RuntimeError(f"Failed to load data at index {idx}: {str(e)}")   


def preprocess_affincraft_item(pkl_data):        
    """专门处理AffinCraft PKL文件的预处理函数"""  
      
    # 添加 NaN 检测 - 检查所有关键的浮点数组字段  
    critical_fields = ['node_feat', 'edge_feat', 'coords', 'masif_desc_straight']  
      
    for field in critical_fields:  
        if field in pkl_data:  
            data = pkl_data[field]  
            if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):  
                if np.isnan(data).any():  
                    # 跳过包含 NaN 的样本  
                    return None  
    # 在 preprocess_affincraft_item 中添加  
    if pkl_data['num_node'][0] <= 0:  
        print(f"警告: 跳过无效的 num_ligand_atoms: {pkl_data['num_node'][0]} (pdbid: {pkl_data['pdbid']})")  
        return None
    # 原有的预处理逻辑        
    node_feat = torch.from_numpy(pkl_data['node_feat'])        
    edge_index = torch.from_numpy(pkl_data['edge_index'])        
    edge_feat = torch.from_numpy(pkl_data['edge_feat'])        
    coords = torch.from_numpy(pkl_data['coords'])        
        
    # 生成角度和距离特征    
    from .wrapper import gen_angle_dist      
    item_data = {      
        'edge_index': edge_index,      
        'pos': coords      
    }      
    angle, dists = gen_angle_dist(item_data)     
        
    # 处理分离的空间边信息        
    lig_spatial_edges = {        
        'index': torch.from_numpy(pkl_data['lig_spatial_edge_index']),        
        'attr': torch.from_numpy(pkl_data['lig_spatial_edge_attr'])        
    }        
            
    pro_spatial_edges = {        
        'index': torch.from_numpy(pkl_data['pro_spatial_edge_index']),        
        'attr': torch.from_numpy(pkl_data['pro_spatial_edge_attr'])        
    }        
            
    # 修改：只保留存在的MaSIF特征      
    masif_features = {}      
    if 'masif_desc_straight' in pkl_data:      
        masif_features['desc_straight'] = torch.from_numpy(pkl_data['masif_desc_straight'])      
          
    # 可选：如果需要其他MaSIF特征，检查是否存在      
    for key in ['masif_input_feat', 'masif_desc_flipped', 'masif_rho_wrt_center',       
                'masif_theta_wrt_center', 'masif_mask']:      
        if key in pkl_data:      
            dict_key = key.replace('masif_', '')      
            masif_features[dict_key] = torch.from_numpy(pkl_data[key])      
            
    # 添加embedding层需要的额外字段        
    N = node_feat.shape[0]        
            
    # 创建基础注意力偏置矩阵        
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)        
            
    # 计算度数信息（从边索引计算）        
    adj = torch.zeros([N, N], dtype=torch.bool)        
    adj[edge_index[0, :], edge_index[1, :]] = True        
    in_degree = adj.long().sum(dim=1).view(-1)        
    out_degree = in_degree  # 对于无向图        
            
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
        'gbscore': torch.from_numpy(pkl_data['gbscore']),        
        'pdbid': pkl_data['pdbid'],    
        'angle': angle,    
        'dists': dists,         
        'pk': pkl_data['pk'],        
        'smiles': pkl_data['smiles'],      
        'rmsd': pkl_data['rmsd']      
    }    
  
def affincraft_collator(items, max_node=512):        
    """AffinCraft数据的批处理函数"""        
            
    # 过滤无效数据        
    items = [item for item in items if item is not None and item['node_feat'].size(0) <= max_node]        
            
    if not items:        
        return None        
            
    max_node_num = max(item['node_feat'].size(0) for item in items)        
    max_edge_num = max(item['edge_feat'].size(0) for item in items)  
      
    # 计算最大 masif 特征数量  
    max_masif_features = max(  
        item['masif_features']['desc_straight'].size(0)   
        for item in items   
        if 'masif_features' in item and 'desc_straight' in item['masif_features']  
    )  
            
    # 批处理特征        
    node_feats = []        
    edge_feats = []        
    edge_indices = []        
    edge_masks = []  
    coords_list = []        
    attn_biases = []        
    in_degrees = []        
    out_degrees = []        
    angles = []  
    dists_list = []  
    masif_desc_straights = []  # masif 特征列表  
    masif_masks = []  # masif 掩码列表  
            
    for item in items:        
        n_node = item['node_feat'].size(0)        
        n_edge = item['edge_feat'].size(0)        
                
        # 填充节点特征        
        padded_node_feat = torch.zeros(max_node_num, item['node_feat'].size(1))        
        padded_node_feat[:n_node] = item['node_feat']        
        node_feats.append(padded_node_feat)        
                
        # 填充边特征  
        padded_edge_feat = torch.zeros(max_edge_num, item['edge_feat'].size(1))        
        padded_edge_feat[:n_edge] = item['edge_feat']        
        edge_feats.append(padded_edge_feat)        
                
        # 填充边索引        
        padded_edge_index = torch.zeros(2, max_edge_num, dtype=torch.long)        
        padded_edge_index[:, :n_edge] = item['edge_index']        
        edge_indices.append(padded_edge_index)        
                
        # 创建边掩码        
        edge_mask = torch.zeros(max_edge_num, dtype=torch.bool)        
        edge_mask[:n_edge] = True        
        edge_masks.append(edge_mask)        
                
        # 其他特征处理保持不变        
        padded_coords = torch.zeros(max_node_num, 3)        
        padded_coords[:n_node] = item['coords']        
        coords_list.append(padded_coords)        
                
        padded_attn_bias = torch.zeros(max_node_num + 1, max_node_num + 1)        
        padded_attn_bias[:n_node+1, :n_node+1] = item['attn_bias']        
        attn_biases.append(padded_attn_bias)        
                
        padded_in_degree = torch.zeros(max_node_num, dtype=torch.long)        
        padded_in_degree[:n_node] = item['in_degree']        
        in_degrees.append(padded_in_degree)        
                
        padded_out_degree = torch.zeros(max_node_num, dtype=torch.long)        
        padded_out_degree[:n_node] = item['out_degree']        
        out_degrees.append(padded_out_degree)        
              
        if 'angle' in item and item['angle'] is not None:  
            angle_size = item['angle'].size(0)  # 使用angle自身的大小  
            padded_angle = torch.zeros(max_node_num, max_node_num, item['angle'].size(-1))  
            padded_angle[:angle_size, :angle_size] = item['angle']  
            angles.append(padded_angle)  
              
            padded_dists = torch.zeros(max_node_num, max_node_num, item['dists'].size(-1))  
            padded_dists[:angle_size, :angle_size] = item['dists']  
            dists_list.append(padded_dists)  
        else:  
            angles.append(None)  
            dists_list.append(None)
              
        # 处理 masif 特征 - 新增填充和掩码处理  
        if 'masif_features' in item and 'desc_straight' in item['masif_features']:  
            masif_feat = item['masif_features']['desc_straight']  
            n_masif = masif_feat.size(0)  
              
            # 填充特征到最大长度  
            padded_masif = torch.zeros(max_masif_features, masif_feat.size(1))  
            padded_masif[:n_masif] = masif_feat  
            masif_desc_straights.append(padded_masif)  
              
            # 生成掩码：True表示有效特征，False表示填充  
            masif_mask = torch.zeros(max_masif_features, dtype=torch.bool)  
            masif_mask[:n_masif] = True  
            masif_masks.append(masif_mask)  
        else:  
            # 如果没有 masif 特征，创建零张量和全False掩码  
            masif_desc_straights.append(torch.zeros(max_masif_features, 80))  
            masif_masks.append(torch.zeros(max_masif_features, dtype=torch.bool))  
      
    return {        
        'node_feat': torch.stack(node_feats),        
        'edge_feat': torch.stack(edge_feats),  
        'edge_index': torch.stack(edge_indices),  
        'edge_mask': torch.stack(edge_masks),  
        'coords': torch.stack(coords_list),        
        'attn_bias': torch.stack(attn_biases),        
        'in_degree': torch.stack(in_degrees),        
        'out_degree': torch.stack(out_degrees),        
        'num_ligand_atoms': torch.tensor([item['num_ligand_atoms'] for item in items]),        
        'gbscore': torch.stack([item['gbscore'] for item in items]),        
        'masif_desc_straight': torch.stack(masif_desc_straights),  # 填充后的 masif 特征  
        'masif_mask': torch.stack(masif_masks),  # 新增：masif 特征掩码  
        'pdbid': [item['pdbid'] for item in items],        
        'pk': torch.tensor([item['pk'] for item in items]),        
        'smiles': [item['smiles'] for item in items],      
        'rmsd': torch.tensor([item['rmsd'] for item in items], dtype=torch.float),      
        'angle': torch.stack([a for a in angles if a is not None]) if any(a is not None for a in angles) else None,        
        'dists': torch.stack([d for d in dists_list if d is not None]) if any(d is not None for d in dists_list) else None,        
    }
  
  
# def create_affincraft_dataloader(pkl_files, batch_size=16, shuffle=True):    
#     """创建AffinCraft数据加载器"""    
#     dataset = AffinCraftDataset(pkl_files)    
        
#     return torch.utils.data.DataLoader(    
#         dataset,    
#         batch_size=batch_size,    
#         shuffle=shuffle,    
#         collate_fn=affincraft_collator,    
#         num_workers=8    
#     )
