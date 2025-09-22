import pickle    
import torch    
import numpy as np    
from pathlib import Path    
from typing import List, Dict, Any, Optional    
from torch.utils.data import Dataset, DataLoader    
import mmap
import os
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
        if len(self._batch_cache) < 10:  # 最多缓存10个批次    
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

class OptimizedBatchedLazyAffinCraftDataset(torch.utils.data.Dataset):  
    def __init__(self, pkl_file_path, batch_size=32):  
        self.pkl_file_path = pkl_file_path  
        self.batch_size = batch_size  
        self._length = None  
        self._file_positions = None  
        self._batch_cache = {}  
        self._mmap_file = None  
        self._build_index_optimized()  
      
    def _build_index_optimized(self):  
        """优化的索引构建 - 使用内存映射和采样策略"""  
        positions = []  
        file_size = os.path.getsize(self.pkl_file_path)  
          
        # 使用内存映射提高I/O效率  
        with open(self.pkl_file_path, 'rb') as f:  
            # 采样策略：每隔一定字节数采样一个位置  
            sample_interval = max(1024 * 1024, file_size // 10000)  # 1MB或文件大小的1/10000  
              
            current_pos = 0  
            sample_count = 0  
              
            while current_pos < file_size and sample_count < 1000:  # 最多采样1000个位置  
                f.seek(current_pos)  
                try:  
                    pos = f.tell()  
                    # 快速跳过pickle对象而不完全反序列化  
                    self._fast_skip_pickle(f)  
                    positions.append(pos)  
                    sample_count += 1  
                    current_pos += sample_interval  
                except (EOFError, pickle.UnpicklingError):  
                    break  
              
            # 如果采样不够，从文件开头精确构建前N个对象的索引  
            if len(positions) < 100:  
                f.seek(0)  
                positions = []  
                count = 0  
                while count < 1000:  # 只构建前1000个对象的精确索引  
                    pos = f.tell()  
                    try:  
                        pickle.load(f)  
                        positions.append(pos)  
                        count += 1  
                    except EOFError:  
                        break  
                    except Exception:  
                        break  
          
        self._file_positions = positions  
        # 估算总长度（基于采样）  
        if positions:  
            avg_object_size = (file_size - positions[0]) / len(positions)  
            estimated_total = int(file_size / avg_object_size)  
            self._length = min(estimated_total, len(positions) * 100)  # 保守估计  
        else:  
            self._length = 0  
              
        print(f"快速索引构建完成，采样了 {len(positions)} 个位置，估算总对象数: {self._length}")  
      
    def _fast_skip_pickle(self, f):  
        """快速跳过pickle对象而不完全反序列化"""  
        # 读取pickle协议头  
        opcode = f.read(1)  
        if not opcode:  
            raise EOFError  
          
        # 使用pickle的内部机制快速定位  
        f.seek(f.tell() - 1)  # 回退一个字节  
        unpickler = pickle.Unpickler(f)  
        unpickler.load()  # 这会跳过整个对象  
      
    def _get_dynamic_position(self, idx):  
        """动态计算文件位置 - 用于超出采样范围的索引"""  
        if idx < len(self._file_positions):  
            return self._file_positions[idx]  
          
        # 对于超出采样范围的索引，使用线性插值估算位置  
        if len(self._file_positions) >= 2:  
            # 基于已知位置估算  
            last_pos = self._file_positions[-1]  
            avg_size = last_pos / len(self._file_positions)  
            estimated_pos = int(avg_size * idx)  
              
            # 从估算位置开始搜索实际位置  
            return self._search_actual_position(estimated_pos, idx)  
          
        return 0  
      
    def _search_actual_position(self, start_pos, target_idx):  
        """从估算位置搜索实际的pickle对象位置"""  
        with open(self.pkl_file_path, 'rb') as f:  
            f.seek(max(0, start_pos - 1024))  # 从稍早位置开始搜索  
              
            current_idx = target_idx - 10  # 保守估计  
            while current_idx < target_idx:  
                try:  
                    pos = f.tell()  
                    pickle.load(f)  
                    current_idx += 1  
                    if current_idx == target_idx:  
                        return pos  
                except (EOFError, pickle.UnpicklingError):  
                    break  
              
            return start_pos  # 回退到估算位置  
      
    def __getitem__(self, idx):  
        # 动态获取位置  
        if idx < len(self._file_positions):  
            pos = self._file_positions[idx]  
        else:  
            pos = self._get_dynamic_position(idx)  
          
        # 批次加载逻辑保持不变  
        batch_id, start_idx, end_idx = self._get_batch_indices(idx)  
        batch_data = self._load_batch_from_position(batch_id, pos, start_idx, end_idx)  
          
        local_idx = idx - start_idx  
        pkl_data = batch_data[local_idx] if local_idx < len(batch_data) else batch_data[0]  
          
        return preprocess_affincraft_item(pkl_data)

def preprocess_affincraft_item(pkl_data):      
    """专门处理AffinCraft PKL文件的预处理函数"""      
          
    # 直接使用PKL中的特征      
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
            # 移除masif_前缀用作字典键    
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
        'angle': angle,  # 新增    
        'dists': dists,  # 新增       
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
    max_edge_num = max(item['edge_feat'].size(0) for item in items)  # 新增：计算最大边数      
          
    # 批处理特征      
    node_feats = []      
    edge_feats = []      
    edge_indices = []      
    edge_masks = []  # 新增：边掩码      
    coords_list = []      
    attn_biases = []      
    in_degrees = []      
    out_degrees = []      
    angles = []  # 添加angle列表    
    dists_list = []  # 添加dists列表    
          
    for item in items:      
        n_node = item['node_feat'].size(0)      
        n_edge = item['edge_feat'].size(0)      
              
        # 填充节点特征      
        padded_node_feat = torch.zeros(max_node_num, item['node_feat'].size(1))      
        padded_node_feat[:n_node] = item['node_feat']      
        node_feats.append(padded_node_feat)      
              
        # 填充边特征 - 关键修改      
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
            
        # 处理angle和distance特征    
        if 'angle' in item and item['angle'] is not None:      
            padded_angle = torch.zeros(max_node_num, max_node_num, item['angle'].size(-1))      
            padded_angle[:n_node, :n_node] = item['angle']      
            angles.append(padded_angle)      
                  
            padded_dists = torch.zeros(max_node_num, max_node_num, item['dists'].size(-1))      
            padded_dists[:n_node, :n_node] = item['dists']      
            dists_list.append(padded_dists)    
        else:    
            # 如果没有angle特征，添加None占位符    
            angles.append(None)    
            dists_list.append(None)    
    
    return {      
        'node_feat': torch.stack(node_feats),      
        'edge_feat': torch.stack(edge_feats),  # 改为tensor      
        'edge_index': torch.stack(edge_indices),  # 改为tensor      
        'edge_mask': torch.stack(edge_masks),  # 新增边掩码      
        'coords': torch.stack(coords_list),      
        'attn_bias': torch.stack(attn_biases),      
        'in_degree': torch.stack(in_degrees),      
        'out_degree': torch.stack(out_degrees),      
        'num_ligand_atoms': torch.tensor([item['num_ligand_atoms'] for item in items]),      
        'gbscore': torch.stack([item['gbscore'] for item in items]),      
        'masif_desc_straight': torch.stack([item['masif_features']['desc_straight'] for item in items]),      
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