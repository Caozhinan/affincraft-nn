import math  # 导入数学库，主要用于开方等数学操作

import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入PyTorch的神经网络模块

def init_params(module, n_layers):
    # 参数初始化函数。对不同类型的模块采用不同初始化方式
    if isinstance(module, nn.Linear):
        # 对线性层（全连接层）权重初始化为均值0，方差与层数相关
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            # 偏置初始化为0
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        # 嵌入层权重初始化为均值0，标准差0.02
        module.weight.data.normal_(mean=0.0, std=0.02)

class AffinCraftNodeFeature(nn.Module):    
    def __init__(self, node_feat_dim=9, hidden_dim=768, n_layers=12, use_masif=True, use_gbscore=True):    
        super().__init__()    
            
        # 1. 基础节点特征编码器（处理9维节点特征）    
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)    
            
        # 2. 图token    
        self.graph_token = nn.Embedding(1, hidden_dim)    
            
        # 3. MaSIF特征处理网络 - 分层池化版本  
        if use_masif:    
            # 分层池化参数  
            self.local_pool_size = 64  # 局部池化窗口大小  
            self.local_pool_stride = 32  # 局部池化步长  
              
            # 局部特征编码器  
            self.local_masif_encoder = nn.Sequential(  
                nn.Linear(80, hidden_dim // 2),  # masif_desc_straight 维度  
                nn.GELU(),  
                nn.Linear(hidden_dim // 2, hidden_dim // 2)  
            )  
              
            # 全局特征编码器  
            self.global_masif_encoder = nn.Sequential(  
                nn.Linear(hidden_dim // 2, hidden_dim),  
                nn.GELU(),  
                nn.Linear(hidden_dim, hidden_dim)  
            )  
              
            # 注意力权重计算（可选）  
            self.attention_weights = nn.Linear(hidden_dim // 2, 1)  
            
        # 4. GB-Score特征处理网络    
        if use_gbscore:    
            self.gbscore_encoder = nn.Sequential(    
                nn.Linear(400, hidden_dim),  # gbscore 维度    
                nn.GELU(),     
                nn.Linear(hidden_dim, hidden_dim)    
            )    
            
        # 5. 特征融合网络  
        fusion_input_dim = hidden_dim    
        if use_masif:    
            fusion_input_dim += hidden_dim    
        if use_gbscore:    
            fusion_input_dim += hidden_dim    
                
        self.feature_fusion = nn.Linear(fusion_input_dim, hidden_dim)    
            
        self.use_masif = use_masif    
        self.use_gbscore = use_gbscore  
        self.hidden_dim = hidden_dim  
            
        # 初始化参数    
        self.apply(lambda module: self._init_params(module, n_layers))    
        
    def _init_params(self, module, n_layers):    
        """参考原模型的初始化方式"""    
        if isinstance(module, nn.Linear):    
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))    
            if module.bias is not None:    
                module.bias.data.zero_()    
        if isinstance(module, nn.Embedding):    
            module.weight.data.normal_(mean=0.0, std=0.02)  
      
    def hierarchical_pool_masif_vectorized(self, masif_feat, masif_mask=None):  
        n_graph, n_patches, feat_dim = masif_feat.shape  
        weight_dtype = self.local_masif_encoder[0].weight.dtype  
        masif_feat = masif_feat.to(weight_dtype)  

        # 使用 unfold 实现滑动窗口(完全 GPU 并行)  
        # [n_graph, n_patches, 80] -> [n_graph, 80, n_patches]  
        masif_transposed = masif_feat.transpose(1, 2)  

        # unfold: [n_graph, 80, n_patches] -> [n_graph, 80, n_windows, window_size]  
        windows = masif_transposed.unfold(2, self.local_pool_size, self.local_pool_stride)  
        # -> [n_graph, n_windows, 80, window_size]  
        windows = windows.permute(0, 2, 1, 3)  

        if masif_mask is not None:  
            # [n_graph, n_patches] -> [n_graph, n_windows, window_size]  
            mask_windows = masif_mask.unfold(1, self.local_pool_size, self.local_pool_stride)  
            # -> [n_graph, n_windows, 1, window_size] 用于广播  
            mask_windows = mask_windows.unsqueeze(2).to(weight_dtype)  

            # 掩码感知池化(批量)  
            masked_windows = windows * mask_windows  # [n_graph, n_windows, 80, window_size]  
            # valid_counts: [n_graph, n_windows, 1, 1]  
            valid_counts = mask_windows.sum(dim=-1, keepdim=True).clamp(min=1.0)  
            # 修正: 在 sum 之后调整 valid_counts 的形状  
            local_pooled = masked_windows.sum(dim=-1) / valid_counts.squeeze(-1)  # [n_graph, n_windows, 80]  
        else:  
            local_pooled = windows.mean(dim=-1)  # [n_graph, n_windows, 80]  

        # 批量通过局部编码器  
        n_windows = local_pooled.size(1)  
        local_pooled_flat = local_pooled.reshape(-1, feat_dim)  # [n_graph*n_windows, 80]  
        local_encoded_flat = self.local_masif_encoder(local_pooled_flat)  # [n_graph*n_windows, hidden_dim//2]  
        local_encoded = local_encoded_flat.reshape(n_graph, n_windows, -1)  # [n_graph, n_windows, hidden_dim//2]  

        # 全局池化  
        global_pooled = self.attention_based_global_pool(local_encoded)  
        final_encoded = self.global_masif_encoder(global_pooled)  

        return final_encoded  
      
    def attention_based_global_pool(self, local_features):  
        """  
        基于注意力的全局池化  
        Args:  
            local_features: [n_graph, n_local_regions, hidden_dim//2]  
        Returns:  
            pooled: [n_graph, hidden_dim//2]  
        """   
        n_graph, n_regions, feat_dim = local_features.shape  
        # 计算注意力权重 - 确保类型匹配  
        attention_scores = self.attention_weights(local_features).squeeze(-1)  # [n_graph, n_regions]  
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [n_graph, n_regions]  

        # 加权池化  
        weighted_features = local_features * attention_weights.unsqueeze(-1)  
        global_pooled = weighted_features.sum(dim=1)  # [n_graph, hidden_dim//2]  

        return global_pooled  
        
    def forward(self, batched_data):    
        # 获取基本信息    
        node_feat = batched_data["node_feat"]  # [n_graph, n_node, 9]    
        n_graph, n_node = node_feat.size()[:2]    

        # 获取模型权重的数据类型，确保输入数据类型匹配  
        weight_dtype = self.node_encoder.weight.dtype  

        # 1. 处理基础节点特征 - 使用动态类型匹配  
        node_features = self.node_encoder(node_feat.to(weight_dtype))  # [n_graph, n_node, hidden_dim]    

        # 2. 创建图token特征    
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)    

        # 3. 处理全局特征（MaSIF和GB-Score）    
        global_features = []    

        # 处理MaSIF特征 - 使用分层池化  
        if self.use_masif and "masif_desc_straight" in batched_data:    
            masif_feat = batched_data["masif_desc_straight"]  # [n_graph, n_patches, 80]  
            masif_mask = batched_data.get("masif_mask", None)  # [n_graph, n_patches]  

            # 使用分层池化  
            masif_emb = self.hierarchical_pool_masif_vectorized(masif_feat, masif_mask)  
            global_features.append(masif_emb)    

        # 处理GB-Score特征 - 使用动态类型匹配  
        if self.use_gbscore and "gbscore" in batched_data:    
            gbscore_feat = batched_data["gbscore"]  # [n_graph, 400]    
            gbscore_emb = self.gbscore_encoder(gbscore_feat.to(weight_dtype))  # 匹配权重类型  
            global_features.append(gbscore_emb)    

        # 4. 融合全局特征到图token中  
        if global_features:    
            # 将图token特征与全局特征拼接    
            graph_token_flat = graph_token_feature.squeeze(1)  # [n_graph, hidden_dim]    
            fusion_input = torch.cat([graph_token_flat] + global_features, dim=1)    

            # 通过融合网络处理    
            fused_graph_token = self.feature_fusion(fusion_input)  # [n_graph, hidden_dim]    
            graph_token_feature = fused_graph_token.unsqueeze(1)  # [n_graph, 1, hidden_dim]    

        # 5. 拼接图token和节点特征    
        graph_node_feature = torch.cat([graph_token_feature, node_features], dim=1)    

        return graph_node_feature  # [n_graph, n_node+1, hidden_dim]

class AffinCraftAttnBias(nn.Module):      
    """      
    增强的边特征Embedding层,专门处理不同类型的边,现在包含angle和distance特征      
    完全向量化的GPU实现  
    """      
    def __init__(self, num_heads=32, hidden_dim=768, n_layers=12):      
        super().__init__()      
        self.num_heads = num_heads      
              
        # 1. 共价键(结构边)embedding      
        self.structural_edge_encoder = nn.Embedding(20, num_heads, padding_idx=0)  
              
        # 2. PLIP相互作用边embedding - 分别处理不同位置的相互作用      
        self.plip_intra_protein_encoder = nn.Embedding(15, num_heads, padding_idx=0)  
        self.plip_intra_ligand_encoder = nn.Embedding(15, num_heads, padding_idx=0)  
        self.plip_inter_molecular_encoder = nn.Embedding(15, num_heads, padding_idx=0)  
              
        # 3. 距离编码器      
        self.distance_encoder = nn.Sequential(      
            nn.Linear(1, num_heads),      
            nn.ReLU(),      
            nn.Linear(num_heads, num_heads)      
        )      
              
        # 4. 边位置类型编码器(区分边的位置)      
        self.edge_location_encoder = nn.Embedding(4, num_heads)  
              
        # 5. 图token虚拟距离      
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)      
            
        # 6. angle和distance特征编码器    
        self.angle_encoder = nn.Sequential(    
            nn.Linear(28, num_heads),  
            nn.ReLU(),    
            nn.Linear(num_heads, num_heads)    
        )    
            
        self.multi_dist_encoder = nn.Sequential(    
            nn.Linear(28, num_heads),  
            nn.ReLU(),     
            nn.Linear(num_heads, num_heads)    
        )    
              
        # 初始化参数      
        self.apply(lambda module: self._init_params(module, n_layers))      
          
    def _init_params(self, module, n_layers):      
        """参数初始化"""      
        if isinstance(module, nn.Linear):      
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))      
            if module.bias is not None:      
                module.bias.data.zero_()      
        if isinstance(module, nn.Embedding):      
            module.weight.data.normal_(mean=0.0, std=0.02)      
          
    def forward(self, batched_data):      
        edge_feat = batched_data["edge_feat"]  
        edge_index = batched_data["edge_index"]  
        edge_mask = batched_data.get("edge_mask")  
        num_ligand_atoms = batched_data["num_ligand_atoms"]  
        attn_bias = batched_data.get("attn_bias")  

        angle = batched_data.get("angle")  
        dists = batched_data.get("dists")  

        n_graph, max_edge_num, _ = edge_feat.size()  
        n_node = batched_data["node_feat"].size(1)  
    
        # 获取模型权重的数据类型,确保 attn_bias 类型匹配  
        weight_dtype = self.structural_edge_encoder.weight.dtype  

        # 初始化注意力偏置矩阵 - 使用与模型权重相同的类型  
        if attn_bias is None:  
            attn_bias = torch.zeros([n_graph, n_node + 1, n_node + 1], dtype=weight_dtype, device=edge_feat.device)  
        else:  
            attn_bias = attn_bias.to(weight_dtype)  
    
        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  
    
        # 分离边类型编码和距离  
        edge_types = edge_feat[:, :, :3].long()  
        distances = edge_feat[:, :, 3:4]  
    
        # 为每条边分类位置类型和相互作用类型  
        edge_embeddings = self._classify_and_embed_edges_vectorized(  
            edge_index, edge_types, distances, num_ligand_atoms, edge_mask  
        )  
    
        # 向量化的注意力偏置更新  
        self._update_attn_bias_vectorized(  
            graph_attn_bias, edge_index, edge_embeddings, edge_mask, n_node  
        )  
    
        # 添加 angle 和 distance 特征  
        if angle is not None and dists is not None:  
            angle_emb = self.angle_encoder(angle.to(weight_dtype))  
            dist_emb = self.multi_dist_encoder(dists.to(weight_dtype))  
            graph_attn_bias[:, :, 1:, 1:] += (angle_emb + dist_emb).permute(0, 3, 1, 2)  
    
        # 添加图token虚拟距离  
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)  
        graph_attn_bias[:, :, 1:, 0] += t  
        graph_attn_bias[:, :, 0, :] += t  
    
        return graph_attn_bias      
          
    def _classify_and_embed_edges_vectorized(self, edge_index, edge_types, distances, num_ligand_atoms, edge_mask=None):  
        """完全向量化的边分类和embedding生成"""  
        n_graph, max_edge_num, _ = edge_types.size()  

        # 获取模型权重类型  
        weight_dtype = self.structural_edge_encoder.weight.dtype  

        # 批量获取源节点和目标节点索引  
        src_idx = edge_index[:, 0, :]  # [n_graph, max_edge_num]  
        tgt_idx = edge_index[:, 1, :]  # [n_graph, max_edge_num]  

        # 安全处理 num_ligand_atoms,确保至少为 1  
        num_lig_expanded = torch.clamp(num_ligand_atoms, min=1).unsqueeze(1)  # [n_graph, 1]  

        # 批量判断是否为配体原子,添加边界检查  
        src_is_ligand = (src_idx > 0) & (src_idx < num_lig_expanded)  # [n_graph, max_edge_num]  
        tgt_is_ligand = (tgt_idx > 0) & (tgt_idx < num_lig_expanded)  

        # 批量距离编码 - 确保输出类型正确  
        distance_emb = self.distance_encoder(distances.to(weight_dtype))  # [n_graph, max_edge_num, num_heads]  

        # 批量处理结构边  
        is_structural = (edge_types[:, :, 0] <= 1)  # [n_graph, max_edge_num]  
        structural_idx = torch.clamp(  
            edge_types[:, :, 0] * 4 + edge_types[:, :, 1] * 2 + edge_types[:, :, 2],  
            min=0, max=19  
        )  
        structural_emb = self.structural_edge_encoder(structural_idx)  # [n_graph, max_edge_num, num_heads]  

        # 批量处理PLIP边  
        is_plip = (edge_types[:, :, 0] == 5)  
        plip_type_idx = torch.clamp(edge_types[:, :, 1], min=0, max=14)  

        # 根据位置选择不同的PLIP encoder  
        both_ligand = src_is_ligand & tgt_is_ligand  
        both_protein = (~src_is_ligand) & (~tgt_is_ligand)  
        inter_molecular = ~(both_ligand | both_protein)  

        # 初始化PLIP embedding - 使用正确的数据类型  
        plip_emb = torch.zeros(n_graph, max_edge_num, self.num_heads, device=edge_types.device, dtype=weight_dtype)  

        # 使用masked indexing批量处理  
        if both_ligand.any():  
            valid_indices = both_ligand & (plip_type_idx >= 0) & (plip_type_idx < 15)  
            if valid_indices.any():  
                plip_emb[valid_indices] = self.plip_intra_ligand_encoder(plip_type_idx[valid_indices]).to(weight_dtype)  

        if both_protein.any():  
            valid_indices = both_protein & (plip_type_idx >= 0) & (plip_type_idx < 15)  
            if valid_indices.any():  
                plip_emb[valid_indices] = self.plip_intra_protein_encoder(plip_type_idx[valid_indices]).to(weight_dtype)  

        if inter_molecular.any():  
            valid_indices = inter_molecular & (plip_type_idx >= 0) & (plip_type_idx < 15)  
            if valid_indices.any():  
                plip_emb[valid_indices] = self.plip_inter_molecular_encoder(plip_type_idx[valid_indices]).to(weight_dtype)  

        # 组合所有embedding - 确保所有张量类型一致  
        structural_emb = structural_emb.to(weight_dtype)  
        edge_embeddings = torch.where(  
            is_structural.unsqueeze(-1),  
            structural_emb,  
            torch.where(is_plip.unsqueeze(-1), plip_emb, torch.zeros_like(distance_emb))  
        ) + distance_emb  

        # 应用edge_mask  
        if edge_mask is not None:  
            edge_embeddings = edge_embeddings * edge_mask.unsqueeze(-1).float()  

        return edge_embeddings  
      
def _update_attn_bias_vectorized(self, graph_attn_bias, edge_index, edge_embeddings, edge_mask, n_node):    
    """向量化的注意力偏置更新"""    
    n_graph, max_edge_num, num_heads = edge_embeddings.shape    
        
    # 获取 graph_attn_bias 的数据类型    
    target_dtype = graph_attn_bias.dtype    
        
    # 确保 edge_embeddings 与 graph_attn_bias 类型一致    
    edge_embeddings = edge_embeddings.to(target_dtype)    
        
    # 获取源和目标索引(+1是因为图token在位置0)    
    src_idx = edge_index[:, 0, :] + 1  # [n_graph, max_edge_num]    
    tgt_idx = edge_index[:, 1, :] + 1    
        
    # 创建有效边的mask    
    if edge_mask is not None:    
        valid_mask = edge_mask.clone()    
    else:    
        valid_mask = torch.ones(n_graph, max_edge_num, dtype=torch.bool, device=edge_embeddings.device)    
        
    # 过滤掉padding边和越界边    
    valid_mask = valid_mask & (src_idx > 0) & (tgt_idx > 0)    
    valid_mask = valid_mask & (src_idx <= n_node) & (tgt_idx <= n_node)    
        
    # 为每个head批量更新    
    for h in range(num_heads):    
        # 获取当前head的embedding    
        head_emb = edge_embeddings[:, :, h]  # [n_graph, max_edge_num]    
            
        # 应用mask - 修复：使用 target_dtype 而不是 .float()  
        masked_emb = head_emb * valid_mask.to(target_dtype)  # 关键修改  
            
        # 批量scatter操作    
        for b in range(n_graph):    
            valid_edges = valid_mask[b]    
            if valid_edges.any():    
                src = src_idx[b, valid_edges]    
                tgt = tgt_idx[b, valid_edges]    
                emb = masked_emb[b, valid_edges]    
                    
                # 正向边: src -> tgt    
                graph_attn_bias[b, h].index_put_((src, tgt), emb, accumulate=True)    
                # 反向边: tgt -> src    
                graph_attn_bias[b, h].index_put_((tgt, src), emb, accumulate=True)


# def preprocess_affincraft_item(pkl_data):  
#     """处理AffinCraft PKL文件，适配新的embedding层"""  
      
#     # 基础数据  
#     node_feat = torch.from_numpy(pkl_data['node_feat'])  
#     edge_index = torch.from_numpy(pkl_data['edge_index'])  
#     edge_feat = torch.from_numpy(pkl_data['edge_feat'])  
#     coords = torch.from_numpy(pkl_data['coords'])  
      
#     # 创建基础注意力偏置矩阵  
#     n_node = node_feat.size(0)  
#     attn_bias = torch.zeros([n_node + 1, n_node + 1], dtype=torch.float)  
      
#     return {  
#         'node_feat': node_feat,  
#         'edge_index': edge_index.T,  # 转置为 [2, n_edge]  
#         'edge_feat': edge_feat,  
#         'coords': coords,  
#         'attn_bias': attn_bias,  
#         'num_ligand_atoms': torch.tensor([pkl_data['num_node'][0]]),  
#         'gbscore': torch.from_numpy(pkl_data['gbscore']),  
#         'masif_desc_straight': torch.from_numpy(pkl_data['masif_desc_straight'])  
#     }