# 建议的文件路径: /data/run01/scw6f3q/zncao/affincraft-nn/graphormer/modules/graphormer_layers.py
import math
import torch
import torch.nn as nn

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
              
        # 基础组件  
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)      
        self.graph_token = nn.Embedding(1, hidden_dim)      
          
        if use_masif:    
            self.local_pool_size = 64    
            self.local_pool_stride = 32    
            self.feature_dim_unifier = nn.Linear(5, 64)    
              
            # 动态维度参数  
            self.unified_feature_dim = 64  
            self.target_spatial_dim = 300  
              
            # 局部编码器  
            self.protein_local_encoder = nn.Sequential(    
                nn.Linear(self.target_spatial_dim, hidden_dim // 2),    
                nn.GELU(),    
                nn.Linear(hidden_dim // 2, hidden_dim // 2)    
            )    
  
            self.ligand_local_encoder = nn.Sequential(    
                nn.Linear(self.target_spatial_dim, hidden_dim // 2),    
                nn.GELU(),    
                nn.Linear(hidden_dim // 2, hidden_dim // 2)    
            )    
  
            # 交叉注意力和池化层  
            self.cross_attention = nn.MultiheadAttention(    
                embed_dim=hidden_dim // 2,    
                num_heads=8,    
                batch_first=True    
            )    
  
            self.protein_attention = nn.Linear(hidden_dim // 2, 1)    
            self.ligand_attention = nn.Linear(hidden_dim // 2, 1)    
  
            self.global_masif_encoder = nn.Sequential(    
                nn.Linear(hidden_dim, hidden_dim),    
                nn.GELU(),    
                nn.Linear(hidden_dim, hidden_dim)    
            )    
              
        if use_gbscore:      
            self.gbscore_encoder = nn.Sequential(      
                nn.Linear(400, hidden_dim),      
                nn.GELU(),       
                nn.Linear(hidden_dim, hidden_dim)      
            )      
              
        # 特征融合网络  
        fusion_input_dim = hidden_dim      
        if use_masif:      
            fusion_input_dim += hidden_dim      
        if use_gbscore:      
            fusion_input_dim += hidden_dim      
                  
        self.feature_fusion = nn.Linear(fusion_input_dim, hidden_dim)      
              
        self.use_masif = use_masif      
        self.use_gbscore = use_gbscore    
        self.hidden_dim = hidden_dim    
        self.n_layers = n_layers  
              
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
  
    def _create_or_get_spatial_dim_unifier(self, spatial_dim, device):  
        """动态创建或获取spatial_dim_unifier层"""  
        layer_name = f'spatial_dim_unifier_{spatial_dim}'  
          
        if not hasattr(self, layer_name):  
            flattened_dim = self.unified_feature_dim * spatial_dim  
            layer = nn.Linear(flattened_dim, self.target_spatial_dim).to(device)  
            setattr(self, layer_name, layer)  
            # 初始化新创建的层  
            self._init_params(layer, self.n_layers)  
              
        return getattr(self, layer_name)  
  
    def hierarchical_pool_masif_vectorized(self, masif_feat, masif_mask=None):    
        n_graph, n_patches, feat_dim = masif_feat.shape    
        weight_dtype = self.protein_local_encoder[0].weight.dtype    
        device = self.protein_local_encoder[0].weight.device    
        masif_feat = masif_feat.to(weight_dtype).to(device)    
      
        # 使用 unfold 实现滑动窗口    
        masif_transposed = masif_feat.transpose(1, 2)    
        windows = masif_transposed.unfold(2, self.local_pool_size, self.local_pool_stride)    
        windows = windows.permute(0, 2, 1, 3)    
      
        if masif_mask is not None:    
            mask_windows = masif_mask.unfold(1, self.local_pool_size, self.local_pool_stride)    
            mask_windows = mask_windows.unsqueeze(2).to(weight_dtype).to(device)    
            masked_windows = windows * mask_windows    
            valid_counts = mask_windows.sum(dim=-1, keepdim=True).clamp(min=1.0)    
            local_pooled = masked_windows.sum(dim=-1) / valid_counts.squeeze(-1)    
        else:    
            local_pooled = windows.mean(dim=-1)    
      
        # 批量通过局部编码器    
        n_windows = local_pooled.size(1)    
        local_pooled_flat = local_pooled.reshape(-1, feat_dim)    
        local_encoded_flat = self.protein_local_encoder[0](local_pooled_flat)  # 使用第一层  
        local_encoded = local_encoded_flat.reshape(n_graph, n_windows, -1)    
      
        # 全局池化    
        global_pooled = self.attention_based_global_pool(local_encoded, self.protein_attention)    
        final_encoded = self.global_masif_encoder(global_pooled)    
      
        return final_encoded    
  
    def attention_based_global_pool(self, local_features, attention_layer):    
        """基于注意力的全局池化"""    
        attention_scores = attention_layer(local_features).squeeze(-1)    
        attention_weights = torch.softmax(attention_scores, dim=-1)    
        weighted_features = local_features * attention_weights.unsqueeze(-1)    
        return weighted_features.sum(dim=1)  
  
    def forward(self, batched_data):        
        # 获取基本信息  
        node_feat = batched_data["node_feat"]        
        n_graph, n_node = node_feat.size()[:2]        
  
        weight_dtype = self.node_encoder.weight.dtype  
        device = next(self.parameters()).device  
          
        # 统一设备  
        for key in batched_data:      
            if isinstance(batched_data[key], torch.Tensor):      
                batched_data[key] = batched_data[key].to(device)    
            
        # 1. 处理基础节点特征  
        node_features = self.node_encoder(node_feat.to(weight_dtype))        
  
        # 2. 创建图token特征  
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)        
  
        # 3. 处理全局特征  
        global_features = []        
           
        if self.use_masif and "protein_masif_feature" in batched_data and "ligand_masif_feature" in batched_data:    
            protein_feat = batched_data["protein_masif_feature"].to(device).requires_grad_(True)    
            ligand_feat = batched_data["ligand_masif_feature"].to(device).requires_grad_(True)    
  
            # 获取实际空间维度  
            protein_spatial_dim = protein_feat.size(2)  
            ligand_spatial_dim = ligand_feat.size(2)  
  
            # 步骤1: 维度统一处理  
            protein_unified = self.feature_dim_unifier(protein_feat)    
            ligand_unified = self.feature_dim_unifier(ligand_feat)    
  
            # 展平后两维  
            protein_flat = protein_unified.view(protein_unified.size(0), protein_unified.size(1), -1)    
            ligand_flat = ligand_unified.view(ligand_unified.size(0), ligand_unified.size(1), -1)    
  
            # 动态创建并使用spatial_dim_unifier  
            protein_spatial_dim_unifier = self._create_or_get_spatial_dim_unifier(protein_spatial_dim, device)  
            ligand_spatial_dim_unifier = self._create_or_get_spatial_dim_unifier(ligand_spatial_dim, device)  
  
            protein_processed = protein_spatial_dim_unifier(protein_flat)    
            ligand_processed = ligand_spatial_dim_unifier(ligand_flat)    
  
            # 步骤2: 分层池化  
            protein_pooled = self.hierarchical_pool_masif_vectorized(protein_processed)    
            ligand_pooled = self.hierarchical_pool_masif_vectorized(ligand_processed)    
  
            # 步骤3: 局部编码  
            protein_encoded = self.protein_local_encoder(protein_pooled)    
            ligand_encoded = self.ligand_local_encoder(ligand_pooled)    
  
            # 步骤4: 单向交叉注意力  
            ligand_cross, _ = self.cross_attention(    
                query=ligand_encoded,      
                key=protein_encoded,       
                value=protein_encoded      
            )    
  
            # 步骤5: 残差连接  
            ligand_enhanced = ligand_encoded + ligand_cross    
  
            # 步骤6: 分别全局池化  
            protein_global = self.attention_based_global_pool(protein_encoded, self.protein_attention)    
            ligand_global = self.attention_based_global_pool(ligand_enhanced, self.ligand_attention)    
  
            # 步骤7: 融合  
            masif_emb = torch.cat([protein_global, ligand_global], dim=-1)    
            masif_emb = self.global_masif_encoder(masif_emb)    
  
            global_features.append(masif_emb)    
  
        # 处理GB-Score特征  
        if self.use_gbscore and "gbscore" in batched_data:        
            gbscore_feat = batched_data["gbscore"]         
            gbscore_emb = self.gbscore_encoder(gbscore_feat.to(weight_dtype))      
            global_features.append(gbscore_emb)        
  
        # 4. 融合全局特征到图token中  
        if global_features:        
            graph_token_flat = graph_token_feature.squeeze(1)        
            fusion_input = torch.cat([graph_token_flat] + global_features, dim=1)        
            fused_graph_token = self.feature_fusion(fusion_input)        
            graph_token_feature = fused_graph_token.unsqueeze(1)      
  
        # 5. 拼接图token和节点特征  
        graph_node_feature = torch.cat([graph_token_feature, node_features], dim=1)        
  
        return graph_node_feature

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
        device = next(self.parameters()).device  
      
    # 确保所有边相关的输入张量都在正确设备上  
        if 'edge_index' in batched_data and isinstance(batched_data['edge_index'], torch.Tensor):  
            batched_data['edge_index'] = batched_data['edge_index'].to(device)  
        if 'edge_feat' in batched_data and isinstance(batched_data['edge_feat'], torch.Tensor):  
            batched_data['edge_feat'] = batched_data['edge_feat'].to(device)  
        if 'coords' in batched_data and isinstance(batched_data['coords'], torch.Tensor):  
            batched_data['coords'] = batched_data['coords'].to(device)  
        if 'angle' in batched_data and isinstance(batched_data['angle'], torch.Tensor):  
            batched_data['angle'] = batched_data['angle'].to(device)  
        if 'dists' in batched_data and isinstance(batched_data['dists'], torch.Tensor):  
            batched_data['dists'] = batched_data['dists'].to(device)  

        # 处理空间边特征  
        if 'lig_spatial_edges' in batched_data:  
            if isinstance(batched_data['lig_spatial_edges'], dict):  
                for key in batched_data['lig_spatial_edges']:  
                    if isinstance(batched_data['lig_spatial_edges'][key], torch.Tensor):  
                        batched_data['lig_spatial_edges'][key] = batched_data['lig_spatial_edges'][key].to(device)  

        if 'pro_spatial_edges' in batched_data:  
            if isinstance(batched_data['pro_spatial_edges'], dict):  
                for key in batched_data['pro_spatial_edges']:  
                    if isinstance(batched_data['pro_spatial_edges'][key], torch.Tensor):  
                        batched_data['pro_spatial_edges'][key] = batched_data['pro_spatial_edges'][key].to(device)      
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
      
    # ====================================================================
    # ==================== START OF OPTIMIZED CODE =======================
    # ====================================================================
    def _update_attn_bias_vectorized(self, graph_attn_bias, edge_index, edge_embeddings, edge_mask, n_node):
        """
        【优化版本】向量化的注意力偏置更新
        此版本消除了内部对 batch size 的Python循环，显著提升性能。
        """
        n_graph, max_edge_num, num_heads = edge_embeddings.shape
        target_dtype = graph_attn_bias.dtype
        device = graph_attn_bias.device

        # 确保 edge_embeddings 类型一致
        edge_embeddings = edge_embeddings.to(target_dtype)

        # 获取源和目标索引 (+1 是因为图token在位置0)
        src_idx = edge_index[:, 0, :] + 1  # [n_graph, max_edge_num]
        tgt_idx = edge_index[:, 1, :] + 1

        # 创建有效边的mask
        if edge_mask is None:
            edge_mask = torch.ones(n_graph, max_edge_num, dtype=torch.bool, device=device)

        # 过滤掉padding边和越界边
        valid_mask = edge_mask & (src_idx > 0) & (tgt_idx > 0)
        valid_mask = valid_mask & (src_idx <= n_node) & (tgt_idx <= n_node)

        # 创建批次索引，用于大规模的index_put
        # b_idx shape: [n_graph, max_edge_num]
        b_idx = torch.arange(n_graph, device=device).unsqueeze(1).expand(-1, max_edge_num)

        # 过滤出有效的索引和embedding值
        valid_b = b_idx[valid_mask]
        valid_src = src_idx[valid_mask]
        valid_tgt = tgt_idx[valid_mask]
        
        # 循环遍历 heads，这部分开销相对较小
        for h in range(num_heads):
            # 获取当前 head 的 embedding
            head_emb = edge_embeddings[:, :, h]  # [n_graph, max_edge_num]
            valid_emb = head_emb[valid_mask]

            # 使用批次索引(valid_b)进行一次性的大规模更新
            # 正向边: src -> tgt
            graph_attn_bias.index_put_(
                (valid_b, torch.full_like(valid_b, h), valid_src, valid_tgt),
                valid_emb,
                accumulate=True
            )
            # 反向边: tgt -> src
            graph_attn_bias.index_put_(
                (valid_b, torch.full_like(valid_b, h), valid_tgt, valid_src),
                valid_emb,
                accumulate=True
            )
    # ====================================================================
    # ===================== END OF OPTIMIZED CODE ========================
    # ====================================================================