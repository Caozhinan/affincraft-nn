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
          
        # 3. MaSIF特征处理网络（参考原模型的指纹处理方式）  
        if use_masif:  
            self.masif_encoder = nn.Sequential(  
                nn.Linear(80, hidden_dim),  # masif_desc_straight/flipped 维度  
                nn.GELU(),  
                nn.Linear(hidden_dim, hidden_dim)  
            )  
          
        # 4. GB-Score特征处理网络  
        if use_gbscore:  
            self.gbscore_encoder = nn.Sequential(  
                nn.Linear(400, hidden_dim),  # gbscore 维度  
                nn.GELU(),   
                nn.Linear(hidden_dim, hidden_dim)  
            )  
          
        # 5. 特征融合网络（参考原模型的reducer设计）  
        fusion_input_dim = hidden_dim  
        if use_masif:  
            fusion_input_dim += hidden_dim  
        if use_gbscore:  
            fusion_input_dim += hidden_dim  
              
        self.feature_fusion = nn.Linear(fusion_input_dim, hidden_dim)  
          
        self.use_masif = use_masif  
        self.use_gbscore = use_gbscore  
          
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
      
    def forward(self, batched_data):  
        # 获取基本信息  
        node_feat = batched_data["node_feat"]  # [n_graph, n_node, 9]  
        n_graph, n_node = node_feat.size()[:2]  
          
        # 1. 处理基础节点特征  
        node_features = self.node_encoder(node_feat.float())  # [n_graph, n_node, hidden_dim]  
          
        # 2. 创建图token特征  
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)  
          
        # 3. 处理全局特征（MaSIF和GB-Score）  
        global_features = []  
          
        # 处理MaSIF特征  
        if self.use_masif and "masif_desc_straight" in batched_data:  
            masif_feat = batched_data["masif_desc_straight"]  # [n_graph, n_patches, 80]  
            # 对MaSIF特征进行全局池化，得到图级别的表示  
            masif_global = masif_feat.mean(dim=1)  # [n_graph, 80]  
            masif_emb = self.masif_encoder(masif_global)  # [n_graph, hidden_dim]  
            global_features.append(masif_emb)  
          
        # 处理GB-Score特征  
        if self.use_gbscore and "gbscore" in batched_data:  
            gbscore_feat = batched_data["gbscore"]  # [n_graph, 400]  
            gbscore_emb = self.gbscore_encoder(gbscore_feat.float())  # [n_graph, hidden_dim]  
            global_features.append(gbscore_emb)  
          
        # 4. 融合全局特征到图token中（参考原模型的做法）  
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
    增强的边特征Embedding层，专门处理不同类型的边  
    """  
    def __init__(self, num_heads=32, hidden_dim=768, n_layers=12):  
        super().__init__()  
        self.num_heads = num_heads  
          
        # 1. 共价键（结构边）embedding  
        self.structural_edge_encoder = nn.Embedding(20, num_heads, padding_idx=0)  # 支持多种键类型  
          
        # 2. PLIP相互作用边embedding - 分别处理不同位置的相互作用  
        self.plip_intra_protein_encoder = nn.Embedding(15, num_heads, padding_idx=0)  # 蛋白质内部PLIP  
        self.plip_intra_ligand_encoder = nn.Embedding(15, num_heads, padding_idx=0)   # 配体内部PLIP    
        self.plip_inter_molecular_encoder = nn.Embedding(15, num_heads, padding_idx=0) # 分子间PLIP  
          
        # 3. 距离编码器  
        self.distance_encoder = nn.Sequential(  
            nn.Linear(1, num_heads),  
            nn.ReLU(),  
            nn.Linear(num_heads, num_heads)  
        )  
          
        # 4. 边位置类型编码器（区分边的位置）  
        self.edge_location_encoder = nn.Embedding(4, num_heads)  # 配体内部、蛋白内部、分子间、其他  
          
        # 5. 图token虚拟距离  
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)  
          
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
        edge_feat = batched_data["edge_feat"]  # [n_graph, max_edge_num, 4]  
        edge_index = batched_data["edge_index"]  # [n_graph, 2, max_edge_num]  
        edge_mask = batched_data.get("edge_mask")  # [n_graph, max_edge_num]  
        num_ligand_atoms = batched_data["num_ligand_atoms"]  
        attn_bias = batched_data.get("attn_bias")  
          
        n_graph, max_edge_num, _ = edge_feat.size()  
        n_node = batched_data["node_feat"].size(1)  

        # 初始化注意力偏置矩阵  
        if attn_bias is None:  
            attn_bias = torch.zeros([n_graph, n_node + 1, n_node + 1], dtype=torch.float, device=edge_feat.device)  

        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  

        # 分离边类型编码和距离  
        edge_types = edge_feat[:, :, :3].long()  # [n_graph, max_edge_num, 3]  
        distances = edge_feat[:, :, 3:4]         # [n_graph, max_edge_num, 1]  

        # 为每条边分类位置类型和相互作用类型  
        edge_embeddings = self._classify_and_embed_edges(  
            edge_index, edge_types, distances, num_ligand_atoms, edge_mask  
        )  # [n_graph, max_edge_num, num_heads]  

        # 将边embedding应用到注意力偏置矩阵（只处理真实边）  
        for batch_idx in range(n_graph):  
            for edge_idx in range(max_edge_num):  
                # 检查边掩码，只处理真实边  
                if edge_mask is not None and not edge_mask[batch_idx, edge_idx]:  
                    continue  

                src_idx = edge_index[batch_idx, 0, edge_idx] + 1  
                tgt_idx = edge_index[batch_idx, 1, edge_idx] + 1  

                # 确保索引有效且不是padding边  
                if (src_idx < n_node + 1 and tgt_idx < n_node + 1 and   
                    src_idx > 0 and tgt_idx > 0):  # 避免指向padding节点  
                    graph_attn_bias[batch_idx, :, src_idx, tgt_idx] += edge_embeddings[batch_idx, edge_idx]  
                    graph_attn_bias[batch_idx, :, tgt_idx, src_idx] += edge_embeddings[batch_idx, edge_idx]  

        # 添加图token虚拟距离  
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)  
        graph_attn_bias[:, :, 1:, 0] += t  
        graph_attn_bias[:, :, 0, :] += t  

        return graph_attn_bias  
      
    def _classify_and_embed_edges(self, edge_index, edge_types, distances, num_ligand_atoms, edge_mask=None):  
        """分类边并生成对应的embedding"""  
        n_graph, max_edge_num, _ = edge_types.size()  
        edge_embeddings = torch.zeros(n_graph, max_edge_num, self.num_heads, device=edge_types.device)  

        for batch_idx in range(n_graph):  
            num_lig = num_ligand_atoms[batch_idx].item()  

            for edge_idx in range(max_edge_num):  
                # 检查边掩码，只处理真实边  
                if edge_mask is not None and not edge_mask[batch_idx, edge_idx]:  
                    continue  

                src_idx = edge_index[batch_idx, 0, edge_idx].item()  
                tgt_idx = edge_index[batch_idx, 1, edge_idx].item()  

                # 确保边索引有效，避免处理padding边  
                if src_idx == 0 and tgt_idx == 0:  # padding边的标识  
                    continue  

                edge_type_code = edge_types[batch_idx, edge_idx]  # [3]  
                distance = distances[batch_idx, edge_idx]  # [1]  

                # 判断边的位置类型  
                src_is_ligand = src_idx < num_lig  
                tgt_is_ligand = tgt_idx < num_lig  

                # 距离编码  
                distance_emb = self.distance_encoder(distance.float())  

                # 根据边类型编码和位置选择合适的embedding  
                if edge_type_code[0] in [0, 1]:  # 结构边（共价键）  
                    # 将3维编码转换为单一索引  
                    edge_type_idx = edge_type_code[0] * 4 + edge_type_code[1] * 2 + edge_type_code[2]  
                    type_emb = self.structural_edge_encoder(edge_type_idx)  

                elif edge_type_code[0] == 5:  # PLIP相互作用边  
                    # PLIP类型索引 (5,1,0) -> 1, (5,2,0) -> 2, etc.  
                    plip_type_idx = edge_type_code[1]  

                    if src_is_ligand and tgt_is_ligand:  
                        # 配体内部空间边  
                        type_emb = self.plip_intra_ligand_encoder(plip_type_idx)  
                        location_emb = self.edge_location_encoder(torch.tensor(0, device=edge_types.device))  
                    elif not src_is_ligand and not tgt_is_ligand:  
                        # 蛋白质内部空间边  
                        type_emb = self.plip_intra_protein_encoder(plip_type_idx)  
                        location_emb = self.edge_location_encoder(torch.tensor(1, device=edge_types.device))  
                    else:  
                        # 蛋白-配体相互作用边  
                        type_emb = self.plip_inter_molecular_encoder(plip_type_idx)  
                        location_emb = self.edge_location_encoder(torch.tensor(2, device=edge_types.device))  

                    type_emb = type_emb + location_emb  

                else:  # 其他类型边  
                    type_emb = torch.zeros(self.num_heads, device=edge_types.device)  

                # 组合类型embedding和距离embedding  
                edge_embeddings[batch_idx, edge_idx] = type_emb + distance_emb  

        return edge_embeddings

def preprocess_affincraft_item(pkl_data):  
    """处理AffinCraft PKL文件，适配新的embedding层"""  
      
    # 基础数据  
    node_feat = torch.from_numpy(pkl_data['node_feat'])  
    edge_index = torch.from_numpy(pkl_data['edge_index'])  
    edge_feat = torch.from_numpy(pkl_data['edge_feat'])  
    coords = torch.from_numpy(pkl_data['coords'])  
      
    # 创建基础注意力偏置矩阵  
    n_node = node_feat.size(0)  
    attn_bias = torch.zeros([n_node + 1, n_node + 1], dtype=torch.float)  
      
    return {  
        'node_feat': node_feat,  
        'edge_index': edge_index.T,  # 转置为 [2, n_edge]  
        'edge_feat': edge_feat,  
        'coords': coords,  
        'attn_bias': attn_bias,  
        'num_ligand_atoms': torch.tensor([pkl_data['num_node'][0]]),  
        'gbscore': torch.from_numpy(pkl_data['gbscore']),  
        'masif_desc_straight': torch.from_numpy(pkl_data['masif_desc_straight'])  
    }