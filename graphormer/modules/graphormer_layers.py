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


# ============================================================
#           AffinCraftNodeFeature（方案二：分层池化）
# ============================================================
class AffinCraftNodeFeature(nn.Module):
    """
    AffinCraft 的节点特征构造模块，带 MaSIF 分层池化（方案二）和 GBScore 融合。

    这里按你的需求做了两个关键点：
      1. 不管原始 S 是 40 还是 60，先把它们统一到 S_target = 60；
      2. 再做后面的分层池化 / 神经网络处理。

    MaSIF 分支总体流程：
        原始输入（collator 给出的）：
            ligand/protein_masif_feature: (B, L, S_raw, 5)

        → feature_dim_unifier: 5 -> 64    （特征维度统一）
        → _pad_spatial_to_target:
            - 在 S 维度上 pad/truncate 到固定 S_target=60
            - 得到 (B, L, 60, 64)
        → hierarchical_pool_masif_vectorized:
            - 在 S 维度上用 unfold 做滑动窗口 (window_size=60, stride=30)
            - 每个窗口 flatten 成 60 * 64 维向量
            - 用 protein_local_encoder / ligand_local_encoder 编码到 D = hidden_dim//2
            - 用 protein_attention / ligand_attention 在窗口维度做注意力池化
          得到 (B, L, D)
        → 蛋白 & 配体做单向 cross-attention（ligand query, protein key/value）
        → 在 L 维度上用 attention_based_global_pool 做全局注意力池化
          得到 protein_global, ligand_global: (B, D)
        → concat 后经 global_masif_encoder 变成 (B, hidden_dim)
        → 和 graph_token + gbscore 一起融合
    """

    def __init__(self, node_feat_dim=9, hidden_dim=768, n_layers=12,
                 use_masif=True, use_gbscore=False):
        super().__init__()

        # 基础组件
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.graph_token = nn.Embedding(1, hidden_dim)

        self.use_masif = use_masif
        self.use_gbscore = use_gbscore
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        if use_masif:
            # ------- MaSIF 特征处理参数（按你要求统一到 S_target = 60） -------
            self.local_pool_size = 60      # 滑动窗口大小（沿 spatial_dim = S_target）
            self.local_pool_stride = 30    # 滑动窗口步长，你可以按需改
            self.unified_feature_dim = 64  # 5 -> 64

            # 5D MaSIF 特征统一到 64 维
            self.feature_dim_unifier = nn.Linear(5, self.unified_feature_dim)

            # 一个窗口的向量长度：F_unified(64) * window_size(60)
            window_vec_dim = self.unified_feature_dim * self.local_pool_size  # 64 * 60

            # 局部编码器：把每个窗口编码到 hidden_dim // 2
            self.protein_local_encoder = nn.Sequential(
                nn.Linear(window_vec_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )
            self.ligand_local_encoder = nn.Sequential(
                nn.Linear(window_vec_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
            )

            # 交叉注意力层：配体 query，蛋白 key / value
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim // 2,
                num_heads=8,
                batch_first=True,
            )

            # 对窗口做注意力池化 & 对序列做全局池化的打分层
            self.protein_attention = nn.Linear(hidden_dim // 2, 1)
            self.ligand_attention = nn.Linear(hidden_dim // 2, 1)

            # 蛋白 + 配体 全局向量拼接后再编码回 hidden_dim
            self.global_masif_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if use_gbscore:
            self.gbscore_encoder = nn.Sequential(
                nn.Linear(400, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # 特征融合网络
        fusion_input_dim = hidden_dim  # 基础 graph_token 维度
        if use_masif:
            fusion_input_dim += hidden_dim  # masif_emb: hidden_dim
        if use_gbscore:
            fusion_input_dim += hidden_dim  # gbscore_emb: hidden_dim

        self.feature_fusion = nn.Linear(fusion_input_dim, hidden_dim)

        # 参数初始化
        self.apply(lambda module: self._init_params(module, n_layers))

    # ====================== 初始化 ======================
    def _init_params(self, module, n_layers):
        """参考原模型的初始化方式"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    # ====================== 辅助：统一 S 维度 ======================
    def _pad_spatial_to_target(self, masif_feat):
        """
        把 masif_feat 在 S 这一维（第 3 维）pad / truncate 到固定 target_S = local_pool_size（默认 60）。

        输入:
            masif_feat: (B, L, S, F)
        输出:
            masif_feat_out: (B, L, target_S, F)

        行为：
            - 如果 S == target_S: 原样返回；
            - 如果 S <  target_S: 在后面补 0；
            - 如果 S >  target_S: 截断到前 target_S 个位置（目前你的数据不会 >60，但为了安全加上）。
        """
        B, L, S, F = masif_feat.shape
        target_S = self.local_pool_size  # 这里就是你要的 S_target=60

        if S == target_S:
            return masif_feat
        elif S < target_S:
            pad_len = target_S - S
            pad = masif_feat.new_zeros(B, L, pad_len, F)  # [B, L, pad_len, F]
            masif_feat_out = torch.cat([masif_feat, pad], dim=2)  # 在 S 维度上拼接
        else:  # S > target_S
            masif_feat_out = masif_feat[:, :, :target_S, :]

        return masif_feat_out

    # ====================== 分层池化（方案二核心） ======================
    def hierarchical_pool_masif_vectorized(self, masif_feat, encoder, attn_layer, masif_mask=None):
        """
        分层池化 + 窗口级注意力池化。

        输入:
            masif_feat: (B, L, S, F_unified)  例如 F_unified=64，此时 S 已被统一到 60
            encoder:   protein_local_encoder 或 ligand_local_encoder
            attn_layer: protein_attention 或 ligand_attention
            masif_mask: (B, L, S) 或 None（目前 collator 未提供，可后续扩展）

        过程:
            (B, L, S, F) → (B*L, S, F)
            → 在 S 上 unfold 窗口: size=local_pool_size(60), stride=local_pool_stride(30)
                得 (B*L, num_windows, F, window_size)
            → 每个窗口 flatten 成向量 (F * window_size)
            → encoder 编码到 D = hidden_dim//2
                得 (B*L, num_windows, D)
            → 用 attn_layer 在 num_windows 维度做注意力池化
                得 (B*L, D)
            → reshape 回 (B, L, D)
        输出:
            (B, L, hidden_dim // 2)
        """
        B, L, S, F = masif_feat.shape
        # 用 encoder 第一层的 dtype / device 对齐
        first_linear = encoder[0]
        weight_dtype = first_linear.weight.dtype
        device = first_linear.weight.device

        x = masif_feat.to(weight_dtype).to(device)  # [B, L, S, F]
        # (B, L, S, F) -> (B*L, S, F)
        x = x.view(-1, S, F)  # [BL, S, F]
        BL = x.size(0)

        # 在 S 上做滑动窗口
        # x_t: [BL, F, S]
        x_t = x.transpose(1, 2)
        # unfold: [BL, F, num_windows, window_size]
        windows = x_t.unfold(2, self.local_pool_size, self.local_pool_stride)
        # 调整为 [BL, num_windows, F, window_size]
        windows = windows.permute(0, 2, 1, 3).contiguous()
        BL_, num_windows, F_, W = windows.shape
        assert BL_ == BL and F_ == F and W == self.local_pool_size, \
            f"Unexpected window shape: {windows.shape}, expect (*, *, {F}, {self.local_pool_size})"

        if masif_mask is not None:
            # 如果将来有 mask（B, L, S），可在这里对 windows 做加权 / 归一化
            pass

        # 每个窗口 flatten 成 (F * window_size) 向量
        windows_flat = windows.reshape(BL * num_windows, F * W)  # [BL*num_windows, F*W]

        # 用对应分子的局部编码器编码到 hidden_dim//2
        encoded_flat = encoder(windows_flat)  # [BL*num_windows, D]
        D = encoded_flat.size(-1)

        local_encoded = encoded_flat.view(BL, num_windows, D)  # [BL, num_windows, D]

        # 窗口维度(num_windows)上的注意力池化
        attn_scores = attn_layer(local_encoded).squeeze(-1)        # [BL, num_windows]
        attn_weights = torch.softmax(attn_scores, dim=-1)          # [BL, num_windows]
        pooled = (local_encoded * attn_weights.unsqueeze(-1)).sum(dim=1)  # [BL, D]

        # reshape 回 (B, L, D)
        pooled = pooled.view(B, L, D)
        return pooled

    # ====================== 全局池化 ======================
    def attention_based_global_pool(self, local_features, attention_layer):
        """
        基于注意力的全局池化。
        输入:
            local_features: (B, L, D)
            attention_layer: Linear(D, 1)
        输出:
            (B, D)
        """
        attention_scores = attention_layer(local_features).squeeze(-1)  # [B, L]
        attention_weights = torch.softmax(attention_scores, dim=-1)     # [B, L]
        weighted_features = local_features * attention_weights.unsqueeze(-1)  # [B, L, D]
        return weighted_features.sum(dim=1)  # [B, D]

    # ====================== 前向传播 ======================
    def forward(self, batched_data):
        # 获取基本信息
        node_feat = batched_data["node_feat"]
        n_graph, n_node = node_feat.size()[:2]

        weight_dtype = self.node_encoder.weight.dtype
        device = next(self.parameters()).device

        # 统一所有 tensor 到当前设备
        for key, val in batched_data.items():
            if isinstance(val, torch.Tensor):
                batched_data[key] = val.to(device)

        # 1. 处理基础节点特征
        node_features = self.node_encoder(node_feat.to(weight_dtype))  # [B, N, hidden_dim]

        # 2. 创建图 token 特征
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)  # [B, 1, hidden_dim]

        # 3. 处理全局特征 (MaSIF + GBScore)
        global_features = []

        # ---------------- MaSIF 分支（方案二） ----------------
        if (
            self.use_masif
            and "protein_masif_feature" in batched_data
            and "ligand_masif_feature" in batched_data
        ):
            # 期望 shape（collator 已经整理成 4D）:
            #   protein_masif_feature: (B, L_pro, S_pro_raw, 5)
            #   ligand_masif_feature:  (B, L_lig, S_lig_raw, 5)
            protein_feat = batched_data["protein_masif_feature"]
            ligand_feat = batched_data["ligand_masif_feature"]

            # 1) 特征维度统一: 5 -> 64
            protein_unified = self.feature_dim_unifier(protein_feat.to(weight_dtype))  # [B, Lp, Sp_raw, 64]
            ligand_unified  = self.feature_dim_unifier(ligand_feat.to(weight_dtype))   # [B, Ll, Sl_raw, 64]

            # 1.5) 在 S 维上 pad / truncate 到固定 S_target = 60
            protein_unified = self._pad_spatial_to_target(protein_unified)  # [B, Lp, 60, 64]
            ligand_unified  = self._pad_spatial_to_target(ligand_unified)   # [B, Ll, 60, 64]

            # 2) 分层池化 + 窗口编码 + 窗口注意力：输出 (B, L, hidden_dim//2)
            protein_encoded = self.hierarchical_pool_masif_vectorized(
                protein_unified,
                encoder=self.protein_local_encoder,
                attn_layer=self.protein_attention,
                masif_mask=None,
            )  # [B, Lp, D]

            ligand_encoded = self.hierarchical_pool_masif_vectorized(
                ligand_unified,
                encoder=self.ligand_local_encoder,
                attn_layer=self.ligand_attention,
                masif_mask=None,
            )  # [B, Ll, D]

            # 3) 单向交叉注意力：ligand query, protein key/value
            #    输入/输出: (B, L, D)
            ligand_cross, _ = self.cross_attention(
                query=ligand_encoded,
                key=protein_encoded,
                value=protein_encoded,
            )

            # 4) 残差连接
            ligand_enhanced = ligand_encoded + ligand_cross  # [B, Ll, D]

            # 5) 分别在 L 维度上做注意力全局池化 -> 图级向量 [B, D]
            protein_global = self.attention_based_global_pool(protein_encoded, self.protein_attention)
            ligand_global = self.attention_based_global_pool(ligand_enhanced, self.ligand_attention)

            # 6) 蛋白 & 配体 融合成 masif_emb: [B, hidden_dim]
            masif_emb = torch.cat([protein_global, ligand_global], dim=-1)  # [B, 2D]  D=hidden_dim//2
            masif_emb = self.global_masif_encoder(masif_emb)                # [B, hidden_dim]

            global_features.append(masif_emb)

        # ---------------- GBScore 分支 ----------------
        if self.use_gbscore and "gbscore" in batched_data:
            gbscore_feat = batched_data["gbscore"]  # [B, 400]
            gbscore_emb = self.gbscore_encoder(gbscore_feat.to(weight_dtype))  # [B, hidden_dim]
            global_features.append(gbscore_emb)

        # 4. 融合全局特征到图 token 中
        if global_features:
            graph_token_flat = graph_token_feature.squeeze(1)  # [B, hidden_dim]
            fusion_input = torch.cat([graph_token_flat] + global_features, dim=1)  # [B, hidden_dim + ...]
            fused_graph_token = self.feature_fusion(fusion_input)  # [B, hidden_dim]
            graph_token_feature = fused_graph_token.unsqueeze(1)   # [B, 1, hidden_dim]

        # 5. 拼接图 token 和节点特征
        graph_node_feature = torch.cat([graph_token_feature, node_features], dim=1)  # [B, 1+N, hidden_dim]

        return graph_node_feature


# ============================================================
#                AffinCraftAttnBias（未大改）
# ============================================================
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
            nn.Linear(num_heads, num_heads),
        )

        # 4. 边位置类型编码器(区分边的位置)
        self.edge_location_encoder = nn.Embedding(4, num_heads)

        # 5. 图token虚拟距离
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # 6. angle和distance特征编码器
        self.angle_encoder = nn.Sequential(
            nn.Linear(28, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, num_heads),
        )

        self.multi_dist_encoder = nn.Sequential(
            nn.Linear(28, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, num_heads),
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
            attn_bias = torch.zeros(
                [n_graph, n_node + 1, n_node + 1],
                dtype=weight_dtype,
                device=edge_feat.device,
            )
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

    def _classify_and_embed_edges_vectorized(self, edge_index, edge_types, distances,
                                             num_ligand_atoms, edge_mask=None):
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
        src_is_ligand = (src_idx > 0) & (src_idx < num_lig_expanded)
        tgt_is_ligand = (tgt_idx > 0) & (tgt_idx < num_lig_expanded)

        # 批量距离编码 - 确保输出类型正确
        distance_emb = self.distance_encoder(distances.to(weight_dtype))  # [n_graph, max_edge_num, num_heads]

        # 批量处理结构边
        is_structural = (edge_types[:, :, 0] <= 1)
        structural_idx = torch.clamp(
            edge_types[:, :, 0] * 4 + edge_types[:, :, 1] * 2 + edge_types[:, :, 2],
            min=0, max=19,
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
        plip_emb = torch.zeros(
            n_graph, max_edge_num, self.num_heads,
            device=edge_types.device, dtype=weight_dtype,
        )

        # 使用masked indexing批量处理
        if both_ligand.any():
            valid_indices = both_ligand & (plip_type_idx >= 0) & (plip_type_idx < 15)
            if valid_indices.any():
                plip_emb[valid_indices] = self.plip_intra_ligand_encoder(
                    plip_type_idx[valid_indices]
                ).to(weight_dtype)

        if both_protein.any():
            valid_indices = both_protein & (plip_type_idx >= 0) & (plip_type_idx < 15)
            if valid_indices.any():
                plip_emb[valid_indices] = self.plip_intra_protein_encoder(
                    plip_type_idx[valid_indices]
                ).to(weight_dtype)

        if inter_molecular.any():
            valid_indices = inter_molecular & (plip_type_idx >= 0) & (plip_type_idx < 15)
            if valid_indices.any():
                plip_emb[valid_indices] = self.plip_inter_molecular_encoder(
                    plip_type_idx[valid_indices]
                ).to(weight_dtype)

        # 组合所有embedding - 确保所有张量类型一致
        structural_emb = structural_emb.to(weight_dtype)
        edge_embeddings = torch.where(
            is_structural.unsqueeze(-1),
            structural_emb,
            torch.where(is_plip.unsqueeze(-1), plip_emb, torch.zeros_like(distance_emb)),
        ) + distance_emb

        # 应用edge_mask
        if edge_mask is not None:
            edge_embeddings = edge_embeddings * edge_mask.unsqueeze(-1).float()

        return edge_embeddings

    # ====================================================================
    # ==================== START OF OPTIMIZED CODE =======================
    # ====================================================================
    def _update_attn_bias_vectorized(self, graph_attn_bias, edge_index,
                                     edge_embeddings, edge_mask, n_node):
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
        b_idx = torch.arange(n_graph, device=device).unsqueeze(1).expand(-1, max_edge_num)

        # 过滤出有效的索引和embedding值
        valid_b = b_idx[valid_mask]
        valid_src = src_idx[valid_mask]
        valid_tgt = tgt_idx[valid_mask]

        for h in range(num_heads):
            head_emb = edge_embeddings[:, :, h]  # [n_graph, max_edge_num]
            valid_emb = head_emb[valid_mask]

            # 正向边: src -> tgt
            graph_attn_bias.index_put_(
                (valid_b, torch.full_like(valid_b, h), valid_src, valid_tgt),
                valid_emb,
                accumulate=True,
            )
            # 反向边: tgt -> src
            graph_attn_bias.index_put_(
                (valid_b, torch.full_like(valid_b, h), valid_tgt, valid_src),
                valid_emb,
                accumulate=True,
            )
    # ====================================================================
    # ===================== END OF OPTIMIZED CODE ========================
    # ====================================================================