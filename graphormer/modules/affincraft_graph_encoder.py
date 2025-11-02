import math  
import torch  
import torch.nn as nn  
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm  
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_  
# import time
from .multihead_attention import MultiheadAttention  
from .graphormer_layers import AffinCraftNodeFeature, AffinCraftAttnBias  
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer  
from .graphormer_3d_encoder import Graphormer3DEncoder  
  
  
class AffinCraftGraphEncoder(nn.Module):  
    def __init__(  
        self,  
        num_encoder_layers: int = 12,  
        embedding_dim: int = 768,  
        ffn_embedding_dim: int = 768,  
        num_attention_heads: int = 32,  
        dropout: float = 0.1,  
        attention_dropout: float = 0.1,  
        activation_dropout: float = 0.1,  
        layerdrop: float = 0.0,  
        encoder_normalize_before: bool = False,  
        apply_graphormer_init: bool = False,  
        activation_fn: str = "gelu",  
        embed_scale: float = None,  
        freeze_embeddings: bool = False,  
        n_trans_layers_to_freeze: int = 0,  
        export: bool = False,  
        traceable: bool = False,  
        q_noise: float = 0.0,  
        qn_block_size: int = 8,  
        sandwich_ln: bool = False,  
        # AffinCraft特定参数  
        node_feat_dim: int = 9,  
        use_masif: bool = True,  
        use_gbscore: bool = True,  
        **kwargs  
    ) -> None:  
          
        super().__init__()  
        self.dropout_module = FairseqDropout(  
            dropout, module_name=self.__class__.__name__  
        )  
        self.layerdrop = layerdrop  
        self.embedding_dim = embedding_dim  
        self.apply_graphormer_init = apply_graphormer_init  
        self.traceable = traceable  
  
        # 使用新的embedding层  
        self.graph_node_feature = AffinCraftNodeFeature(  
            node_feat_dim=node_feat_dim,  
            hidden_dim=embedding_dim,  
            n_layers=num_encoder_layers,  
            use_masif=use_masif,  
            use_gbscore=use_gbscore  
        )  
  
        self.graph_attn_bias = AffinCraftAttnBias(  
            num_heads=num_attention_heads,  
            hidden_dim=embedding_dim,  
            n_layers=num_encoder_layers  
        )  
  
        self.embed_scale = embed_scale  
  
        if q_noise > 0:  
            self.quant_noise = apply_quant_noise_(  
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),  
                q_noise,  
                qn_block_size,  
            )  
        else:  
            self.quant_noise = None  
  
        if encoder_normalize_before:  
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)  
        else:  
            self.emb_layer_norm = None  
  
        # Transformer层结构  
        if self.layerdrop > 0.0:  
            self.layers = LayerDropModuleList(p=self.layerdrop)  
        else:  
            self.layers = nn.ModuleList([])  
          
        self.layers.extend([  
            self.build_graphormer_graph_encoder_layer(  
                embedding_dim=self.embedding_dim,  
                ffn_embedding_dim=ffn_embedding_dim,  
                num_attention_heads=num_attention_heads,  
                dropout=self.dropout_module.p,  
                attention_dropout=attention_dropout,  
                activation_dropout=activation_dropout,  
                activation_fn=activation_fn,  
                export=export,  
                q_noise=q_noise,  
                qn_block_size=qn_block_size,  
                sandwich_ln=sandwich_ln,  
            )  
            for i in range(num_encoder_layers)  
        ])  
  
        def freeze_module_params(m):  
            if m is not None:  
                for p in m.parameters():  
                    p.requires_grad = False  
  
        if freeze_embeddings:  
            freeze_module_params(self.graph_node_feature)  
            freeze_module_params(self.graph_attn_bias)  
  
        for layer in range(n_trans_layers_to_freeze):  
            freeze_module_params(self.layers[layer])  
  
    def build_graphormer_graph_encoder_layer(  
        self,  
        embedding_dim,  
        ffn_embedding_dim,  
        num_attention_heads,  
        dropout,  
        attention_dropout,  
        activation_dropout,  
        activation_fn,  
        export,  
        q_noise,  
        qn_block_size,  
        sandwich_ln,  
    ):  
        return GraphormerGraphEncoderLayer(  
            embedding_dim=embedding_dim,  
            ffn_embedding_dim=ffn_embedding_dim,  
            num_attention_heads=num_attention_heads,  
            dropout=dropout,  
            attention_dropout=attention_dropout,  
            activation_dropout=activation_dropout,  
            activation_fn=activation_fn,  
            export=export,  
            q_noise=q_noise,  
            qn_block_size=qn_block_size,  
            sandwich_ln=sandwich_ln,  
        )  
  
    def forward(  
        self,  
        batched_data,  
        perturb=None,  
        last_state_only: bool = False,  
        token_embeddings: torch.Tensor = None,  
        attn_mask: torch.Tensor = None,  
    ):  
        # import time  
        # forward_start = time.time()  

        is_tpu = False  

        # 计算padding mask  
        # mask_start = time.time()  
        node_feat = batched_data["node_feat"]  
        n_graph, n_node = node_feat.size()[:2]  
        padding_mask = torch.zeros(n_graph, n_node, device=node_feat.device, dtype=torch.bool)  
        padding_mask_cls = torch.zeros(  
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype  
        )  
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)  
        # print(f"[TIMING] AffinCraft padding_mask: {time.time() - mask_start:.4f}s")  
    
        # 节点特征embedding  
        if token_embeddings is not None:  
            x = token_embeddings  
        else:  
            # node_feat_start = time.time()  
            x = self.graph_node_feature(batched_data)  
            # print(f"[TIMING] AffinCraft graph_node_feature: {time.time() - node_feat_start:.4f}s")  
    
        if perturb is not None:  
            x[:, 1:, :] += perturb  
    
        # 注意力偏置  
        # attn_bias_start = time.time()  
        attn_bias = self.graph_attn_bias(batched_data)  
        # print(f"[TIMING] AffinCraft graph_attn_bias: {time.time() - attn_bias_start:.4f}s")  
    
        # 后处理  
        # process_start = time.time()  
        if self.embed_scale is not None:  
            x = x * self.embed_scale  
        if self.quant_noise is not None:  
            x = self.quant_noise(x)  
        if self.emb_layer_norm is not None:  
            x = self.emb_layer_norm(x)  
        x = self.dropout_module(x)  
        x = x.transpose(0, 1)  
        # print(f"[TIMING] AffinCraft pre-processing: {time.time() - process_start:.4f}s")  
    
        inner_states = []  
        if not last_state_only:  
            inner_states.append(x)  
    
        # Transformer层  
        # layers_start = time.time()  
        for layer_idx, layer in enumerate(self.layers):  
            # layer_start = time.time()  
            x, _ = layer(  
                x,  
                self_attn_padding_mask=padding_mask,  
                self_attn_mask=attn_mask,  
                self_attn_bias=attn_bias,  
            )  
            # print(f"[TIMING] AffinCraft layer {layer_idx}: {time.time() - layer_start:.4f}s")  
            if not last_state_only:  
                inner_states.append(x)  
        # print(f"[TIMING] AffinCraft all layers total: {time.time() - layers_start:.4f}s")  
    
        graph_rep = x[0, :, :]  
    
        if last_state_only:  
            inner_states = [x]  
    
        # print(f"[TIMING] AffinCraft forward total: {time.time() - forward_start:.4f}s")  

        if self.traceable:  
            return torch.stack(inner_states), graph_rep  
        else:  
            return inner_states, graph_rep