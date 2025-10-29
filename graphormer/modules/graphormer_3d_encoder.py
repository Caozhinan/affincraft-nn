from typing import Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional
from .graphormer_layers import init_params
from math import pi

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


def init_params(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None, bias=True):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden, bias=bias)
        self.layer2 = nn.Linear(hidden, output_size, bias=bias)

    def forward(self, x):
        x = functional.gelu(self.layer1(x))
        x = self.layer2(x)
        return x


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.uniform_(self.mul.weight, -1, 1)
        with torch.no_grad():
            self.mul.weight[self.mul.padding_idx].fill_(0)
            self.bias.weight[self.bias.padding_idx].fill_(0)

    def forward(self, x, edge_types):
        # x: B x N x N; edge_types: B x N x N x Z
        mul = self.mul(edge_types).mean(dim=-2)
        bias = self.bias(edge_types).mean(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        xx = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(xx.float(), mean, std).type_as(self.means.weight)


class GBF2DEncoder(nn.Module):
    def __init__(self,
                 num_dist_head_kernel: int = 128,
                 num_edge_types: int = 512*16,
                 num_heads: int = 32,
                 embedding_dim: int = 768, **kwargs):
        super().__init__()
        self.num_dist_head_kernel = num_dist_head_kernel
        self.edge_types = num_edge_types
        self.num_heads = num_heads
        self.embed_dim = embedding_dim
        self.dist_encoder = GaussianLayer(num_dist_head_kernel, self.edge_types)
        self.bias_proj: Callable[[torch.Tensor], torch.Tensor] = NonLinear(
            num_dist_head_kernel, num_heads
        )
        self.edge_proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(num_dist_head_kernel, embedding_dim)

    def forward(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, pos, node_type_edge = batched_data["x"], batched_data["pos"], batched_data['node_type_edge']
        # x: B x N x 9
        atoms = x[:, :, 0]
        padding_mask = atoms.eq(0)
        # padding_mask: B x N
        p1 = padding_mask.unsqueeze(1).repeat(1, atoms.shape[1], 1).eq(0)  # B x N x 1
        p2 = padding_mask.unsqueeze(2).repeat(1, 1, atoms.shape[1]).eq(0)  # B x 1 x N
        pair_mask = p1 & p2
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: torch.Tensor = delta_pos.norm(dim=-1)
        dist[dist < 0.01] = 100

        gbf_feature = self.dist_encoder(dist, node_type_edge)  # B x N x N x K
        # print("gbf feature:", gbf_feature.max(), gbf_feature.min())
        # B x N x N x K
        edge_features = gbf_feature.masked_fill(pair_mask.unsqueeze(-1), 0.0)
        to_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        to_attn_bias = to_attn_bias.masked_fill(pair_mask.unsqueeze(-1).permute(0, 3, 1, 2), 0.0)

        dist_emb = edge_features.sum(dim=-2)  # B x N x K
        # dist_emb_norm = (~padding_mask).sum(dim=1).unsqueeze(1).unsqueeze(2)  # B x 1 x 1
        dist_emb = self.edge_proj(dist_emb) / 100
        return dist_emb, to_attn_bias
        # B x N x H


class GBF3DEncoder(nn.Module):
    def __init__(self,
                 num_dist_head_kernel: int = 128,
                 num_edge_types: int = 512*16,
                 num_heads: int = 32,
                 embedding_dim: int = 768, **kwargs):
        super().__init__()
        self.num_dist_head_kernel = num_dist_head_kernel
        self.num_edge_types = num_edge_types
        self.num_heads = num_heads
        self.embed_dim = embedding_dim
        self.dist_encoder = GaussianLayer(num_dist_head_kernel, self.num_edge_types)
        self.bias_proj: Callable[[torch.Tensor], torch.Tensor] = NonLinear(
            num_dist_head_kernel, num_heads
        )
        self.edge_proj: Callable[[torch.Tensor], torch.Tensor] = nn.Linear(num_dist_head_kernel, embedding_dim)

    def forward(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, pos, node_type_edge = batched_data["x"], batched_data["pos"], batched_data['node_type_edge']
        # x: B x N x 9
        atoms = x[:, :, 0]
        padding_mask = atoms.eq(0)
        p1 = padding_mask.unsqueeze(1).repeat(1, atoms.shape[1], 1).eq(0)  # B x N x 1
        p2 = padding_mask.unsqueeze(2).repeat(1, 1, atoms.shape[1]).eq(0)  # B x 1 x N
        pair_mask = p1 & p2
        # padding_mask: B x N
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist: torch.Tensor = delta_pos.norm(dim=-1)
        dist[dist < 0.01] = 100
        # this is the fucking important but dirty trick!
        gbf_feature = self.dist_encoder(dist, node_type_edge)  # B x N x N x K
        # print("gbf feature:", gbf_feature.max(), gbf_feature.min())
        # B x N x N x K
        edge_features = gbf_feature.masked_fill(pair_mask.unsqueeze(-1), 0.0)
        to_attn_bias = self.bias_proj(gbf_feature).permute(0, 3, 1, 2).contiguous()
        to_attn_bias = to_attn_bias.masked_fill(pair_mask.unsqueeze(-1).permute(0, 3, 1, 2), 0.0)

        dist_emb = edge_features.sum(dim=-2)  # B x N x K
        # dist_emb_norm = (~padding_mask).sum(dim=1).unsqueeze(1).unsqueeze(2)  # B x 1 x 1
        dist_emb = self.edge_proj(dist_emb) / 100
        # print("dist_emb", dist_emb.abs().mean().item(), dist_emb.abs().median(), dist_emb.abs().max().item())
        return dist_emb, to_attn_bias


class Graphormer3DEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "gbf",
        **kwargs

    ) -> None:

        super().__init__()

        args = {k: v for k, v in locals().items() if k not in ['self', '__class__', 'args', 'kwargs']}
        args = {**args, **kwargs}
        if encoder_name == "gbf":
            self.encoder = GBF2DEncoder(**args)
        elif encoder_name == "gbf3d":
            self.encoder = GBF3DEncoder(**args)
        else:
            raise NotImplementedError(f"Not implemented {encoder_name}")

    def forward(self, batched_data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(batched_data)
