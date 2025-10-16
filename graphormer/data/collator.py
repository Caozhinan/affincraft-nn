# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch_geometric as pyg
import numpy as np

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 <= padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def gen_node_type_edge(items, max_node_num):
    num_ligand_nodes = [int(item.num_node[0]) for item in items]
    num_protein_nodes = [int(item.num_node[1]) for item in items]
    t = 10
    node_type_edges = []
    for idx in range(len(items)):
        n_nodes, n_ligand_nodes = num_ligand_nodes[idx] + num_protein_nodes[idx], num_ligand_nodes[idx]
        node_type_edge = torch.zeros(n_nodes, n_nodes, dtype=torch.long)
        node_type_edge[:n_ligand_nodes, :n_ligand_nodes] = 3
        node_type_edge[n_ligand_nodes:, :n_ligand_nodes] = 4
        node_type_edge[:n_ligand_nodes, n_ligand_nodes:] = 5
        node_type_edge[n_ligand_nodes:, n_ligand_nodes:] = 6

        atom_type = items[idx].x[:, :t] - 1 - torch.arange(0, items[idx].x[:, :t].shape[-1] * 512, 512, dtype=torch.long)
        atom_i = atom_type.unsqueeze(1).repeat(1, n_nodes, 1)
        atom_j = atom_type.unsqueeze(0).repeat(n_nodes, 1, 1)
        atom_type_pair = torch.cat([atom_i, atom_j], dim=-1)
        # N x N x i
        node_atom_edge = torch.cat([node_type_edge.unsqueeze(2),
                                    atom_type_pair,
                                    torch.clamp(torch.div(items[idx].angle.sum(dim=2).to(torch.long), 10, rounding_mode='floor'), 0, 1024).unsqueeze(2),
                                    # angle must at last, since 1024 dim
                                    ], dim=-1)
        # N x N x K, max_at=512*4
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_atom_edge = pad_edge_type_unsqueeze(node_atom_edge, max_node_num)
        node_type_edges.append(node_atom_edge.long())

    node_type_edge = torch.cat(node_type_edges)
    return node_type_edge


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    original_len = len(items)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    filtered_len = len(items)
    assert filtered_len == original_len, f"filtered_len = {filtered_len}, original_len = {original_len}"

    max_node_num = max(i.x.size(0) for i in items)
    num_node = torch.stack([item.num_node for item in items])
    node_type_edge = gen_node_type_edge(items, max_node_num)

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.pdbid,
            item.frame if hasattr(item, "frame") else 0,
            item.rmsd_lig if hasattr(item, "rmsd_lig") else 0.0,
            item.rmsd_pro if hasattr(item, "rmsd_pro") else 0.0,
            # item.angle,
            # item.dists,
            torch.cat([item.rfscore, item.gbscore, item.ecif]).float() / 100  # 100 + 400 + 1540 = 2040
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        pdbid,
        frame,
        rmsd_lig,
        rmsd_pro,
        # angles,
        # distss,
        fps
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")


    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, multi_hop_max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    frame = torch.LongTensor(frame)
    rmsd_lig = torch.tensor(rmsd_lig, dtype=torch.float)
    rmsd_pro = torch.tensor(rmsd_pro, dtype=torch.float)

    # angle = torch.cat(
    #     [pad_edge_type_unsqueeze(i, max_node_num) for i in angles]
    # )
    # dists = torch.cat(
    #     [pad_edge_type_unsqueeze(i, max_node_num) for i in distss]
    # )
    fps = torch.stack(fps).float()

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
        pos=pos,
        pdbid=pdbid,
        frame=frame,
        num_node=num_node,
        node_type_edge=node_type_edge,
        rmsd_lig=rmsd_lig,
        rmsd_pro=rmsd_pro,
        # angle=angle,
        # dists=dists,
        fp=fps
    )