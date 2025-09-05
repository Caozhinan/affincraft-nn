# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import fairseq.modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import init_graphormer_params, AffinCraftGraphEncoder
from ..modules.affincraft_graph_encoder import AffinCraftGraphEncoder

logger = logging.getLogger(__name__)


@register_model("graphormer")
class GraphormerModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--layerdrop",
            type=float,
            metavar="D",
            default=0.0,
            help="layer wise drop",
        )
        # encoder args
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--apply-graphormer-init",
            action="store_true",
            help="use custom param initialization for Graphormer",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--embed-scale",
            type=float,
            default=-1.0,
            help="Embedding scale apply to node embedding"
        )
        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(  # you shall not pass
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input and output embeddings",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )

        parser.add_argument(
            "--sandwich-ln",
            default=False,
            action="store_true",
            help="use sandwich layernorm for the encoder block",
        )
        # 3d_encoder

        parser.add_argument(
            "--dist-head",
            type=str,
            choices=['none', 'gbf', 'gbf3d', 'bucket', 'embed3d'],
            default='none',
            help="3d encoding head"
        )

        parser.add_argument(
            "--num-dist-head-kernel",
            type=int,
            default=128,
            help="Number of kernels in distance head"
        )
        parser.add_argument(
            "--num-edge-types",
            type=int,
            default=512*16,
            help="number of atom type for gbf dist head"
        )

        parser.add_argument(
            "--sample-weight-estimator",
            default=False,
            action="store_true",
            help="add soft weight for loss"
        )
        parser.add_argument(
            "--sample-weight-estimator-pat",
            default="pdbbind",
            type=str,
            help="pattern to assign 1.0 weight"
        )

        parser.add_argument(
            "--fingerprint",
            default=False,
            action='store_true',
            help="pattern to assign 1.0 weight"
        )


    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphormerEncoder(args)
        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes

        self.graph_encoder = AffinCraftGraphEncoder(  
            num_encoder_layers=args.encoder_layers,  
            embedding_dim=args.encoder_embed_dim,  
            ffn_embedding_dim=args.encoder_ffn_embed_dim,  
            num_attention_heads=args.encoder_attention_heads,  
            dropout=args.dropout,  
            attention_dropout=args.attention_dropout,  
            activation_dropout=args.act_dropout,  
            layerdrop=args.layerdrop,  
            encoder_normalize_before=args.encoder_normalize_before,  
            apply_graphormer_init=args.apply_graphormer_init,  
            activation_fn=args.activation_fn,  
            embed_scale=args.embed_scale if args.embed_scale > 0 else None,  
            sandwich_ln=args.sandwich_ln,  
            # AffinCraft特定参数  
            node_feat_dim=9,  
            use_masif=True,  
            use_gbscore=True,  
        ) 

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
            else:
                raise NotImplementedError
        self.sample_weight_estimator = args.sample_weight_estimator
        self.sample_weight_estimator_pat = args.sample_weight_estimator_pat
        if args.fingerprint:
            self.fpnn = nn.Sequential(nn.Linear(2040, args.encoder_embed_dim),
                                      nn.GELU(),
                                      nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim))
            self.reducer = nn.Linear(args.encoder_embed_dim*2, args.encoder_embed_dim)
        else:
            self.fpnn = None

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )
        # inner_stats: N x B x H
        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        # x: B x N x H
        if self.fpnn is not None:
            fpemb = self.fpnn(batched_data["fp"])
            # fpemb: B x H
            x = self.reducer(torch.cat([x[:, 0, :].squeeze(dim=1), fpemb], dim=1))
        else:
            x = x[:, 0, :].squeeze(dim=1)
        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        if self.sample_weight_estimator:
            weight = torch.ones(x.shape, dtype=x.dtype, device=x.device) * 0.01
            wmask = torch.ones(weight.shape, dtype=torch.bool, device=weight.device)
            wones = torch.ones(weight.shape, dtype=weight.dtype, device=weight.device)
            for idx, i in enumerate(batched_data['pdbid']):
                if i.endswith(self.sample_weight_estimator_pat):
                    wmask[idx] = False
            weight = torch.where(wmask, weight, wones)
            return x, weight
        return x

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict


@register_model_architecture("graphormer", "graphormer")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)


@register_model_architecture("graphormer", "graphormer_base")
def graphormer_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    base_architecture(args)


@register_model_architecture("graphormer", "graphormer_slim")
def graphormer_slim_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 80)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 80)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    base_architecture(args)


@register_model_architecture("graphormer", "graphormer_large")
def graphormer_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)

    args.encoder_layers = getattr(args, "encoder_layers", 24)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    base_architecture(args)