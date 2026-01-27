# --------------------------------------------------------------------------------------
# DISCLAIMER:
# This file is adapted from the TransolverPlusPlus GitHub repository:
# https://github.com/thuml/Transolver_plus/tree/main
# Retrieved on September 17, 2025.
# --------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.distributed.nn as dist_nn
import torch.nn as nn
import torch.nn.functional as F
#from einops import rearrange
from torch.utils.checkpoint import checkpoint

from graphphysics.models.layers import TemporalAttention, build_mlp


def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau

    y = F.softmax(y, dim=-1)

    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        use_rope_embeddings: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU(),
        )
        self.use_rope_embeddings = use_rope_embeddings
        self.rope_pos_dimension = rope_pos_dimension
        self.rope_base = rope_base
        self.use_gated_attention = use_gated_attention
        if self.use_rope_embeddings:
            if rope_pos_dimension <= 0:
                raise ValueError(
                    "rope_pos_dimension must be positive when use_rope_embeddings=True."
                )
            self.rope_projection = nn.Linear(rope_pos_dimension * 2, dim_head)
        else:
            self.rope_projection = None

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for ps in [self.in_project_slice]:
            torch.nn.init.orthogonal_(ps.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        if self.use_gated_attention:
            self.attn_gate = nn.Sequential(
                nn.Linear(2 * dim_head, dim_head),
                nn.SiLU(),
                nn.Linear(dim_head, dim_head),
                nn.Sigmoid(),
            )
        else:
            self.attn_gate = None

    def _rope_features(self, pos: torch.Tensor) -> torch.Tensor:
        if self.rope_pos_dimension <= 0:
            return torch.zeros(
                pos.shape[0],
                pos.shape[1],
                0,
                device=pos.device,
                dtype=pos.dtype,
            )
        if pos.size(-1) < self.rope_pos_dimension:
            raise ValueError(
                f"Expected at least {self.rope_pos_dimension} positional dimensions, "
                f"but received {pos.size(-1)}."
            )
        pos_slice = pos[..., : self.rope_pos_dimension]
        log_base = torch.log(
            torch.tensor(self.rope_base, device=pos.device, dtype=pos.dtype)
        )
        inv_freq = torch.exp(
            -torch.arange(self.rope_pos_dimension, device=pos.device, dtype=pos.dtype)
            * log_base
            / max(self.rope_pos_dimension, 1)
        )
        angles = pos_slice * inv_freq.view(1, 1, self.rope_pos_dimension)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(self, x, pos=None):
        B, N, C = x.shape

        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        if self.use_rope_embeddings and pos is not None:
            rope_feat = self._rope_features(pos).to(x_mid.dtype)
            rope_proj = self.rope_projection(rope_feat)
            x_mid = x_mid + rope_proj.unsqueeze(1)

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            dist_nn.all_reduce(slice_norm, op=dist_nn.ReduceOp.SUM)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            dist_nn.all_reduce(slice_token, op=dist_nn.ReduceOp.SUM)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token
        )
        if self.attn_gate is not None:
            gate_input = torch.cat([slice_token, out_slice_token], dim=-1)
            gate = self.attn_gate(gate_input)
            out_slice_token = gate * out_slice_token

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Transolver_plus_block(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_rope_embeddings: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_gated_attention: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_1D_Eidetic(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_rope_embeddings=use_rope_embeddings,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_gated_attention=use_gated_attention,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = build_mlp(
            in_size=hidden_dim,
            hidden_size=hidden_dim * mlp_ratio,
            out_size=hidden_dim,
            nb_of_layers=2,
            layer_norm=False,
            act=act,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, pos=None):
        if self.training:
            fx = (
                checkpoint(
                    lambda z: self.Attn(z, pos), self.ln_1(fx), use_reentrant=True
                )
                + fx
            )
        else:
            fx += self.Attn(self.ln_1(fx), pos)
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=False,
        use_rope_embeddings: bool = False,
        use_gated_attention: bool = False,
        rope_pos_dimension: int = 3,
        rope_base: float = 10000.0,
        use_temporal_block: bool = False,
    ):
        super(Model, self).__init__()
        self.__name__ = "UniPDE_3D"
        self.ref = ref
        self.unified_pos = unified_pos
        self.use_rope_embeddings = use_rope_embeddings
        self.use_gated_attention = use_gated_attention
        self.rope_pos_dimension = rope_pos_dimension
        self.rope_base = rope_base
        self.use_temporal_block = use_temporal_block
        if self.unified_pos:
            self.preprocess = build_mlp(
                in_size=fun_dim + self.ref * self.ref * self.ref,
                hidden_size=n_hidden * 2,
                out_size=n_hidden,
                nb_of_layers=2,
                layer_norm=False,
                act=act,
            )
        else:
            self.preprocess = build_mlp(
                in_size=fun_dim + space_dim,
                hidden_size=n_hidden * 2,
                out_size=n_hidden,
                nb_of_layers=2,
                layer_norm=False,
                act=act,
            )

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(3, n_hidden)
        block_kwargs = dict(
            num_heads=n_head,
            hidden_dim=n_hidden,
            dropout=dropout,
            act=act,
            mlp_ratio=mlp_ratio,
            out_dim=out_dim,
            slice_num=slice_num,
            use_rope_embeddings=use_rope_embeddings,
            rope_pos_dimension=rope_pos_dimension,
            rope_base=rope_base,
            use_gated_attention=use_gated_attention,
        )
        if self.use_temporal_block:
            self.blocks = nn.ModuleList(
                [
                    Transolver_plus_block(
                        **block_kwargs,
                        last_layer=False,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.output_proj = nn.Linear(n_hidden, out_dim)
        else:
            self.blocks = nn.ModuleList(
                [
                    Transolver_plus_block(
                        **block_kwargs,
                        last_layer=(_ == n_layers - 1),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.output_proj = None
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float)
        )
        self.temporal_block = (
            TemporalAttention(
                hidden_size=n_hidden, num_heads=n_head, use_gate=use_gated_attention
            )
            if self.use_temporal_block
            else None
        )

    def get_grid(self, my_pos):
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat(
            [batchsize, 1, self.ref, self.ref, 1]
        )
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat(
            [batchsize, self.ref, 1, self.ref, 1]
        )
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat(
            [batchsize, self.ref, self.ref, 1, 1]
        )
        grid_ref = (
            torch.cat((gridx, gridy, gridz), dim=-1)
            .cuda()
            .reshape(batchsize, self.ref**3, 3)
        )

        pos = (
            torch.sqrt(
                torch.sum(
                    (my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1
                )
            )
            .reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref)
            .contiguous()
        )
        return pos

    def forward(self, x, pos, condition):
        if self.use_rope_embeddings and pos is None:
            raise ValueError(
                "use_rope_embeddings=True requires node positions for Transolver."
            )
        fx, _ = None, None
        if pos is not None and self.unified_pos:
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if condition is not None:
            condition = self.embedding(condition).unsqueeze(1)
            fx = fx + condition

        prev_fx = fx
        for block in self.blocks:
            prev_fx = fx
            fx = block(fx, pos=pos)
        if self.temporal_block is not None:
            fused = []
            for b in range(fx.size(0)):
                fused.append(self.temporal_block(prev_fx[b], fx[b], None))
            fx = torch.stack(fused, dim=0)
            fx = self.output_proj(fx)
        elif self.output_proj is not None:
            fx = self.output_proj(fx)
        return fx
