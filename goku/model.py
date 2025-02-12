from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis.unbind(-1)
    cos = cos[None, None]
    sin = sin[None, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, inner_dim=None, bias=True):
        super().__init__()
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = dim_out if dim_out is not None else dim
        self.fc1 = nn.Linear(dim, inner_dim, bias=bias)
        self.fc2 = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states =  F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * output).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, heads=8, dim_head=64, dropout=0.0, bias=False):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim if kv_dim is not None else q_dim
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.head_dim = dim_head
        self.num_heads = heads

        assert (dim_head * heads) == self.inner_dim, f"{dim_head=}, {heads=}, {self.inner_dim=}"

        self.q_proj = nn.Linear(self.q_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)

        self.o_proj = nn.Linear(self.inner_dim, self.q_dim, bias=bias)

        self.q_norm = RMSNorm(self.inner_dim)
        self.k_norm = RMSNorm(self.inner_dim)

    def prepare_attention_mask(
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L694
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ):
        head_size = self.num_heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def forward(
        self,
        inputs_q,
        inputs_kv,
        attention_mask=None,
        cross_attention=False,
        rope_pos_embed=None,

    ):

        inputs_kv = inputs_q if inputs_kv is None else inputs_kv

        query_states = self.q_proj(inputs_q)
        key_states = self.k_proj(inputs_kv)
        value_states = self.v_proj(inputs_kv)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.training:
            # from flash_attn import flash_attn_varlen_func
            # todo: image-video joint training, packing sequence and training with variable length `flash_attn_varlen_func`
            pass

        batch_size, q_len = inputs_q.shape[:2]
        kv_len = inputs_kv.shape[1]

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_pos_embed is not None:
            query_states = apply_rotary_emb(query_states, rope_pos_embed)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states, rope_pos_embed)

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, kv_len, batch_size)
            attention_mask = attention_mask.view(batch_size, self.num_heads, -1, attention_mask.shape[-1])

        # todo: flash attention implementation
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):  # for reproducibility
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, self.inner_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Block(nn.Module):
    # todo: transformer block
    pass
