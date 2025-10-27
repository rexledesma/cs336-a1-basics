from math import sqrt

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        std = sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x.square(), "... d_model -> ... 1", "mean") + self.eps)
        rms_norm = einsum(x / rms, self.weight, "... d_model, d_model -> ... d_model")

        return rms_norm.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid() * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.silu = SiLU()
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w1(x))
        glu = gate * self.w3(x)
        swiglu = self.w2(glu)

        return swiglu


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # Create position and dimension indices
        position = torch.arange(max_seq_len)
        dim = torch.arange(0, d_k, 2).float()

        # Calculate inverse frequencies and compute angles using einops
        inv_freq = 1.0 / (theta ** (dim / d_k))
        angle = einsum(position, inv_freq, "seq_len, d_k -> seq_len d_k").repeat_interleave(2, -1)

        # Register as non-persistent buffers (they don't need to be saved in state_dict)
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get the rotation matrices for the given positions
        cos = self.get_buffer("cos")[token_positions]
        sin = self.get_buffer("sin")[token_positions]

        # Reshape input to separate pairs of dimensions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_perm = rearrange(torch.stack([-x2, x1], dim=-1), "... d_k pair -> ... (d_k pair)")

        # Apply rotation to each pair
        x_out = x * cos + x_perm * sin

        return x_out


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x_stable = x - x.max(dim=dim, keepdim=True).values
        x_exp = x_stable.exp()
        x_exp_sum = x_exp.sum(dim=dim, keepdim=True)

        return x_exp / x_exp_sum


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = Softmax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        d_k = Q.shape[-1]

        scaled_logits = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / sqrt(d_k)

        if M is not None:
            scaled_logits = scaled_logits.masked_fill(~M, -torch.inf)

        probs = self.softmax(scaled_logits, dim=-1)
        attention = einsum(probs, V, "... n m, ... m d_v -> ... n d_v")

        return attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()

        self.h: int = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attention = Attention()
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... seq (h d_k) -> ... h seq d_k", h=self.h)
        K = rearrange(self.k_proj(x), "... seq (h d_k) -> ... h seq d_k", h=self.h)
        V = rearrange(self.v_proj(x), "... seq (h d_k) -> ... h seq d_k", h=self.h)

        seq_len = Q.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        if self.rope:
            positions = torch.arange(seq_len)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        multi_head_attention = rearrange(self.attention(Q, K, V, mask), "... h seq d_k -> ... seq (h d_k)")

        return self.output_proj(multi_head_attention)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.ln1(x))
        z = y + self.ffn(self.ln2(y))

        return z


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float,
    ):
        super().__init__()

        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length)

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*(TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)))
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embeddings(x)
        logits = self.lm_head(self.ln_final(self.layers(tokens)))

        return logits
