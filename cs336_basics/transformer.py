from math import sqrt

import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int


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
        self.weight: Float[torch.Tensor, "d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((out_features, in_features), device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out "]:
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

        self.weight: Float[torch.Tensor, "vocab_size d_model"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight: Float[torch.Tensor, " d_model"] = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        rms_norm = self.weight * x / rms

        return rms_norm.to(in_dtype)


def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x.sigmoid() * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        gate = silu(self.w1(x))
        data = self.w3(x)
        out = self.w2(gate * data)

        return out


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
        position = torch.arange(max_seq_len, device=device)
        dim = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        # Calculate inverse frequencies and compute angles using einops
        inv_freq = 1.0 / (theta ** (dim / d_k))
        angle = einsum(position, inv_freq, "seq_len, d_k -> seq_len d_k")

        # Register as non-persistent buffers (they don't need to be saved in state_dict)
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)

    def forward(
        self, x: Float[torch.Tensor, "... seq_len d_model"], token_positions: Int[torch.Tensor, " seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        # Get the rotation matrices for the given positions
        cos = self.get_buffer("cos")[token_positions]
        sin = self.get_buffer("sin")[token_positions]

        # Reshape input to separate pairs of dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation to each pair
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # Reshape back to the original dimension
        return rearrange(x_rotated, "... d_k pair -> ... (d_k pair)")


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    # For numerical stability, subtract the largest entry from all elements
    x_stable = x - x.max(dim=dim, keepdim=True).values
    x_exp = x_stable.exp()

    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... queries d_k"],
    k: Float[torch.Tensor, "... kvs d_k"],
    v: Float[torch.Tensor, "... kvs d_v"],
    m: Bool[torch.Tensor, "... queries kvs"] | None = None,
) -> Float[torch.Tensor, "... seq_len d_k"]:
    *_, d_k = q.shape

    scaled_logits = einsum(q, k, "... queries d_k, ... kvs d_k -> ... queries kvs") / sqrt(d_k)

    # The mask determines which keys the query should attend to
    if m is not None:
        scaled_logits = scaled_logits.masked_fill(~m, -torch.inf)

    # One way to think about the attention operation is that it is a differentiable lookup table.
    # We are figuring out which positions are relevant for a given query.
    probs = softmax(scaled_logits, dim=-1)

    # Produce a weighted combination of the value vectors, which can be thought of as the context
    # that should be added to the initial input, after emphasizing relevant positions.
    weighted_context = einsum(probs, v, "... queries kvs, ... kvs d_v -> ... queries d_v")

    return weighted_context


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()

        # The number of heads, or the number of separate learned subspaces for each token projection
        # into the space of queries, keys, and values.
        # This allows the model to attend and provide context according to different relational views
        # of the tokens.
        self.h: int = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # First, we project our input into the space of queries, keys, and vectors
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Next, we essentially split up the model dimension into the h subspaces, and treat
        # them as a batch dimension
        q = rearrange(q, "... queries (h d_k) -> ... h queries d_k", h=self.h)
        k = rearrange(k, "... kvs (h d_k) -> ... h kvs d_k", h=self.h)
        v = rearrange(v, "... kvs (h d_k) -> ... h kvs d_k", h=self.h)

        # Create a causal mask to prevent the model from attending to future tokens in the sequence
        seq_len = x.shape[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # Include the positional encoding to the query and key vectors
        if self.rope:
            positions = torch.arange(seq_len)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # Calculate the attention operation, and then rearrange
        weighted_context = scaled_dot_product_attention(q, k, v, mask)
        weighted_context = rearrange(weighted_context, "... h queries d_k -> ... queries (h d_k)")

        # Think of this as a remixing operation: after operating on the h subspaces, we remix
        # them so that information from each separate head can influence any dimension of the model.
        weighted_context = self.output_proj(weighted_context)

        return weighted_context


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        # An intuition to have a pre-norm block (in place of a post-norm) is that
        # we keep a clean residual stream. The information from the input is preserved,
        # while components (e.g. attention heads, mlps, layer norms) add onto this stream.
        #
        # See https://transformer-circuits.pub/2021/framework/index.html#residual-comms for more intuition.

        # First, we augment the residual stream with the information from the attention operation
        h = x + self.attn(self.ln1(x))

        # Then, augment the residual stream with a non-linear rewrite of the information to a potentially
        # richer representation
        h = h + self.ffn(self.ln2(h))

        return h


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

    def forward(self, x: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        # First, transform the token ids into the embedding space
        tokens = self.token_embeddings(x)

        # Next, feed the tokens through the transformer blocks
        h = self.layers(tokens)

        # Then, pass the tokens through the final layers to obtain a distribution over the vocabulary
        h = self.ln_final(h)
        probs = self.lm_head(h)

        return probs
