from math import sqrt

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize a linear layer with the specified input and output features.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            device (torch.device | None, optional): Device to place the layer on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the layer's parameters. Defaults to None.
        """
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
        """
        Forward pass of the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize an embedding layer with the specified number of embeddings and embedding dimension.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of each embedding.
            device (torch.device | None, optional): Device to place the layer on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of the layer's parameters. Defaults to None.
        """
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
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token IDs.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
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

        rms = torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        rms_norm = (x / rms) * self.weight

        return rms_norm.to(in_dtype)
