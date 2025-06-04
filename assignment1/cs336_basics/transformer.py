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
