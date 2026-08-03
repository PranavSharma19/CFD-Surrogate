"""Memory-conscious GATv2 baseline adapted from Steve v4."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn import GATv2Conv


class FlowEncoder(nn.Module):
    def __init__(self, input_dim: int = 7, context_dim: int = 48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, context_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GATBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.convolution = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=hidden_dim,
            dropout=dropout,
            add_self_loops=True,
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        update = self.convolution(x, edge_index, edge_attr=edge_attr)
        return x + self.dropout(functional.silu(self.normalization(update)))


class AorticGATBaseline(nn.Module):
    """Non-equivariant baseline; the equivariant model must outperform it."""

    def __init__(
        self,
        node_dim: int = 10,
        edge_dim: int = 4,
        conditioning_dim: int = 7,
        hidden_dim: int = 96,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.flow_encoder = FlowEncoder(conditioning_dim, context_dim=48)
        self.blocks = nn.ModuleList(GATBlock(hidden_dim, heads, dropout) for _ in range(layers))
        self.film_gamma = nn.Linear(48, hidden_dim)
        self.film_beta = nn.Linear(48, hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)
        for block in self.blocks:
            x = block(x, data.edge_index, edge_attr)
        context = self.flow_encoder(data.conditioning.view(1, -1))
        x = self.film_gamma(context) * x + self.film_beta(context)
        return self.output(x)

