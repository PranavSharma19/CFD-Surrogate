"""PointNet-style per-surface-point baseline with global flow conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn

from aorta_surrogate.models.gat_baseline import FlowEncoder


class AorticPointNetBaseline(nn.Module):
    """Non-equivariant baseline that deliberately ignores mesh edges."""

    def __init__(self, node_dim: int = 10, conditioning_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.flow_encoder = FlowEncoder(conditioning_dim, context_dim=48)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 48, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data) -> torch.Tensor:
        local = self.local_encoder(data.x)
        global_geometry = torch.amax(local, dim=0, keepdim=True).expand(local.shape[0], -1)
        flow = self.flow_encoder(data.conditioning.view(1, -1)).expand(local.shape[0], -1)
        return self.decoder(torch.cat([local, global_geometry, flow], dim=-1))
