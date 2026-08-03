"""A compact rotation/translation-equivariant surface message-passing model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _edge_invariants(data, edge_index):
    source, target = edge_index
    relative = data.pos[source] - data.pos[target]
    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1.0e-8)
    direction = relative / distance
    source_normal = data.normal[source]
    target_normal = data.normal[target]
    source_tangent = data.tangent[source]
    target_tangent = data.tangent[target]
    invariants = torch.cat(
        [
            distance,
            (source_normal * target_normal).sum(dim=-1, keepdim=True),
            (direction * source_normal).sum(dim=-1, keepdim=True),
            (direction * target_normal).sum(dim=-1, keepdim=True),
            (source_tangent * target_tangent).sum(dim=-1, keepdim=True),
            (direction * source_tangent).sum(dim=-1, keepdim=True),
            (direction * target_tangent).sum(dim=-1, keepdim=True),
            (source_normal * source_tangent).sum(dim=-1, keepdim=True),
            (target_normal * target_tangent).sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return (
        source, target, direction, source_normal, target_normal,
        source_tangent, target_tangent, invariants,
    )


class InvariantMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, invariant_dim: int = 9):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + invariant_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden, source, target, invariants):
        message = self.message(torch.cat([hidden[source], hidden[target], invariants], dim=-1))
        message = message.to(hidden.dtype)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, message)
        degree = torch.zeros(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype).unsqueeze(-1))
        aggregate = aggregate / degree.clamp_min(1.0)
        return hidden + self.update(torch.cat([hidden, aggregate], dim=-1))


class EquivariantSurfaceGNN(nn.Module):
    """Produces a tangential vector field that rotates with the input surface."""

    def __init__(
        self,
        scalar_dim: int = 12,
        conditioning_dim: int = 7,
        hidden_dim: int = 96,
        layers: int = 4,
        use_multiscale: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.gradient_checkpointing = gradient_checkpointing
        invariant_dim = 10 if use_multiscale else 9
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList(
            InvariantMessageLayer(hidden_dim, invariant_dim) for _ in range(layers)
        )
        self.vector_coefficients = nn.Sequential(
            nn.Linear(hidden_dim * 2 + invariant_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, data) -> torch.Tensor:
        scalar_values = data.scalar_x_multiscale if self.use_multiscale else data.scalar_x
        hidden = self.scalar_encoder(scalar_values)
        hidden = hidden + self.condition_encoder(data.conditioning.view(1, -1))
        edge_index = data.edge_index
        edge_scale = None
        if self.use_multiscale:
            fine_count = edge_index.shape[1]
            edge_index = torch.cat([edge_index, data.coarse_edge_index], dim=1)
            edge_scale = torch.cat(
                [
                    torch.zeros(fine_count, 1, dtype=hidden.dtype, device=hidden.device),
                    torch.ones(data.coarse_edge_index.shape[1], 1, dtype=hidden.dtype, device=hidden.device),
                ],
                dim=0,
            )
        (
            source, target, direction, source_normal, target_normal,
            source_tangent, target_tangent, invariants,
        ) = _edge_invariants(data, edge_index)
        if edge_scale is not None:
            invariants = torch.cat([invariants, edge_scale], dim=-1)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    layer,
                    hidden,
                    source,
                    target,
                    invariants,
                    use_reentrant=False,
                )
            else:
                hidden = layer(hidden, source, target, invariants)
        coefficients = self.vector_coefficients(
            torch.cat([hidden[source], hidden[target], invariants], dim=-1)
        )
        vector_message = (
            coefficients[:, :1] * direction
            + coefficients[:, 1:2] * source_normal
            + coefficients[:, 2:3] * target_normal
            + coefficients[:, 3:4] * source_tangent
            + coefficients[:, 4:5] * target_tangent
        )
        vector_message = vector_message.to(hidden.dtype)
        output = torch.zeros(data.num_nodes, 3, dtype=hidden.dtype, device=hidden.device)
        output.index_add_(0, target, vector_message)
        degree = torch.zeros(data.num_nodes, 1, dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype).unsqueeze(-1))
        output = output / degree.clamp_min(1.0)
        # WSS is tangential to a rigid wall. This projection is itself equivariant.
        return output - (output * data.normal).sum(dim=-1, keepdim=True) * data.normal
