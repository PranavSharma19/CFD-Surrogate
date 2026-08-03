import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from aorta_surrogate.models.equivariant_surface import EquivariantSurfaceGNN


def test_output_rotates_with_surface():
    torch.manual_seed(7)
    position = torch.randn(8, 3)
    normal = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    tangent = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7]],
        dtype=torch.long,
    )
    scalar = torch.randn(8, 12)
    conditioning = torch.randn(7)
    model = EquivariantSurfaceGNN(hidden_dim=32, layers=2).eval()
    original = Data(pos=position, normal=normal, tangent=tangent, edge_index=edges, scalar_x=scalar, conditioning=conditioning)
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle), torch.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rotated = Data(
        pos=position @ rotation.T,
        normal=normal @ rotation.T,
        tangent=tangent @ rotation.T,
        edge_index=edges,
        scalar_x=scalar,
        conditioning=conditioning,
    )
    with torch.no_grad():
        expected = model(original) @ rotation.T
        actual = model(rotated)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_multiscale_output_rotates_with_surface():
    torch.manual_seed(9)
    position = torch.randn(6, 3)
    normal = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    tangent = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    fine = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]])
    coarse = torch.tensor([[0, 2, 1, 3, 2, 4, 3, 5], [2, 0, 3, 1, 4, 2, 5, 3]])
    scalar = torch.randn(6, 14)
    conditioning = torch.randn(7)
    angle = torch.tensor(-0.41)
    rotation = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0],
         [-torch.sin(angle), 0.0, torch.cos(angle)]]
    )
    model = EquivariantSurfaceGNN(scalar_dim=14, hidden_dim=24, layers=2, use_multiscale=True).eval()
    common = dict(edge_index=fine, coarse_edge_index=coarse, scalar_x_multiscale=scalar, conditioning=conditioning)
    original = Data(pos=position, normal=normal, tangent=tangent, **common)
    rotated = Data(pos=position @ rotation.T, normal=normal @ rotation.T, tangent=tangent @ rotation.T, **common)
    with torch.no_grad():
        expected = model(original) @ rotation.T
        actual = model(rotated)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
