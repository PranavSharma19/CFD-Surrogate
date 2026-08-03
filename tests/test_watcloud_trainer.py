import numpy as np
import torch

from aorta_surrogate.training.fold_trainer import _build_model
from aorta_surrogate.training.watcloud_trainer import (
    balanced_loss_terms,
    choose_region_seed,
    region_balanced_reduce,
)


def test_choose_region_seed_uses_only_supported_present_regions():
    regions = np.asarray([5, 5, 2, 2, 6])
    rng = np.random.default_rng(7)
    for _ in range(10):
        region_id, node_id = choose_region_seed(regions, rng)
        assert region_id == 2
        assert regions[node_id] == 2


def test_region_balanced_reduce_blends_global_and_macro_means():
    values = torch.tensor([1.0, 3.0, 10.0])
    regions = torch.tensor([0, 0, 1])
    expected = 0.5 * (14.0 / 3.0) + 0.5 * 6.0
    torch.testing.assert_close(region_balanced_reduce(values, regions), torch.tensor(expected))


def test_balanced_loss_terms_ignore_invalid_nodes_and_remain_finite():
    prediction = torch.tensor([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    terms = balanced_loss_terms(
        prediction,
        target,
        1.0,
        torch.tensor([0, 1, 2]),
        torch.tensor([True, False, True]),
        0.5,
    )
    assert set(terms) == {"vector", "magnitude", "direction"}
    assert all(bool(torch.isfinite(value)) for value in terms.values())
    assert float(terms["vector"]) < 1.0


def test_cloud_model_builder_honors_width_depth_and_checkpointing():
    model = _build_model(
        "equivariant", hidden_dim=32, layers=2, gradient_checkpointing=True
    )
    assert model.scalar_encoder[0].out_features == 32
    assert len(model.layers) == 2
    assert model.gradient_checkpointing is True
