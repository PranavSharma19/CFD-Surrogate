import numpy as np
import pytest
import zarr

from aorta_surrogate.data.volume_mesh_qc import mask_enrichment
from aorta_surrogate.training.pyg_adapter import load_target_valid_mask


def test_mask_enrichment_compares_population_and_global_prevalence():
    mask = np.asarray([True, False, False, False])
    population = np.asarray([True, True, False, False])
    assert mask_enrichment(mask, population) == 2.0


def test_mask_enrichment_handles_empty_population():
    assert mask_enrichment(np.asarray([True]), np.asarray([False])) == 0.0


def test_missing_quality_mask_defaults_to_all_valid(tmp_path):
    np.testing.assert_array_equal(
        load_target_valid_mask(tmp_path / "case", 3), np.ones(3, dtype=bool)
    )


def test_severe_sensitivity_combines_primary_and_severe_masks(tmp_path):
    case_dir = tmp_path / "case"
    group = zarr.open_group(str(case_dir / "quality_masks.zarr"), mode="w")
    group.create_array("target_valid", data=np.asarray([True, False, True, True]))
    group.create_array(
        "volume_mesh_severe_sensitivity",
        data=np.asarray([False, False, True, False]),
    )
    np.testing.assert_array_equal(
        load_target_valid_mask(case_dir, 4, policy="primary"),
        [True, False, True, True],
    )
    np.testing.assert_array_equal(
        load_target_valid_mask(case_dir, 4, policy="severe_sensitivity"),
        [True, False, False, True],
    )


def test_severe_sensitivity_fails_closed_without_quality_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_target_valid_mask(
            tmp_path / "case", 3, policy="severe_sensitivity"
        )
