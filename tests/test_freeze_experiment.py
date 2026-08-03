import pytest

from aorta_surrogate.training.freeze_experiment import _validate_contract


def _fixtures():
    development = ["a", "b", "c"]
    locked = ["z"]
    folds = [["a"], ["b"], ["c"]]
    split = {
        "development": development,
        "locked_test": locked,
        "development_cv_folds": folds,
    }
    contract = {
        "data": {
            "development_case_ids": development,
            "locked_case_ids_not_staged": locked,
            "development_cv_folds": folds,
            "runtime_case_artifacts": [
                "case_manifest.json",
                "boundary_conditions.json",
                "features.zarr",
                "targets.zarr",
                "quality_masks.zarr",
            ],
            "source_archives_required": False,
        }
    }
    return contract, split


def test_frozen_contract_must_match_split_exactly():
    contract, split = _fixtures()
    assert _validate_contract(contract, split) == (["a", "b", "c"], ["z"])


def test_frozen_contract_rejects_locked_development_overlap():
    contract, split = _fixtures()
    contract["data"]["development_case_ids"] = ["a", "b", "z"]
    split["development"] = ["a", "b", "z"]
    contract["data"]["development_cv_folds"] = [["a"], ["b"], ["z"]]
    split["development_cv_folds"] = [["a"], ["b"], ["z"]]
    with pytest.raises(ValueError, match="overlap"):
        _validate_contract(contract, split)


def test_frozen_contract_rejects_duplicate_fold_patient():
    contract, split = _fixtures()
    contract["data"]["development_cv_folds"] = [["a"], ["a"], ["c"]]
    split["development_cv_folds"] = [["a"], ["a"], ["c"]]
    with pytest.raises(ValueError, match="partition"):
        _validate_contract(contract, split)
