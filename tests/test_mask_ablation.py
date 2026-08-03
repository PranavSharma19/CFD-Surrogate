from aorta_surrogate.training.mask_ablation import _region_balanced


def test_region_balanced_report_keeps_sparse_region_but_excludes_it_from_supported_macro():
    regions = {
        "large": {
            "sampled_node_count": 2000,
            "wss_vector_mae_pa": 1.0,
            "wss_magnitude_mae_pa": 0.8,
            "wss_magnitude_relative_error": 0.5,
        },
        "sparse": {
            "sampled_node_count": 100,
            "wss_vector_mae_pa": 2.0,
            "wss_magnitude_mae_pa": 1.5,
            "wss_magnitude_relative_error": 2.5,
        },
    }
    report = _region_balanced(regions)
    assert report["macro_wss_magnitude_relative_error"] == 1.5
    assert report["support_aware"]["included_regions"] == ["large"]
    assert report["support_aware"]["excluded_regions"] == ["sparse"]
    assert (
        report["support_aware"]["macro_wss_magnitude_relative_error"] == 0.5
    )
