from aorta_surrogate.data.dataset_report import create_patient_split


def test_patient_split_is_disjoint_and_deterministic():
    rows = [
        {"case_id": f"{number:04d}_H_ABAO_AAA", "wall_nodes": number * 100, "heart_period_seconds": 0.7 + number / 1000}
        for number in range(31, 46)
    ]
    first = create_patient_split(rows)
    second = create_patient_split(list(reversed(rows)))

    assert first == second
    assert len(first["locked_test"]) == 3
    assert len(first["development"]) == 12
    assert set(first["locked_test"]).isdisjoint(first["development"])
    folded = [case for fold in first["development_cv_folds"] for case in fold]
    assert sorted(folded) == first["development"]
