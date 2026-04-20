import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    DEFAULT_QRS_TOLERANCE_SAMPLES,
    mape,
    mape_stats,
    match_qrs,
    score_dataset,
    score_hrv_dataset,
    score_record,
)
from src.io import HRV_KEYS, load_training_data


def test_default_qrs_tolerance_is_50_ms_at_100_hz():
    assert DEFAULT_QRS_TOLERANCE_SAMPLES == 5


def test_synthetic_qrs_matching_with_default_tolerance():
    expert = np.array([100, 200, 300, 400])
    detected = np.array([101, 205, 299, 500])

    tp, fp, fn = match_qrs(
        detected, expert, tol_samples=DEFAULT_QRS_TOLERANCE_SAMPLES
    )

    assert (tp, fp, fn) == (3, 1, 1)


def test_synthetic_qrs_matching_with_zero_tolerance():
    expert = np.array([100, 200, 300, 400])
    detected = np.array([101, 205, 299, 500])

    tp, fp, fn = match_qrs(detected, expert, tol_samples=0)

    assert (tp, fp, fn) == (0, 4, 4)


def test_qrs_matching_handles_unsorted_inputs():
    expert = np.array([400, 100, 300, 200])
    detected = np.array([500, 299, 101, 205])

    assert match_qrs(
        detected, expert, tol_samples=DEFAULT_QRS_TOLERANCE_SAMPLES
    ) == (3, 1, 1)


def test_qrs_matching_uses_inclusive_tolerance():
    expert = np.array([100])
    detected = np.array([105])

    assert match_qrs(
        detected, expert, tol_samples=DEFAULT_QRS_TOLERANCE_SAMPLES
    ) == (1, 0, 0)


def test_score_record_empty_detections():
    expert = np.array([100, 200, 300])
    detected = np.array([], dtype=np.int64)

    score = score_record(detected, expert)

    assert score["tp"] == 0
    assert score["fp"] == 0
    assert score["fn"] == 3
    assert score["sens"] == 0.0
    assert score["ppv"] == 0.0
    assert score["f1"] == 0.0


def test_score_record_expert_as_detection_is_perfect():
    # Phase 02 Option A from the plan:
    # The Pan-Tompkins Recording 1 baseline assertion is deferred until Phase 03,
    # when this repository has its own detector output. This test validates the
    # scorer against the brief's smoke case: expert QRS passed as both detection
    # and truth must score perfectly.
    _, qrs_expert_list = load_training_data()
    expert = qrs_expert_list[0]

    score = score_record(expert, expert, tol_samples=DEFAULT_QRS_TOLERANCE_SAMPLES)

    assert score["tp"] == len(expert)
    assert score["fp"] == 0
    assert score["fn"] == 0
    assert score["sens"] == 1.0
    assert score["ppv"] == 1.0
    assert score["f1"] == 1.0


def test_score_dataset_aggregates_counts_before_metrics():
    expert = [np.array([100, 200]), np.array([100, 200])]
    detected = [np.array([100, 999]), np.array([100, 200])]

    result = score_dataset(detected, expert, tol_samples=0)
    aggregate = result["aggregate"]

    assert len(result["records"]) == 2
    assert aggregate["tp"] == 3
    assert aggregate["fp"] == 1
    assert aggregate["fn"] == 1
    assert np.isclose(aggregate["sens"], 0.75)
    assert np.isclose(aggregate["ppv"], 0.75)
    assert np.isclose(aggregate["f1"], 0.75)


def test_synthetic_hrv_mape():
    ref = np.array([100, 200, 400, 800])
    pred = np.array([110, 180, 420, 760])

    assert np.isclose(mape(ref, pred), 7.5)


def test_nan_handling_in_mape():
    ref = np.array([100, 200])
    pred = np.array([110, np.nan])

    stats = mape_stats(ref, pred)

    assert np.isclose(stats["mape"], 10.0)
    assert stats["n_valid"] == 1
    assert stats["n_nan"] == 1


def test_score_hrv_dataset_scores_all_required_parameters():
    ref = {key: np.array([100, 200, 400, 800], dtype=float) for key in HRV_KEYS}
    pred = {key: np.array([110, 180, 420, 760], dtype=float) for key in HRV_KEYS}

    result = score_hrv_dataset(ref, pred)

    assert set(result) == set(HRV_KEYS)
    for key in HRV_KEYS:
        assert np.isclose(result[key]["mape"], 7.5)
        assert result[key]["n_valid"] == 4
        assert result[key]["n_nan"] == 0


if __name__ == "__main__":
    test_default_qrs_tolerance_is_50_ms_at_100_hz()
    test_synthetic_qrs_matching_with_default_tolerance()
    test_synthetic_qrs_matching_with_zero_tolerance()
    test_qrs_matching_handles_unsorted_inputs()
    test_qrs_matching_uses_inclusive_tolerance()
    test_score_record_empty_detections()
    test_score_record_expert_as_detection_is_perfect()
    test_score_dataset_aggregates_counts_before_metrics()
    test_synthetic_hrv_mape()
    test_nan_handling_in_mape()
    test_score_hrv_dataset_scores_all_required_parameters()
    print("Phase 02 evaluation tests passed.")
