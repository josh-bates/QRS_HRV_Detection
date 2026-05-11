import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import score_dataset, score_record
from src.io import FS, load_training_data
from src.qrs_detector import detect_qrs


def test_synthetic_sinusoid_does_not_trigger_many_detections():
    duration_seconds = 30
    t = np.arange(duration_seconds * FS) / FS
    ecg = 1000.0 * np.sin(2 * np.pi * 1.0 * t)

    detections = detect_qrs(ecg, fs=FS)

    expected_heart_rate_count = duration_seconds
    assert len(detections) < 0.05 * expected_heart_rate_count


def test_synthetic_qrs_train_scores_high():
    duration_seconds = 100
    rng = np.random.default_rng(3997)
    ecg = rng.normal(0, 10.0, duration_seconds * FS)
    peaks = np.arange(FS, duration_seconds * FS, FS, dtype=np.int64)

    half_width = 4
    shape = 1000.0 * np.bartlett(2 * half_width + 1)
    for peak in peaks:
        start = peak - half_width
        stop = peak + half_width + 1
        ecg[start:stop] += shape

    detections = detect_qrs(ecg, fs=FS)
    score = score_record(detections, peaks + 1)

    assert score["f1"] > 0.95


def test_recording_1_training_data_scores_high():
    ecg_list, qrs_expert_list = load_training_data()

    detections = detect_qrs(ecg_list[0], fs=FS)
    score = score_record(detections, qrs_expert_list[0])

    assert score["sens"] >= 0.99
    assert score["ppv"] >= 0.99
    assert score["f1"] >= 0.99


def test_training_dataset_revised_detector_scores_high():
    ecg_list, qrs_expert_list = load_training_data()
    detections = [detect_qrs(ecg, fs=FS) for ecg in ecg_list]

    aggregate = score_dataset(detections, qrs_expert_list)["aggregate"]

    assert aggregate["sens"] >= 0.98
    assert aggregate["ppv"] >= 0.98
    assert aggregate["f1"] >= 0.98


if __name__ == "__main__":
    test_synthetic_sinusoid_does_not_trigger_many_detections()
    test_synthetic_qrs_train_scores_high()
    test_recording_1_training_data_scores_high()
    # The all-recording baseline test is intentionally left for pytest or the
    # training runner because it takes several minutes on the full dataset.
    print("Phase 03 detector smoke tests passed.")
