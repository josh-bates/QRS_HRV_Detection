import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import score_dataset, score_hrv_dataset
from src.hrv import compute_hrv, hrv_for_recording
from src.io import FS, HRV_KEYS, load_training_data
from src.qrs_detector import detect_qrs
from src.reference import REFERENCE_HRV


def _average_mape(scores):
    return float(np.mean([scores[key]["mape"] for key in HRV_KEYS]))


def test_expert_qrs_raw_hrv_ceiling_stays_below_8_percent():
    ecg_list, qrs_expert_list = load_training_data()
    rows = [
        compute_hrv(qrs, len(ecg), fs=FS)
        for ecg, qrs in zip(ecg_list, qrs_expert_list)
    ]
    pred = {
        key: np.asarray([getattr(row, key) for row in rows], dtype=np.float64)
        for key in HRV_KEYS
    }

    scores = score_hrv_dataset(REFERENCE_HRV, pred)

    assert _average_mape(scores) < 8.0
    assert scores["avgRR"]["mape"] < 0.5
    assert scores["pNN50"]["mape"] < 2.5


def test_production_pipeline_quality_on_training_set():
    ecg_list, qrs_expert_list = load_training_data()
    detections = [detect_qrs(ecg, fs=FS) for ecg in ecg_list]

    qrs_scores = score_dataset(detections, qrs_expert_list)["aggregate"]
    assert qrs_scores["f1"] >= 0.994

    hrv_rows = [
        hrv_for_recording(qrs, fs=FS, ecg_len_samples=len(ecg))
        for ecg, qrs in zip(ecg_list, detections)
    ]
    pred = {
        key: np.asarray([row[key] for row in hrv_rows], dtype=np.float64)
        for key in HRV_KEYS
    }
    hrv_scores = score_hrv_dataset(REFERENCE_HRV, pred)

    assert _average_mape(hrv_scores) < 8.0
    assert hrv_scores["LF"]["mape"] < 10.0
