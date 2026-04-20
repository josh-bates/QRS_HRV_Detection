import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hrv import (
    avg_rr,
    flag_rr,
    hrv_for_recording,
    lf_hf_welch,
    pnn50,
    rmssd,
    rr_from_qrs,
    sd_rr,
    segment_windows,
)
from src.io import FS, load_training_data


def test_rr_derivation():
    qrs = np.array([100, 200, 350, 500])

    rr_ms = rr_from_qrs(qrs, fs=100)

    np.testing.assert_allclose(rr_ms, np.array([1000, 1500, 1500], dtype=float))


def test_rr_flagging():
    rr_ms = np.array([500, 3000, 1000, 100], dtype=float)

    mask = flag_rr(rr_ms)

    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


def test_time_domain_parameters_on_clean_rr():
    rr = np.array([800, 820, 780, 810, 790], dtype=float)
    mask = np.ones(rr.size, dtype=bool)

    assert avg_rr(rr) == 800.0
    assert np.isclose(sd_rr(rr), np.std(rr, ddof=1))
    assert np.isclose(rmssd(rr, mask), np.sqrt(np.mean(np.array([20, -40, 30, -20]) ** 2)))
    assert pnn50(rr, mask) == 0.0


def test_pnn50_known_crossings():
    rr = np.array([800, 870, 810, 750, 820], dtype=float)
    mask = np.ones(rr.size, dtype=bool)

    assert pnn50(rr, mask) == 100.0


def test_diff_features_do_not_cross_flagged_intervals():
    rr = np.array([800, 100, 870], dtype=float)
    mask = np.array([True, False, True])

    assert np.isnan(rmssd(rr, mask))
    assert np.isnan(pnn50(rr, mask))


def test_window_validity_threshold_edges():
    rr = np.full(600, 1000.0)
    qrs = np.concatenate(([1], 1 + np.cumsum(rr / 10).astype(np.int64)))

    mask = np.zeros_like(rr, dtype=bool)
    mask[:239] = True
    windows = segment_windows(qrs, rr, mask, fs=100)
    assert not windows[0].is_valid_window

    mask[:240] = True
    windows = segment_windows(qrs, rr, mask, fs=100)
    assert windows[0].is_valid_window


def test_frequency_domain_hf_dominates_for_hf_modulated_rr():
    duration_seconds = 300
    rr_times = np.arange(0, duration_seconds, 1.0)
    rr_ms = 1000.0 + 30.0 * np.sin(2 * np.pi * 0.25 * rr_times)
    mask = np.ones(rr_ms.size, dtype=bool)

    lf, hf, ratio = lf_hf_welch(rr_times, rr_ms, mask)

    assert np.isfinite(lf)
    assert np.isfinite(hf)
    assert hf > 10 * max(lf, 1e-12)
    assert ratio < 0.1


def test_recording_1_expert_qrs_avg_rr_matches_reference():
    _, qrs_expert_list = load_training_data()

    result = hrv_for_recording(qrs_expert_list[0], fs=FS)

    assert abs(result["avgRR"] - 992) / 992 < 0.02


if __name__ == "__main__":
    test_rr_derivation()
    test_rr_flagging()
    test_time_domain_parameters_on_clean_rr()
    test_pnn50_known_crossings()
    test_diff_features_do_not_cross_flagged_intervals()
    test_window_validity_threshold_edges()
    test_frequency_domain_hf_dominates_for_hf_modulated_rr()
    test_recording_1_expert_qrs_avg_rr_matches_reference()
    print("Phase 04 HRV tests passed.")

