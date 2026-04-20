"""From-scratch Pan-Tompkins QRS detector baseline.

This is the Phase 03 baseline detector for the BMET3997/9997 major project.
It implements the classic Pan-Tompkins structure without importing any
QRS-detection library:

1. 5-15 Hz Butterworth band-pass filtering with ``sosfilt``.
2. Causal five-point derivative filter with ``lfilter``.
3. Squaring.
4. 150 ms trailing moving-window integration with ``lfilter``.
5. Pan-Tompkins-style adaptive thresholds, searchback, and T-wave rejection.
6. R-peak snapping using only samples at or before the candidate time.

Returned QRS locations are 1-indexed MATLAB sample positions, matching the
training annotations and required submission format.

Expected baseline context
-------------------------
The project brief reports that a stock Pan-Tompkins baseline performs very well
on Recording 1 but fails on several later noisy/degraded recordings, producing
an aggregate F1 around 0.87. That failure is intentional for this phase: later
work will add post-detection refinement and robust HRV handling. Recordings
expected to need later attention include roughly 24, 25, 26, 27, 29, 30, 31,
34, and 35.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from src.io import FS


@dataclass(frozen=True)
class PanTompkinsResult:
    """Intermediate outputs useful for debugging and later refinement."""

    qrs_samples: np.ndarray
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    candidate_peaks: np.ndarray
    integrated_qrs: np.ndarray


def detect_qrs(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detect QRS locations and return 1-indexed MATLAB sample positions."""

    return detect_qrs_with_debug(ecg, fs=fs).qrs_samples


def detect_qrs_with_debug(ecg: np.ndarray, fs: int = FS) -> PanTompkinsResult:
    """Run the detector and retain intermediate arrays for diagnostics."""

    ecg_array = _as_ecg_vector(ecg)
    bandpassed = bandpass_filter(ecg_array, fs=fs)
    derivative = derivative_filter(bandpassed, fs=fs)
    squared = square_signal(derivative)
    integrated = moving_window_integrate(squared, fs=fs)
    candidate_peaks = find_candidate_peaks(integrated, fs=fs)
    integrated_qrs = adaptive_threshold_qrs(
        integrated=integrated,
        bandpassed=bandpassed,
        candidate_peaks=candidate_peaks,
        fs=fs,
    )
    snapped = snap_to_r_peaks(integrated_qrs, ecg_array, fs=fs)

    return PanTompkinsResult(
        qrs_samples=snapped,
        bandpassed=bandpassed,
        derivative=derivative,
        squared=squared,
        integrated=integrated,
        candidate_peaks=candidate_peaks,
        integrated_qrs=integrated_qrs,
    )


def bandpass_filter(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Apply a causal 5-15 Hz Butterworth band-pass filter."""

    ecg_array = _as_ecg_vector(ecg)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = fs / 2
    low = 5 / nyquist
    high = 15 / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"fs={fs} is incompatible with a 5-15 Hz band-pass")

    demeaned = ecg_array - np.mean(ecg_array)
    sos = butter(3, [low, high], btype="bandpass", output="sos")
    filtered = sosfilt(sos, demeaned)

    max_abs = np.max(np.abs(filtered))
    if max_abs > 0:
        filtered = filtered / max_abs
    return filtered.astype(np.float64, copy=False)


def derivative_filter(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Apply the causal five-point Pan-Tompkins derivative filter."""

    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    # y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
    kernel = np.array([2, 1, 0, -1, -2], dtype=np.float64) / 8.0
    derivative = lfilter(kernel, [1.0], signal_array)

    max_abs = np.max(np.abs(derivative))
    if max_abs > 0:
        derivative = derivative / max_abs
    return derivative


def square_signal(signal: np.ndarray) -> np.ndarray:
    """Square the derivative signal to emphasize high-slope events."""

    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    return signal_array * signal_array


def moving_window_integrate(
    signal: np.ndarray, fs: int = FS, window_seconds: float = 0.150
) -> np.ndarray:
    """Apply trailing moving-window integration over approximately one QRS width."""

    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    window = max(1, int(round(window_seconds * fs)))
    kernel = np.ones(window, dtype=np.float64) / window
    return lfilter(kernel, [1.0], signal_array)


def find_candidate_peaks(integrated: np.ndarray, fs: int = FS) -> np.ndarray:
    """Enumerate local maxima with causal one-sample confirmation delay."""

    refractory_samples = max(1, int(round(0.200 * fs)))
    signal = np.asarray(integrated, dtype=np.float64).reshape(-1)
    peaks: list[int] = []
    for idx in range(1, signal.size - 1):
        # At sample idx+1, a streaming implementation can confirm whether idx
        # was a local maximum. This is a fixed one-sample decision delay, not
        # zero-phase lookahead.
        if signal[idx] < signal[idx - 1] or signal[idx] <= signal[idx + 1]:
            continue
        if peaks and idx - peaks[-1] < refractory_samples:
            if signal[idx] > signal[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
    return np.asarray(peaks, dtype=np.int64)


def adaptive_threshold_qrs(
    integrated: np.ndarray,
    bandpassed: np.ndarray,
    candidate_peaks: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """Classify candidate peaks using Pan-Tompkins adaptive thresholds.

    Returns 0-indexed QRS locations on the integrated signal. Use
    ``snap_to_r_peaks`` before saving output.
    """

    integrated = np.asarray(integrated, dtype=np.float64).reshape(-1)
    bandpassed = np.asarray(bandpassed, dtype=np.float64).reshape(-1)
    peaks = np.asarray(candidate_peaks, dtype=np.int64).reshape(-1)
    if integrated.size != bandpassed.size:
        raise ValueError("integrated and bandpassed signals must have the same length")
    if peaks.size == 0:
        return np.asarray([], dtype=np.int64)

    init_len = min(integrated.size, max(1, int(round(2.0 * fs))))
    signal_level_i = float(np.max(integrated[:init_len]) / 3.0)
    noise_level_i = float(np.mean(integrated[:init_len]) / 2.0)
    threshold_i1 = signal_level_i
    threshold_i2 = noise_level_i

    signal_level_f = float(np.max(bandpassed[:init_len]) / 3.0)
    noise_level_f = float(np.mean(bandpassed[:init_len]) / 2.0)
    threshold_f1 = signal_level_f
    threshold_f2 = noise_level_f

    refractory = max(1, int(round(0.200 * fs)))
    qrs_width = max(1, int(round(0.150 * fs)))
    twave_window = max(1, int(round(0.360 * fs)))
    slope_window = max(1, int(round(0.075 * fs)))

    qrs_integrated: list[int] = []
    qrs_filtered: list[int] = []
    rr_regular_mean = 0.0

    for peak in peaks:
        peak_value = float(integrated[peak])
        filt_peak_value, filt_peak_index = _bandpass_peak_before(
            bandpassed, peak, qrs_width
        )

        mean_rr = _recent_rr_mean(qrs_integrated)
        if mean_rr > 0 and len(qrs_integrated) >= 9:
            latest_rr = qrs_integrated[-1] - qrs_integrated[-2]
            if latest_rr <= 0.92 * mean_rr or latest_rr >= 1.16 * mean_rr:
                threshold_i1 *= 0.5
                threshold_f1 *= 0.5
            else:
                rr_regular_mean = mean_rr

        test_rr = rr_regular_mean if rr_regular_mean > 0 else mean_rr
        if test_rr > 0 and qrs_integrated:
            elapsed = peak - qrs_integrated[-1]
            if elapsed >= int(round(1.66 * test_rr)):
                searchback_peak = _searchback_peak(
                    integrated,
                    start=qrs_integrated[-1] + refractory,
                    stop=peak - refractory,
                )
                if (
                    searchback_peak is not None
                    and integrated[searchback_peak] > threshold_i2
                    and searchback_peak - qrs_integrated[-1] >= refractory
                ):
                    qrs_integrated.append(searchback_peak)
                    _, searchback_filtered = _bandpass_peak_before(
                        bandpassed, searchback_peak, qrs_width
                    )
                    qrs_filtered.append(searchback_filtered)
                    signal_level_i = 0.25 * integrated[searchback_peak] + 0.75 * signal_level_i
                    searchback_f_value = bandpassed[searchback_filtered]
                    if searchback_f_value > threshold_f2:
                        signal_level_f = 0.25 * searchback_f_value + 0.75 * signal_level_f

        skip_as_twave = False
        if peak_value >= threshold_i1:
            if len(qrs_integrated) >= 3 and (peak - qrs_integrated[-1]) <= twave_window:
                candidate_slope = _mean_recent_slope(integrated, peak, slope_window)
                previous_slope = _mean_recent_slope(
                    integrated, qrs_integrated[-1], slope_window
                )
                if abs(candidate_slope) <= 0.5 * abs(previous_slope):
                    skip_as_twave = True
                    noise_level_i = 0.125 * peak_value + 0.875 * noise_level_i
                    noise_level_f = 0.125 * filt_peak_value + 0.875 * noise_level_f

            if not skip_as_twave:
                if not qrs_integrated or peak - qrs_integrated[-1] >= refractory:
                    qrs_integrated.append(int(peak))
                    if filt_peak_value >= threshold_f1:
                        qrs_filtered.append(int(filt_peak_index))
                        signal_level_f = 0.125 * filt_peak_value + 0.875 * signal_level_f
                    signal_level_i = 0.125 * peak_value + 0.875 * signal_level_i
                else:
                    noise_level_i = 0.125 * peak_value + 0.875 * noise_level_i
                    noise_level_f = 0.125 * filt_peak_value + 0.875 * noise_level_f
        elif peak_value >= threshold_i2:
            noise_level_i = 0.125 * peak_value + 0.875 * noise_level_i
            noise_level_f = 0.125 * filt_peak_value + 0.875 * noise_level_f
        else:
            noise_level_i = 0.125 * peak_value + 0.875 * noise_level_i
            noise_level_f = 0.125 * filt_peak_value + 0.875 * noise_level_f

        if noise_level_i != 0 or signal_level_i != 0:
            threshold_i1 = noise_level_i + 0.25 * abs(signal_level_i - noise_level_i)
            threshold_i2 = 0.5 * threshold_i1
        if noise_level_f != 0 or signal_level_f != 0:
            threshold_f1 = noise_level_f + 0.25 * abs(signal_level_f - noise_level_f)
            threshold_f2 = 0.5 * threshold_f1

    if qrs_filtered:
        return np.asarray(qrs_filtered, dtype=np.int64)
    return np.asarray(qrs_integrated, dtype=np.int64)


def snap_to_r_peaks(
    candidate_peaks: np.ndarray,
    signal: np.ndarray,
    fs: int = FS,
    window_seconds: float = 0.250,
) -> np.ndarray:
    """Snap detections to past local absolute extrema and return 1-indexed samples."""

    candidates = np.asarray(candidate_peaks, dtype=np.int64).reshape(-1)
    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)

    radius = max(1, int(round(window_seconds * fs)))
    snapped: list[int] = []
    last = -10**12
    refractory = max(1, int(round(0.200 * fs)))
    for candidate in candidates:
        start = max(0, int(candidate) - radius)
        stop = min(signal_array.size, int(candidate) + 1)
        if start >= stop:
            continue
        local = signal_array[start:stop]
        snapped_index = start + int(np.argmax(np.abs(local)))
        if snapped_index - last < refractory:
            if snapped and abs(signal_array[snapped_index]) > abs(signal_array[snapped[-1]]):
                snapped[-1] = snapped_index
                last = snapped_index
            continue
        snapped.append(snapped_index)
        last = snapped_index

    snapped_array = np.asarray(snapped, dtype=np.int64)
    if snapped_array.size:
        edge_margin = max(1, radius)
        keep = snapped_array >= edge_margin
        snapped_array = snapped_array[keep]

    return (snapped_array + 1).astype(np.int64, copy=False)


def _as_ecg_vector(ecg: np.ndarray) -> np.ndarray:
    array = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("ECG signal is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("ECG signal contains non-finite values")
    return array


def _bandpass_peak_before(
    bandpassed: np.ndarray, peak: int, qrs_width: int
) -> tuple[float, int]:
    start = max(0, int(peak) - qrs_width)
    stop = min(bandpassed.size, int(peak) + 1)
    if start >= stop:
        return float(bandpassed[int(peak)]), int(peak)
    local = bandpassed[start:stop]
    offset = int(np.argmax(local))
    index = start + offset
    return float(local[offset]), index


def _searchback_peak(integrated: np.ndarray, start: int, stop: int) -> int | None:
    start = max(0, int(start))
    stop = min(integrated.size, int(stop))
    if start >= stop:
        return None
    return start + int(np.argmax(integrated[start:stop]))


def _recent_rr_mean(qrs_integrated: list[int]) -> float:
    if len(qrs_integrated) < 9:
        return 0.0
    return float(np.mean(np.diff(qrs_integrated[-9:])))


def _mean_recent_slope(signal: np.ndarray, peak: int, slope_window: int) -> float:
    start = max(0, int(peak) - slope_window)
    stop = min(signal.size, int(peak) + 1)
    if stop - start < 2:
        return 0.0
    return float(np.mean(np.diff(signal[start:stop])))
