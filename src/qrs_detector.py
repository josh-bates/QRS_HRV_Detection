"""Hybrid causal QRS detector with robust fallback recovery.

Pipeline
--------
1. 5-15 Hz Butterworth band-pass (causal ``sosfilt``).
2. Five-point causal derivative filter.
3. Squaring + 150 ms trailing moving-window integration.
4. Stage A – integrated stream: Pan-Tompkins adaptive threshold candidates.
5. R-peak snapping on the raw ECG.
6. Sparse-record fallback: robust raw local maxima when Stage A collapses.
7. Gap-only recovery: insert high-confidence bandpass peaks in long missed-beat
   gaps when Stage A is otherwise plausible but under-detecting.

Non-causal global statistics (whole-record mean / max normalisaton) have been
removed. Adaptive thresholds are seeded from the first two seconds only.

Returned QRS locations are 1-indexed MATLAB sample positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from src.io import FS
from src.qrs_detector_causal_alt import detect_qrs_causal_alt


_MIN_PLAUSIBLE_QRS_RATE_HZ = 0.40
_RAW_FALLBACK_MIN_DURATION_S = 600.0
_RAW_FALLBACK_PERCENTILE = 97.0
_GAP_RECOVERY_MAX_RATE_HZ = 0.90
_GAP_RECOVERY_LONG_GAP_FRACTION = 0.005
_GAP_RECOVERY_PERCENTILE = 98.0
_GAP_RECOVERY_FACTOR = 2.0
_RECOVERY_GAP_MARGIN_S = 0.250
_MERGE_REFRACTORY_S = 0.200


@dataclass(frozen=True)
class PanTompkinsResult:
    """Intermediate outputs useful for debugging and ablation."""

    qrs_samples: np.ndarray
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    candidate_peaks: np.ndarray
    integrated_qrs: np.ndarray
    refinement: object | None = None


def detect_qrs(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detect QRS locations and return 1-indexed MATLAB sample positions."""
    return detect_qrs_with_debug(ecg, fs=fs).qrs_samples


def detect_qrs_with_debug(ecg: np.ndarray, fs: int = FS) -> PanTompkinsResult:
    """Run the full hybrid pipeline and retain intermediate arrays."""
    ecg_array = _as_ecg_vector(ecg)

    # Stage A ─ signal conditioning
    bandpassed = bandpass_filter(ecg_array, fs=fs)
    derivative = derivative_filter(bandpassed, fs=fs)
    squared = square_signal(derivative)
    integrated = moving_window_integrate(squared, fs=fs)

    # Stage A ─ integrated-stream candidates
    int_candidates = find_candidate_peaks(integrated, fs=fs)
    integrated_qrs = adaptive_threshold_qrs(
        integrated=integrated,
        bandpassed=bandpassed,
        candidate_peaks=int_candidates,
        fs=fs,
    )
    snapped_qrs = snap_to_r_peaks(integrated_qrs, ecg_array, fs=fs)

    duration_s = ecg_array.size / fs
    qrs_rate_hz = (snapped_qrs.size / duration_s) if duration_s > 0 else 0.0
    if (
        duration_s >= _RAW_FALLBACK_MIN_DURATION_S
        and qrs_rate_hz < _MIN_PLAUSIBLE_QRS_RATE_HZ
    ):
        qrs_samples = raw_peak_fallback(ecg_array, fs=fs)
    else:
        qrs_samples = recover_long_gap_beats(snapped_qrs, bandpassed, ecg_array, fs=fs)

    alternate_qrs = detect_qrs_causal_alt(ecg_array, fs=fs)
    qrs_samples = select_qrs_sequence(qrs_samples, alternate_qrs, fs=fs)

    return PanTompkinsResult(
        qrs_samples=qrs_samples,
        bandpassed=bandpassed,
        derivative=derivative,
        squared=squared,
        integrated=integrated,
        candidate_peaks=int_candidates,
        integrated_qrs=integrated_qrs,
        refinement=None,
    )





# ─── signal conditioning ──────────────────────────────────────────────────────

def bandpass_filter(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Apply a causal 5-15 Hz Butterworth band-pass filter.

    The filtered signal is normalised by its global peak amplitude so the
    adaptive thresholds in ``adaptive_threshold_qrs`` operate on a consistent
    scale regardless of recording amplitude.  The 5 Hz low-cut removes DC so
    explicit demeaning is unnecessary.
    """
    ecg_array = _as_ecg_vector(ecg)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    nyquist = fs / 2
    low = 5 / nyquist
    high = 15 / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"fs={fs} is incompatible with a 5-15 Hz band-pass")
    sos = butter(3, [low, high], btype="bandpass", output="sos")
    # Subtract a causal DC estimate (first-second mean) before filtering.
    # This prevents large DC offsets from creating a startup transient that
    # would dominate the global-max normalisation and shrink QRS to near-zero.
    dc_estimate = float(np.mean(ecg_array[: max(1, fs)]))
    filtered = sosfilt(sos, ecg_array - dc_estimate)
    max_abs = np.max(np.abs(filtered))
    if max_abs > 0:
        filtered = filtered / max_abs
    return filtered.astype(np.float64)


def derivative_filter(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Apply the causal five-point Pan-Tompkins derivative filter."""
    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    # y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
    kernel = np.array([2, 1, 0, -1, -2], dtype=np.float64) / 8.0
    return lfilter(kernel, [1.0], signal_array)


def square_signal(signal: np.ndarray) -> np.ndarray:
    """Square the derivative signal to emphasise high-slope events."""
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    return s * s


def moving_window_integrate(
    signal: np.ndarray, fs: int = FS, window_seconds: float = 0.150
) -> np.ndarray:
    """Apply trailing moving-window integration over approximately one QRS width."""
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    window = max(1, int(round(window_seconds * fs)))
    kernel = np.ones(window, dtype=np.float64) / window
    return lfilter(kernel, [1.0], s)


# ─── Stage A candidate finders ───────────────────────────────────────────────

def find_candidate_peaks(integrated: np.ndarray, fs: int = FS) -> np.ndarray:
    """Enumerate local maxima with causal one-sample confirmation delay."""
    refractory_samples = max(1, int(round(0.200 * fs)))
    signal = np.asarray(integrated, dtype=np.float64).reshape(-1)
    peaks: list[int] = []
    for idx in range(1, signal.size - 1):
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
    """Pan-Tompkins adaptive threshold pass on the integrated stream.

    Returns 0-indexed candidate positions before Stage B retiming.
    """
    integrated = np.asarray(integrated, dtype=np.float64).reshape(-1)
    bandpassed = np.asarray(bandpassed, dtype=np.float64).reshape(-1)
    peaks = np.asarray(candidate_peaks, dtype=np.int64).reshape(-1)
    if integrated.size != bandpassed.size:
        raise ValueError("integrated and bandpassed must have equal length")
    if peaks.size == 0:
        return np.asarray([], dtype=np.int64)

    # Bandpass is causal-envelope-normalised so the integrated signal is
    # approximately amplitude-stable; original 2-second initialisation works.
    init_len = min(integrated.size, max(1, int(round(2.0 * fs))))
    init_block = integrated[:init_len]
    signal_level_i = float(np.max(init_block) / 3.0)
    noise_level_i  = float(np.mean(init_block) / 2.0)
    threshold_i1   = signal_level_i
    threshold_i2   = noise_level_i

    bp_block = np.abs(bandpassed[:init_len])
    signal_level_f = float(np.max(bp_block) / 3.0)
    noise_level_f  = float(np.mean(bp_block) / 2.0)
    threshold_f1   = signal_level_f
    threshold_f2   = noise_level_f

    refractory = max(1, int(round(0.200 * fs)))
    qrs_width = max(1, int(round(0.150 * fs)))
    twave_window = max(1, int(round(0.360 * fs)))
    slope_window = max(1, int(round(0.075 * fs)))

    qrs_integrated: list[int] = []
    qrs_filtered: list[int] = []
    rr_regular_mean = 0.0

    for peak in peaks:
        peak_value = float(integrated[peak])
        filt_peak_value, filt_peak_index = _bandpass_peak_before(bandpassed, peak, qrs_width)

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
                sb = _searchback_peak(
                    integrated,
                    start=qrs_integrated[-1] + refractory,
                    stop=peak - refractory,
                )
                if sb is not None and integrated[sb] > threshold_i2 and sb - qrs_integrated[-1] >= refractory:
                    qrs_integrated.append(sb)
                    _, sb_filt = _bandpass_peak_before(bandpassed, sb, qrs_width)
                    qrs_filtered.append(sb_filt)
                    signal_level_i = 0.25 * integrated[sb] + 0.75 * signal_level_i
                    sb_fv = bandpassed[sb_filt]
                    if sb_fv > threshold_f2:
                        signal_level_f = 0.25 * sb_fv + 0.75 * signal_level_f

        skip_as_twave = False
        if peak_value >= threshold_i1:
            if len(qrs_integrated) >= 3 and (peak - qrs_integrated[-1]) <= twave_window:
                cs = _mean_recent_slope(integrated, peak, slope_window)
                ps = _mean_recent_slope(integrated, qrs_integrated[-1], slope_window)
                if abs(cs) <= 0.5 * abs(ps):
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


def bandpass_proposals(bandpassed: np.ndarray, fs: int = FS) -> np.ndarray:
    """Second causal proposal stream: local maxima in |bandpassed| with adaptive gate.

    Uses a different failure mode from the integrated stream: it responds to
    high-amplitude QRS events that may not produce a prominent integrated peak
    (e.g. narrow spiky complexes, low heart rate, or post-ectopic beats).
    """
    bp = np.asarray(bandpassed, dtype=np.float64).reshape(-1)
    abs_bp = np.abs(bp)
    refractory = max(1, int(round(0.200 * fs)))
    init_len = min(abs_bp.size, max(1, int(round(2.0 * fs))))

    signal_level = float(np.max(abs_bp[:init_len]) / 3.0)
    noise_level  = float(np.mean(abs_bp[:init_len]) / 2.0)
    threshold    = noise_level + 0.25 * abs(signal_level - noise_level)

    peaks: list[int] = []
    for idx in range(1, abs_bp.size - 1):
        if abs_bp[idx] <= abs_bp[idx - 1] or abs_bp[idx] <= abs_bp[idx + 1]:
            continue
        if peaks and idx - peaks[-1] < refractory:
            if abs_bp[idx] > abs_bp[peaks[-1]]:
                peaks[-1] = idx
            continue
        if abs_bp[idx] >= threshold:
            peaks.append(idx)
            # update threshold adaptively
            if abs_bp[idx] > signal_level:
                signal_level = 0.125 * abs_bp[idx] + 0.875 * signal_level
            else:
                noise_level = 0.125 * abs_bp[idx] + 0.875 * noise_level
            threshold = noise_level + 0.25 * abs(signal_level - noise_level)

    return np.asarray(peaks, dtype=np.int64)


# ─── legacy public function (kept for ablation scripts) ──────────────────────

def snap_to_r_peaks(
    candidate_peaks: np.ndarray,
    signal: np.ndarray,
    fs: int = FS,
    window_seconds: float = 0.250,
) -> np.ndarray:
    """Original snap-to-peak logic (not used in main pipeline; retained for ablation)."""
    candidates = np.asarray(candidate_peaks, dtype=np.int64).reshape(-1)
    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if candidates.size == 0:
        return np.asarray([], dtype=np.int64)
    radius = max(1, int(round(window_seconds * fs)))
    snapped: list[int] = []
    last = -(10**12)
    refractory = max(1, int(round(0.200 * fs)))
    for candidate in candidates:
        start = max(0, int(candidate) - radius)
        stop = min(signal_array.size, int(candidate) + 1)
        if start >= stop:
            continue
        local = signal_array[start:stop]
        si = start + int(np.argmax(np.abs(local)))
        if si - last < refractory:
            if snapped and abs(signal_array[si]) > abs(signal_array[snapped[-1]]):
                snapped[-1] = si
                last = si
            continue
        snapped.append(si)
        last = si
    arr = np.asarray(snapped, dtype=np.int64)
    if arr.size:
        arr = arr[arr >= max(1, radius)]
    return (arr + 1).astype(np.int64)


def raw_peak_fallback(
    ecg: np.ndarray,
    fs: int = FS,
    percentile: float = _RAW_FALLBACK_PERCENTILE,
) -> np.ndarray:
    """Detect prominent raw positive peaks when Stage A returns too few beats."""
    signal = _as_ecg_vector(ecg)
    peaks = find_candidate_peaks(signal, fs=fs)
    if peaks.size == 0:
        return np.asarray([], dtype=np.int64)
    threshold = float(np.percentile(signal, percentile))
    peaks = peaks[signal[peaks] >= threshold]
    return (peaks + 1).astype(np.int64)


def recover_long_gap_beats(
    primary_qrs: np.ndarray,
    bandpassed: np.ndarray,
    ecg: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """Fill long gaps with high-confidence positive bandpass candidates.

    This is intentionally gated. It targets records where Stage A tracks most
    beats but has repeated missed-beat gaps; applying the same candidates
    globally creates false positives on clean records.
    """
    primary = np.asarray(primary_qrs, dtype=np.int64).reshape(-1)
    if primary.size < 10:
        return primary

    duration_s = len(ecg) / fs
    rate_hz = (primary.size / duration_s) if duration_s > 0 else 0.0
    if rate_hz >= _GAP_RECOVERY_MAX_RATE_HZ:
        return primary

    rr = np.diff(primary)
    plausible_rr = rr[
        (rr >= int(round(0.45 * fs)))
        & (rr <= int(round(1.50 * fs)))
    ]
    if plausible_rr.size == 0:
        return primary
    median_rr = float(np.median(plausible_rr))
    long_gap_mask = rr > _GAP_RECOVERY_FACTOR * median_rr
    if float(np.mean(long_gap_mask)) <= _GAP_RECOVERY_LONG_GAP_FRACTION:
        return primary

    backup = _positive_bandpass_percentile_peaks(
        bandpassed, fs=fs, percentile=_GAP_RECOVERY_PERCENTILE
    )
    if backup.size == 0:
        return primary

    margin = int(round(_RECOVERY_GAP_MARGIN_S * fs))
    extras: list[int] = []
    for left_qrs, right_qrs, is_long_gap in zip(primary[:-1], primary[1:], long_gap_mask):
        if not is_long_gap:
            continue
        gap_start = int(left_qrs) + margin
        gap_stop = int(right_qrs) - margin
        if gap_stop <= gap_start:
            continue
        lo = int(np.searchsorted(backup, gap_start))
        hi = int(np.searchsorted(backup, gap_stop + 1))
        extras.extend(backup[lo:hi].tolist())

    if not extras:
        return primary
    return _merge_qrs_by_raw_amplitude(primary, np.asarray(extras, dtype=np.int64), ecg, fs=fs)


def select_qrs_sequence(primary_qrs: np.ndarray, alternate_qrs: np.ndarray, fs: int = FS) -> np.ndarray:
    """Choose between primary and alternate detectors using RR-only diagnostics."""
    primary = np.asarray(primary_qrs, dtype=np.int64).reshape(-1)
    alternate = np.asarray(alternate_qrs, dtype=np.int64).reshape(-1)
    if primary.size < 10 or alternate.size < 10:
        return primary

    ratio = alternate.size / primary.size
    if not (0.995 <= ratio <= 1.055):
        return primary

    p = _rr_sequence_features(primary, fs=fs)
    a = _rr_sequence_features(alternate, fs=fs)
    if p is None or a is None:
        return primary

    cleaner_regular_record = (
        p["short"] <= 0.003
        and a["p01"] >= p["p01"] + 0.02
        and a["p99"] <= p["p99"] - 0.02
        and a["mad"] <= p["mad"] - 0.005
    )
    recovered_regular_record = (
        p["cv"] > 2.0
        and a["cv"] <= 0.75 * p["cv"]
        and a["p99"] <= 1.10
        and a["short"] <= 0.015
    )
    if cleaner_regular_record or recovered_regular_record:
        return alternate
    return primary


# ─── private helpers ─────────────────────────────────────────────────────────


def _positive_bandpass_percentile_peaks(
    bandpassed: np.ndarray,
    fs: int,
    percentile: float,
) -> np.ndarray:
    signal = np.asarray(bandpassed, dtype=np.float64).reshape(-1)
    peaks = find_candidate_peaks(signal, fs=fs)
    if peaks.size == 0:
        return np.asarray([], dtype=np.int64)
    threshold = float(np.percentile(signal, percentile))
    peaks = peaks[signal[peaks] >= threshold]
    return (peaks + 1).astype(np.int64)


def _merge_qrs_by_raw_amplitude(
    primary_qrs: np.ndarray,
    extra_qrs: np.ndarray,
    ecg: np.ndarray,
    fs: int,
) -> np.ndarray:
    raw = np.asarray(ecg, dtype=np.float64).reshape(-1)
    refractory = max(1, int(round(_MERGE_REFRACTORY_S * fs)))
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray(primary_qrs, dtype=np.int64).reshape(-1),
                np.asarray(extra_qrs, dtype=np.int64).reshape(-1),
            ]
        )
    )
    candidates = candidates[(candidates >= 1) & (candidates <= raw.size)]
    if candidates.size == 0:
        return candidates.astype(np.int64)

    zero_indexed = candidates - 1
    merged: list[int] = []
    for idx in zero_indexed.tolist():
        if merged and idx - merged[-1] < refractory:
            if raw[idx] > raw[merged[-1]]:
                merged[-1] = idx
            continue
        merged.append(int(idx))
    return (np.asarray(merged, dtype=np.int64) + 1).astype(np.int64)


def _rr_sequence_features(qrs_samples: np.ndarray, fs: int) -> dict[str, float] | None:
    qrs = np.asarray(qrs_samples, dtype=np.int64).reshape(-1)
    if qrs.size < 10:
        return None
    rr = np.diff(qrs).astype(np.float64) / fs
    rr = rr[np.isfinite(rr)]
    if rr.size < 8 or np.mean(rr) <= 0:
        return None
    median = float(np.median(rr))
    if median <= 0:
        return None
    return {
        "short": float(np.mean(rr < 0.35)),
        "p01": float(np.percentile(rr, 1)),
        "p99": float(np.percentile(rr, 99)),
        "cv": float(np.std(rr) / np.mean(rr)),
        "mad": float(np.median(np.abs(rr - median)) / median),
    }


def _causal_peak_envelope(signal: np.ndarray, decay_alpha: float) -> np.ndarray:
    """Fast-attack / slow-decay causal peak envelope.

    At each sample the envelope either rises instantly to the new absolute value
    (if larger) or decays by ``decay_alpha`` toward zero.  No future samples are
    used.  ``decay_alpha`` close to 1.0 gives a very slow decay (e.g. 0.99999
    at 100 Hz ≈ 12-minute half-life), ensuring the envelope tracks long-term
    amplitude drifts across an overnight recording.
    """
    abs_sig = np.abs(signal)
    env = np.empty_like(abs_sig)
    y = float(abs_sig[0]) if abs_sig.size > 0 else 1e-9
    for i, x in enumerate(abs_sig.tolist()):
        y = x if x > y else decay_alpha * y
        env[i] = y
    return np.maximum(env, 1e-9)


def _as_ecg_vector(ecg: np.ndarray) -> np.ndarray:
    array = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("ECG signal is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("ECG signal contains non-finite values")
    return array


def _bandpass_peak_before(bandpassed: np.ndarray, peak: int, qrs_width: int) -> tuple[float, int]:
    start = max(0, int(peak) - qrs_width)
    stop = min(bandpassed.size, int(peak) + 1)
    if start >= stop:
        return float(bandpassed[int(peak)]), int(peak)
    local = bandpassed[start:stop]
    offset = int(np.argmax(local))
    return float(local[offset]), start + offset


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
