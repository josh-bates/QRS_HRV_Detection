"""Alternate causal Pan-Tompkins detector adapted from the updated submission.

This path is used only as a guarded fallback by ``src.qrs_detector``.  It keeps
the useful chunked threshold refresh and group-delay correction from the
downloaded assignment code without changing the public project API.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, lfilter, lfilter_zi

from src.io import FS


def detect_qrs_causal_alt(ecg: np.ndarray, fs: int = FS, chunk_minutes: int = 30) -> np.ndarray:
    """Return alternate QRS detections as 1-indexed MATLAB sample positions."""
    signal = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return np.asarray([], dtype=np.int64)

    corrected, _ = _polarity_correct(signal, fs=fs)
    chunk_samples = int(chunk_minutes * 60 * fs)
    overlap = int(2 * fs)
    chunks: list[np.ndarray] = []

    for start in range(0, corrected.size, chunk_samples):
        end = min(start + chunk_samples, corrected.size)
        lo = max(0, start - overlap)
        qrs = _detect_core(corrected[lo:end], fs=fs) + lo
        chunks.append(qrs[(qrs >= start) & (qrs < end)])

    if not chunks:
        return np.asarray([], dtype=np.int64)
    qrs_zero = np.concatenate(chunks)
    qrs_zero = _post_refine(qrs_zero, corrected, fs=fs)
    qrs_zero = qrs_zero[(qrs_zero >= 0) & (qrs_zero < corrected.size)]
    return (np.sort(np.unique(qrs_zero)) + 1).astype(np.int64)


def _detect_core(ecg: np.ndarray, fs: int) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.size < 3 * fs:
        return np.asarray([], dtype=np.int64)

    nyquist = fs / 2.0
    b, a = butter(3, [5.0 / nyquist, 15.0 / nyquist], btype="band")
    zi = lfilter_zi(b, a) * ecg[0]
    bandpassed, _ = lfilter(b, a, ecg, zi=zi)
    max_bp = np.max(np.abs(bandpassed))
    if max_bp > 0:
        bandpassed = bandpassed / max_bp

    derivative = lfilter(np.array([1, 2, 0, -2, -1], dtype=np.float64) * fs / 8.0, 1, bandpassed)
    max_derivative = np.max(np.abs(derivative))
    if max_derivative > 0:
        derivative = derivative / max_derivative

    squared = derivative * derivative
    window = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(window, dtype=np.float64) / window, mode="full")[: squared.size]

    locs, _ = find_peaks(integrated, distance=max(1, int(round(0.200 * fs))))
    if locs.size == 0:
        return np.asarray([], dtype=np.int64)
    peaks = integrated[locs]

    init_len = min(integrated.size, max(1, 2 * fs))
    threshold_i = float(np.max(integrated[:init_len]) / 3.0)
    noise_i = float(np.mean(integrated[:init_len]) / 2.0)
    signal_i = threshold_i
    threshold_f = float(np.max(np.abs(bandpassed[:init_len])) / 3.0)
    noise_f = float(np.mean(np.abs(bandpassed[:init_len])) / 2.0)
    signal_f = threshold_f

    qrs_i: list[int] = []
    qrs_raw: list[int] = []
    selected_rr = 0.0
    mean_rr = 0.0
    search_width = max(1, int(round(0.150 * fs)))
    refractory = max(1, int(round(0.200 * fs)))

    for loc, peak in zip(locs.tolist(), peaks.tolist()):
        lo = max(0, loc - search_width)
        hi = min(loc + 1, bandpassed.size)
        if hi <= lo:
            continue
        segment = bandpassed[lo:hi]
        rel = int(np.argmax(np.abs(segment)))
        filt_value = float(np.abs(segment[rel]))
        filt_idx = lo + rel

        if len(qrs_i) >= 9:
            recent_rr = np.diff(qrs_i[-9:])
            mean_rr = float(np.mean(recent_rr))
            latest_rr = qrs_i[-1] - qrs_i[-2]
            if latest_rr <= 0.92 * mean_rr or latest_rr >= 1.16 * mean_rr:
                threshold_i *= 0.5
                threshold_f *= 0.5
            else:
                selected_rr = mean_rr

        test_rr = selected_rr if selected_rr else mean_rr
        if test_rr and qrs_i and (loc - qrs_i[-1]) >= round(1.66 * test_rr):
            sb_lo = qrs_i[-1] + refractory
            sb_hi = loc - refractory
            if sb_hi > sb_lo and sb_hi <= integrated.size:
                sb_seg = integrated[sb_lo:sb_hi]
                if sb_seg.size:
                    sb_rel = int(np.argmax(sb_seg))
                    sb = sb_lo + sb_rel
                    sb_peak = float(sb_seg[sb_rel])
                    if sb_peak > noise_i:
                        qrs_i.append(sb)
                        f_lo = max(0, sb - search_width)
                        f_hi = min(sb + 1, bandpassed.size)
                        f_seg = bandpassed[f_lo:f_hi]
                        if f_seg.size:
                            f_rel = int(np.argmax(np.abs(f_seg)))
                            f_value = float(np.abs(f_seg[f_rel]))
                            if f_value > noise_f:
                                qrs_raw.append(f_lo + f_rel)
                                signal_f = 0.25 * f_value + 0.75 * signal_f
                        signal_i = 0.25 * sb_peak + 0.75 * signal_i

        skip = False
        if peak >= threshold_i:
            if len(qrs_i) >= 3 and (loc - qrs_i[-1]) <= round(0.360 * fs):
                slope_window = max(1, int(round(0.075 * fs)))
                if loc - slope_window >= 0 and qrs_i[-1] - slope_window >= 0:
                    slope_current = np.mean(np.diff(integrated[loc - slope_window : loc]))
                    slope_previous = np.mean(np.diff(integrated[qrs_i[-1] - slope_window : qrs_i[-1]]))
                    if abs(slope_current) <= 0.5 * abs(slope_previous):
                        skip = True
                        noise_f = 0.125 * filt_value + 0.875 * noise_f
                        noise_i = 0.125 * peak + 0.875 * noise_i
            if not skip:
                qrs_i.append(int(loc))
                if filt_value >= threshold_f:
                    qrs_raw.append(int(filt_idx))
                    signal_f = 0.125 * filt_value + 0.875 * signal_f
                signal_i = 0.125 * peak + 0.875 * signal_i
        else:
            noise_f = 0.125 * filt_value + 0.875 * noise_f
            noise_i = 0.125 * peak + 0.875 * noise_i

        threshold_i = noise_i + 0.25 * abs(signal_i - noise_i)
        noise_threshold_i = 0.5 * threshold_i
        if peak < noise_threshold_i:
            noise_i = 0.125 * peak + 0.875 * noise_i
        threshold_f = noise_f + 0.25 * abs(signal_f - noise_f)

    qrs = np.asarray(sorted(set(qrs_raw)), dtype=np.int64)
    group_delay = int(round(0.058 * fs))
    qrs = qrs - group_delay
    return qrs[qrs >= 0]


def _polarity_correct(ecg: np.ndarray, fs: int) -> tuple[np.ndarray, bool]:
    n = min(int(60 * fs), ecg.size)
    segment = ecg[:n] - np.mean(ecg[:n])
    top = np.percentile(segment, 99.5)
    bottom = np.percentile(segment, 0.5)
    if abs(bottom) > abs(top) * 1.3:
        return -ecg, True
    return ecg, False


def _post_refine(qrs: np.ndarray, ecg: np.ndarray, fs: int) -> np.ndarray:
    if qrs.size < 3:
        return qrs

    qrs = np.sort(np.unique(qrs.astype(np.int64)))
    refractory = int(round(0.250 * fs))
    cleaned = [int(qrs[0])]
    for beat in qrs[1:].tolist():
        if beat - cleaned[-1] >= refractory:
            cleaned.append(int(beat))
        elif abs(ecg[beat]) > abs(ecg[cleaned[-1]]):
            cleaned[-1] = int(beat)

    qrs = np.asarray(cleaned, dtype=np.int64)
    rr = np.diff(qrs)
    recovered = [int(qrs[0])]
    for i in range(1, qrs.size):
        gap = int(qrs[i] - recovered[-1])
        past = rr[max(0, i - 8) : i]
        past = past[(past >= 0.25 * fs) & (past <= 2.0 * fs)]
        if past.size >= 3:
            med_rr = float(np.median(past))
            if gap > 1.5 * med_rr and gap > 0.8 * fs:
                n_missed = int(round(gap / med_rr)) - 1
                lo = recovered[-1] + int(0.2 * fs)
                hi = int(qrs[i]) - int(0.2 * fs)
                if hi > lo and n_missed > 0:
                    segment = np.abs(ecg[lo:hi]).astype(np.float64)
                    threshold = np.median(np.abs(ecg[max(0, recovered[-1] - int(5 * fs)) : recovered[-1]])) * 3.0
                    for _ in range(n_missed):
                        if segment.size == 0:
                            break
                        peak = int(np.argmax(segment))
                        if segment[peak] < threshold:
                            break
                        recovered.append(lo + peak)
                        block_lo = max(0, peak - int(0.2 * fs))
                        block_hi = min(segment.size, peak + int(0.2 * fs))
                        segment[block_lo:block_hi] = 0.0
        recovered.append(int(qrs[i]))

    return np.asarray(sorted(set(recovered)), dtype=np.int64)
