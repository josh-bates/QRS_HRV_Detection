"""Refit _OUTPUT_CALIBRATION using LOO-CV on training data.

Runs the production QRS detector and pre-calibration HRV pipeline on all 35
training recordings, then for each HRV field fits:

  - scale-only:   predicted_cal = scale * predicted
  - scale+offset: predicted_cal = scale * predicted + offset

The fit that achieves lower LOO-CV MAPE is reported. Fields where calibration
does not improve LOO-CV MAPE are left at identity (1.0, 0.0).

Usage:
    python3.12 scripts/refit_calibration.py

Output:
    - Per-field table: uncalibrated MAPE, LOO-CV MAPE (scale), LOO-CV MAPE
      (scale+offset), recommended transform.
    - Copy-pasteable _OUTPUT_CALIBRATION dict for src/hrv.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hrv import HRV_KEYS, compute_hrv
from src.io import FS, load_training_data
from src.qrs_detector import detect_qrs
from src.reference import REFERENCE_HRV


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _mape(pred: np.ndarray, ref: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(ref) & (ref != 0)
    if not valid.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((pred[valid] - ref[valid]) / ref[valid])))


def _fit_scale_loo(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Fit a global scale via median ratio, evaluate with LOO-CV.

    Returns (scale, uncal_mape, loo_mape).
    """
    n = len(pred)
    valid = np.isfinite(pred) & np.isfinite(ref) & (ref != 0) & (pred != 0)
    uncal_mape = _mape(pred, ref)

    loo_errors: list[float] = []
    for i in range(n):
        train = valid.copy()
        train[i] = False
        if train.sum() < 2:
            continue
        scale = float(np.median(ref[train] / pred[train]))
        if valid[i]:
            err = abs(scale * pred[i] - ref[i]) / abs(ref[i])
            loo_errors.append(err)

    loo_mape = float(100.0 * np.mean(loo_errors)) if loo_errors else uncal_mape
    scale_full = float(np.median(ref[valid] / pred[valid])) if valid.any() else 1.0
    return scale_full, uncal_mape, loo_mape


def _fit_affine_loo(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Fit a scale+offset via OLS, evaluate with LOO-CV.

    Returns (scale, offset, loo_mape).
    """
    n = len(pred)
    valid = np.isfinite(pred) & np.isfinite(ref) & (ref != 0) & (pred != 0)

    loo_errors: list[float] = []
    for i in range(n):
        train = valid.copy()
        train[i] = False
        if train.sum() < 3:
            continue
        p_tr, r_tr = pred[train], ref[train]
        coeffs = np.polyfit(p_tr, r_tr, 1)  # ref = scale*pred + offset
        scale, offset = float(coeffs[0]), float(coeffs[1])
        if valid[i]:
            err = abs(scale * pred[i] + offset - ref[i]) / abs(ref[i])
            loo_errors.append(err)

    loo_mape = float(100.0 * np.mean(loo_errors)) if loo_errors else float("nan")

    # Full-data affine fit for final constants
    if valid.sum() >= 3:
        coeffs_full = np.polyfit(pred[valid], ref[valid], 1)
        scale_full = float(coeffs_full[0])
        offset_full = float(coeffs_full[1])
    else:
        scale_full, offset_full = 1.0, 0.0

    return scale_full, offset_full, loo_mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading training data …")
    ecg_list, _ = load_training_data()
    n = len(ecg_list)
    print(f"Loaded {n} recordings.\n")

    # Optional: test HF window clipping alongside calibration. Set to None to
    # disable (matches current production) or e.g. 90.0 to clip the top 10 % of
    # per-window HF values within each recording before averaging.
    HF_CLIP_PCT: float | None = None  # change to 90.0 to trial clipping

    print("Running QRS detection + pre-calibration HRV …")
    if HF_CLIP_PCT is not None:
        print(f"  HF window clipping enabled at {HF_CLIP_PCT}th percentile.")
    t0 = time.perf_counter()
    pred_raw: list[dict[str, float]] = []
    for idx, ecg in enumerate(ecg_list, start=1):
        qrs = detect_qrs(ecg, fs=FS)
        hrv = compute_hrv(qrs, ecg_len_samples=len(ecg), fs=FS, hf_clip_pct=HF_CLIP_PCT)
        pred_raw.append({k: float(getattr(hrv, k)) for k in HRV_KEYS})
        print(f"  Record {idx:02d}: {len(qrs)} QRS", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s.\n")

    # Collect into arrays
    pred_arrays: dict[str, np.ndarray] = {}
    for key in HRV_KEYS:
        pred_arrays[key] = np.array([r[key] for r in pred_raw], dtype=np.float64)

    ref_arrays: dict[str, np.ndarray] = {
        key: np.asarray(REFERENCE_HRV[key], dtype=np.float64) for key in HRV_KEYS
    }

    print(
        f"{'Field':<14} {'Uncal%':>8} {'ScaleOnly-LOO%':>16} {'Affine-LOO%':>13} "
        f"{'Best':>8} {'Scale':>10} {'Offset':>10}"
    )
    print("-" * 82)

    recommended: dict[str, tuple[float, float]] = {}

    for key in HRV_KEYS:
        pred = pred_arrays[key]
        ref = ref_arrays[key]

        scale_so, uncal_mape, loo_so = _fit_scale_loo(pred, ref)
        scale_af, offset_af, loo_af = _fit_affine_loo(pred, ref)

        # Choose best transform that improves over uncalibrated
        best_loo = min(loo_so, loo_af, uncal_mape)

        if best_loo >= uncal_mape - 0.05:
            # No meaningful improvement — leave at identity
            rec_scale, rec_offset = 1.0, 0.0
            best_label = "none"
        elif loo_so <= loo_af:
            rec_scale, rec_offset = scale_so, 0.0
            best_label = "scale"
        else:
            rec_scale, rec_offset = scale_af, offset_af
            best_label = "affine"

        recommended[key] = (rec_scale, rec_offset)

        print(
            f"{key:<14} {uncal_mape:>8.3f} {loo_so:>16.3f} {loo_af:>13.3f} "
            f"{best_label:>8} {rec_scale:>10.6f} {rec_offset:>10.4f}"
        )

    print()
    print("Current calibration (from src/hrv.py):")
    current = {
        "avgRR":       (1.0, 0.0),
        "sdRR":        (0.9866953876221705, 0.0),
        "RMSSD":       (1.0123328025886276, -1.2176481640411145),
        "pNN50":       (1.0, 0.0),
        "LF":          (1.1610889821772639, 0.0),
        "HF":          (1.0, 0.0),
        "LF_HFratio":  (1.0, 0.0),
    }
    for key in HRV_KEYS:
        sc, off = current[key]
        print(f"  {key:<14}: scale={sc:.6f}  offset={off:.4f}")

    print()
    print("Recommended _OUTPUT_CALIBRATION (paste into src/hrv.py):")
    print("_OUTPUT_CALIBRATION: dict[str, tuple[float, float]] = {")
    for key in HRV_KEYS:
        sc, off = recommended[key]
        print(f'    "{key}": ({sc!r}, {off!r}),')
    print("}")

    print()
    print("Per-field effect of recommended calibration on full-data MAPE:")
    print(f"{'Field':<14} {'Before%':>8} {'After%':>8} {'Delta%':>9}")
    print("-" * 42)
    total_before = 0.0
    total_after = 0.0
    for key in HRV_KEYS:
        pred = pred_arrays[key]
        ref = ref_arrays[key]
        sc, off = recommended[key]
        pred_cal = sc * pred + off
        before = _mape(pred, ref)
        after = _mape(pred_cal, ref)
        total_before += before
        total_after += after
        delta = after - before
        print(f"{key:<14} {before:>8.3f} {after:>8.3f} {delta:>+9.3f}")
    print("-" * 42)
    avg_before = total_before / len(HRV_KEYS)
    avg_after = total_after / len(HRV_KEYS)
    print(f"{'Average':<14} {avg_before:>8.3f} {avg_after:>8.3f} {avg_after - avg_before:>+9.3f}")


if __name__ == "__main__":
    main()
