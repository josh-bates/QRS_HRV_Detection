"""Evaluation helpers for QRS detection and HRV outputs.

This module mirrors the scoring rules in the BMET3997/9997 major-project
brief. QRS indices are assumed to be 1-indexed MATLAB sample positions, matching
``src.io`` and the required submission format.

The QRS scorer intentionally follows the brief's non-one-to-one rule:

* TP counts expert beats that have at least one detection within tolerance.
* FN counts expert beats that have no detection within tolerance.
* FP counts detections that have no expert beat within tolerance.

No bipartite matching is applied. This differs from stricter event-matching
metrics, but it is the rule shown in the assignment brief.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.io import HRV_KEYS


DEFAULT_QRS_TOLERANCE_SAMPLES = 5
"""Default QRS tolerance: ±50 ms at fs = 100 Hz."""


def match_qrs(
    qrs_detected: Iterable[int] | np.ndarray,
    qrs_expert: Iterable[int] | np.ndarray,
    tol_samples: int = DEFAULT_QRS_TOLERANCE_SAMPLES,
) -> tuple[int, int, int]:
    """Count QRS true positives, false positives, and false negatives.

    Parameters
    ----------
    qrs_detected:
        Detected QRS sample locations, using 1-indexed MATLAB positions.
    qrs_expert:
        Expert/reference QRS sample locations, using 1-indexed MATLAB positions.
    tol_samples:
        Inclusive matching tolerance in samples. The project default is
        ``5`` samples, equivalent to ±50 ms at 100 Hz.

    Returns
    -------
    tuple[int, int, int]
        ``(tp, fp, fn)`` under the brief's rule. A true positive is counted for
        each expert beat with any detection satisfying ``abs(expert-detected)
        <= tol_samples``. A false positive is counted for each detection whose
        distance from every expert beat is ``> tol_samples``.
    """

    _validate_tolerance(tol_samples)
    detected = _as_qrs_vector(qrs_detected, "qrs_detected")
    expert = _as_qrs_vector(qrs_expert, "qrs_expert")

    if expert.size == 0:
        return 0, int(detected.size), 0
    if detected.size == 0:
        return 0, 0, int(expert.size)

    expert_matched = _has_neighbor_within_tolerance(
        queries=expert, references=detected, tol_samples=tol_samples
    )
    detection_matched = _has_neighbor_within_tolerance(
        queries=detected, references=expert, tol_samples=tol_samples
    )

    tp = int(np.count_nonzero(expert_matched))
    fn = int(expert.size - tp)
    fp = int(detected.size - np.count_nonzero(detection_matched))
    return tp, fp, fn


def score_record(
    qrs_detected: Iterable[int] | np.ndarray,
    qrs_expert: Iterable[int] | np.ndarray,
    tol_samples: int = DEFAULT_QRS_TOLERANCE_SAMPLES,
) -> dict[str, int | float]:
    """Score one ECG recording's QRS detections.

    Returns a dictionary with ``tp``, ``fp``, ``fn``, ``sens``, ``ppv``, and
    ``f1``. Divide-by-zero cases return ``0.0`` for the affected metric.
    """

    tp, fp, fn = match_qrs(qrs_detected, qrs_expert, tol_samples)
    return _score_from_counts(tp, fp, fn)


def score_dataset(
    qrs_detected_list: Iterable[Iterable[int] | np.ndarray],
    qrs_expert_list: Iterable[Iterable[int] | np.ndarray],
    tol_samples: int = DEFAULT_QRS_TOLERANCE_SAMPLES,
) -> dict[str, list[dict[str, int | float]] | dict[str, int | float]]:
    """Score QRS detections across all recordings.

    Aggregate sensitivity, PPV, and F1 are computed from accumulated TP/FP/FN
    counts across the dataset, matching the brief. They are not the average of
    per-record metrics.
    """

    detected_records = list(qrs_detected_list)
    expert_records = list(qrs_expert_list)
    if len(detected_records) != len(expert_records):
        raise ValueError(
            "qrs_detected_list and qrs_expert_list must contain the same number "
            f"of records, got {len(detected_records)} and {len(expert_records)}"
        )

    rows: list[dict[str, int | float]] = []
    total_tp = total_fp = total_fn = 0
    for record_index, (detected, expert) in enumerate(
        zip(detected_records, expert_records), start=1
    ):
        row = score_record(detected, expert, tol_samples)
        row["recording"] = record_index
        rows.append(row)
        total_tp += int(row["tp"])
        total_fp += int(row["fp"])
        total_fn += int(row["fn"])

    aggregate = _score_from_counts(total_tp, total_fp, total_fn)
    aggregate["n_records"] = len(rows)
    return {"records": rows, "aggregate": aggregate}


def mape(ref: Iterable[float] | np.ndarray, pred: Iterable[float] | np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error in percent.

    The calculation is ``100 * mean(abs(ref - pred) / abs(ref))`` over valid
    pairs only. Valid pairs have finite ``ref`` and ``pred`` values and
    ``ref != 0``. NaN predictions are skipped because later HRV stages are
    allowed to drop invalid windows/recordings; use ``mape_stats`` or
    ``score_hrv_dataset`` when the skipped count matters.

    Returns ``np.nan`` if no valid pairs remain.
    """

    return mape_stats(ref, pred)["mape"]


def mape_stats(
    ref: Iterable[float] | np.ndarray,
    pred: Iterable[float] | np.ndarray,
) -> dict[str, float | int]:
    """Compute MAPE and validity counts for one HRV parameter.

    NaN or infinite predictions are counted as ``n_nan`` and skipped. Zero or
    non-finite reference values are counted separately and skipped. The BMET3997
    reference HRV parameters are not expected to contain zeros, but skipping
    them avoids divide-by-zero artefacts in diagnostics.
    """

    ref_array = _as_float_vector(ref, "ref")
    pred_array = _as_float_vector(pred, "pred")
    if ref_array.size != pred_array.size:
        raise ValueError(
            f"ref and pred must have the same length, got {ref_array.size} "
            f"and {pred_array.size}"
        )

    pred_nan_mask = ~np.isfinite(pred_array)
    ref_invalid_mask = (~np.isfinite(ref_array)) | (ref_array == 0)
    valid_mask = ~(pred_nan_mask | ref_invalid_mask)

    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        mape_value = np.nan
    else:
        errors = np.abs(ref_array[valid_mask] - pred_array[valid_mask]) / np.abs(
            ref_array[valid_mask]
        )
        mape_value = float(np.mean(errors) * 100.0)

    return {
        "mape": mape_value,
        "n_valid": n_valid,
        "n_nan": int(np.count_nonzero(pred_nan_mask)),
        "n_zero_ref": int(np.count_nonzero(ref_array == 0)),
        "n_invalid_ref": int(np.count_nonzero(ref_invalid_mask)),
        "n_total": int(ref_array.size),
    }


def score_hrv_dataset(
    ref_dict: dict[str, Iterable[float] | np.ndarray],
    pred_dict: dict[str, Iterable[float] | np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Score the seven HRV outputs against reference values.

    Returns one nested dictionary per HRV parameter:
    ``{param: {mape, n_valid, n_nan, n_zero_ref, n_invalid_ref, n_total}}``.
    """

    scores: dict[str, dict[str, float | int]] = {}
    for key in HRV_KEYS:
        if key not in ref_dict:
            raise KeyError(f"Missing reference HRV parameter '{key}'")
        if key not in pred_dict:
            raise KeyError(f"Missing predicted HRV parameter '{key}'")
        scores[key] = mape_stats(ref_dict[key], pred_dict[key])
    return scores


def _score_from_counts(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    sens = _safe_divide(tp, tp + fn)
    ppv = _safe_divide(tp, tp + fp)
    f1 = _safe_divide(2.0 * sens * ppv, sens + ppv)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "sens": float(sens),
        "ppv": float(ppv),
        "f1": float(f1),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _validate_tolerance(tol_samples: int) -> None:
    if int(tol_samples) != tol_samples or tol_samples < 0:
        raise ValueError(f"tol_samples must be a non-negative integer, got {tol_samples}")


def _as_qrs_vector(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.asarray([], dtype=np.int64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite sample indices")
    return array.reshape(-1).astype(np.int64)


def _as_float_vector(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _has_neighbor_within_tolerance(
    queries: np.ndarray, references: np.ndarray, tol_samples: int
) -> np.ndarray:
    """Return whether each query has any reference within an inclusive tolerance."""

    sorted_references = np.sort(references)
    insertion_points = np.searchsorted(sorted_references, queries)
    matched = np.zeros(queries.shape, dtype=bool)

    right_mask = insertion_points < sorted_references.size
    if np.any(right_mask):
        matched[right_mask] |= (
            np.abs(sorted_references[insertion_points[right_mask]] - queries[right_mask])
            <= tol_samples
        )

    left_mask = insertion_points > 0
    if np.any(left_mask):
        left_indices = insertion_points[left_mask] - 1
        matched[left_mask] |= (
            np.abs(sorted_references[left_indices] - queries[left_mask]) <= tol_samples
        )

    return matched
