#!/usr/bin/env python3
"""Validate a generated BMET3997/9997 submission MAT file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import HRV_KEYS, NUM_RECORDINGS, TEMPLATE_KEYS


QRS_MIN_COUNT = 10_000
QRS_MAX_COUNT = 100_000
MAX_TOTAL_HRV_NANS = 2

HRV_RANGES = {
    "avgRR": (300.0, 2000.0),
    "sdRR": (5.0, 500.0),
    "RMSSD": (5.0, 500.0),
    "pNN50": (0.0, 100.0),
    "LF": (10.0, 50_000.0),
    "HF": (10.0, 50_000.0),
    "LF_HFratio": (0.01, 100.0),
}


def validate_submission_file(path: str | Path) -> list[str]:
    """Return a list of validation failures for a submission MAT file."""

    mat_path = Path(path)
    failures: list[str] = []
    if not mat_path.exists():
        return [f"File does not exist: {mat_path}"]

    try:
        mat = loadmat(mat_path)
    except Exception as exc:  # pragma: no cover - exact scipy exception varies
        return [f"Could not load MAT file: {exc}"]

    public_keys = {key for key in mat if not key.startswith("__")}
    expected_keys = set(TEMPLATE_KEYS)
    if public_keys != expected_keys:
        failures.append(
            f"Variables mismatch: expected {sorted(expected_keys)}, got {sorted(public_keys)}"
        )
    if "ECG" in public_keys:
        failures.append("Submission must not contain ECG")

    qrs = mat.get("QRS")
    if qrs is None:
        failures.append("Missing QRS")
    elif qrs.shape != (1, NUM_RECORDINGS):
        failures.append(f"QRS must have shape (1, {NUM_RECORDINGS}), got {qrs.shape}")
    elif qrs.dtype != object:
        failures.append(f"QRS must be a MATLAB cell/object array, got dtype {qrs.dtype}")
    else:
        for idx in range(NUM_RECORDINGS):
            values = np.asarray(qrs[0, idx]).reshape(-1)
            if not np.issubdtype(values.dtype, np.integer):
                failures.append(f"QRS{{{idx + 1}}} must contain integer samples")
            count = values.size
            if count < QRS_MIN_COUNT or count > QRS_MAX_COUNT:
                failures.append(
                    f"QRS{{{idx + 1}}} has {count} detections; expected "
                    f"{QRS_MIN_COUNT}-{QRS_MAX_COUNT}"
                )
            if count and np.any(values <= 0):
                failures.append(f"QRS{{{idx + 1}}} contains non-positive sample indices")

    total_nan = 0
    for key in HRV_KEYS:
        values = mat.get(key)
        if values is None:
            failures.append(f"Missing {key}")
            continue
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != NUM_RECORDINGS:
            failures.append(f"{key} must contain {NUM_RECORDINGS} values, got {array.size}")
            continue

        nan_count = int(np.count_nonzero(~np.isfinite(array)))
        total_nan += nan_count
        if nan_count:
            failures.append(f"{key} contains {nan_count} NaN/inf value(s)")

        finite = array[np.isfinite(array)]
        if finite.size:
            lower, upper = HRV_RANGES[key]
            out_of_range = (finite < lower) | (finite > upper)
            if np.any(out_of_range):
                bad = finite[out_of_range]
                failures.append(
                    f"{key} has {bad.size} value(s) outside {lower:g}-{upper:g}; "
                    f"min={np.min(finite):.4g}, max={np.max(finite):.4g}"
                )

    if total_nan > MAX_TOTAL_HRV_NANS:
        failures.append(
            f"Submission has {total_nan} total HRV NaN/inf values; "
            f"allowed maximum is {MAX_TOTAL_HRV_NANS}"
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Submission .mat file to validate")
    args = parser.parse_args(argv)

    failures = validate_submission_file(args.path)
    if failures:
        print(f"Submission validation failed: {args.path}")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Submission validation passed: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

