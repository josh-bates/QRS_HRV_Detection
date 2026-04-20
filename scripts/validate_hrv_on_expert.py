#!/usr/bin/env python3
"""Validate Phase 04 HRV features using expert training QRS annotations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import score_hrv_dataset
from src.hrv import hrv_for_recordings
from src.io import HRV_KEYS, load_training_data
from src.reference import REFERENCE_HRV


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _, qrs_expert_list = load_training_data()
    predictions = hrv_for_recordings(qrs_expert_list)
    scores = score_hrv_dataset(REFERENCE_HRV, predictions)

    output_path = REPORTS_DIR / "hrv_validation_on_expert.csv"
    write_report(output_path, scores, predictions)

    print("HRV validation on expert QRS")
    print("parameter,mape_percent,n_valid,n_nan")
    for key in HRV_KEYS:
        row = scores[key]
        print(f"{key},{row['mape']:.4f},{row['n_valid']},{row['n_nan']}")
    print(f"Saved report: {output_path}")


def write_report(
    output_path: Path,
    scores: dict[str, dict[str, float | int]],
    predictions: dict[str, np.ndarray],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "mape_percent",
                "n_valid",
                "n_nan",
                "n_zero_ref",
                "n_invalid_ref",
                "n_total",
                "prediction_mean",
                "reference_mean",
            ],
        )
        writer.writeheader()
        for key in HRV_KEYS:
            score = scores[key]
            writer.writerow(
                {
                    "parameter": key,
                    "mape_percent": f"{score['mape']:.8f}",
                    "n_valid": score["n_valid"],
                    "n_nan": score["n_nan"],
                    "n_zero_ref": score["n_zero_ref"],
                    "n_invalid_ref": score["n_invalid_ref"],
                    "n_total": score["n_total"],
                    "prediction_mean": f"{np.nanmean(predictions[key]):.8f}",
                    "reference_mean": f"{np.nanmean(REFERENCE_HRV[key]):.8f}",
                }
            )


if __name__ == "__main__":
    main()
