#!/usr/bin/env python3
"""Phase 01 sanity checks and training-data exploration.

This script verifies the exact data-access assumptions from the assignment
brief, writes a plot of recording 1 with expert QRS markers, and produces a
per-recording summary CSV for later detector/debugging work.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Matplotlib may try to write under ~/.config, which is read-only in some
# sandboxes. Set this before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.io import FS, NUM_RECORDINGS, load_training_data


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ecg_list, qrs_expert_list = load_training_data()

    assert len(ecg_list) == NUM_RECORDINGS
    assert len(qrs_expert_list) == NUM_RECORDINGS

    first_100 = ecg_list[0][:100]
    assert isinstance(first_100, np.ndarray)
    assert first_100.dtype == np.float64
    assert first_100.shape == (100,)

    first_three_qrs = qrs_expert_list[0][:3].tolist()
    assert first_three_qrs == [40, 128, 213], first_three_qrs

    rec1_duration_hours = len(ecg_list[0]) / (FS * 3600)
    assert np.isclose(rec1_duration_hours, 8.2139, atol=1e-3), rec1_duration_hours

    write_recording_1_plot(ecg_list[0], qrs_expert_list[0])
    write_training_summary(ecg_list, qrs_expert_list)

    print("Phase 01 data sanity checks passed.")
    print(f"Recording 1 duration: {rec1_duration_hours:.4f} hours")
    print(f"Recording 1 first three expert QRS samples: {first_three_qrs}")
    print(f"Saved plot: {REPORTS_DIR / 'rec1_first14.png'}")
    print(f"Saved summary: {REPORTS_DIR / 'training_data_summary.csv'}")


def write_recording_1_plot(ecg: np.ndarray, qrs_expert: np.ndarray) -> None:
    """Plot the first 12.5 seconds of recording 1 with the first 14 QRS marks."""

    sample_count = 1250
    qrs_markers = qrs_expert[:14]
    x = np.arange(1, sample_count + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, ecg[:sample_count], color="black", linewidth=0.8, label="ECG")
    ax.plot(
        qrs_markers,
        ecg[qrs_markers - 1],
        "go",
        markersize=4,
        label="Expert QRS",
    )
    ax.set_title("Recording 1: first 14 expert QRS detections")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ECG (uV)")
    ax.set_xlim(1, sample_count)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rec1_first14.png", dpi=150)
    plt.close(fig)


def write_training_summary(
    ecg_list: list[np.ndarray], qrs_expert_list: list[np.ndarray]
) -> None:
    """Write a per-recording reference table useful for later debugging."""

    summary_path = REPORTS_DIR / "training_data_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "recording",
                "length_samples",
                "duration_hours",
                "expert_qrs_count",
                "mean_heart_rate_bpm",
                "ecg_min_uV",
                "ecg_max_uV",
                "ecg_mean_uV",
            ],
        )
        writer.writeheader()

        for idx, (ecg, qrs) in enumerate(zip(ecg_list, qrs_expert_list), start=1):
            duration_seconds = len(ecg) / FS
            mean_hr_bpm = 60 * len(qrs) / duration_seconds
            writer.writerow(
                {
                    "recording": idx,
                    "length_samples": len(ecg),
                    "duration_hours": f"{duration_seconds / 3600:.6f}",
                    "expert_qrs_count": len(qrs),
                    "mean_heart_rate_bpm": f"{mean_hr_bpm:.6f}",
                    "ecg_min_uV": f"{float(np.min(ecg)):.6f}",
                    "ecg_max_uV": f"{float(np.max(ecg)):.6f}",
                    "ecg_mean_uV": f"{float(np.mean(ecg)):.6f}",
                }
            )


if __name__ == "__main__":
    main()

