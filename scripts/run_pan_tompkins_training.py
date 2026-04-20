#!/usr/bin/env python3
"""Run the Phase 03 Pan-Tompkins baseline on all training recordings."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import score_dataset
from src.io import load_training_data
from src.qrs_detector import detect_qrs


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ecg_list, qrs_expert_list = load_training_data()

    detections = []
    elapsed_by_record = []
    for idx, ecg in enumerate(ecg_list, start=1):
        start = time.perf_counter()
        qrs = detect_qrs(ecg)
        elapsed = time.perf_counter() - start
        detections.append(qrs)
        elapsed_by_record.append(elapsed)
        print(f"Detected recording {idx:02d}: {len(qrs):6d} QRS in {elapsed:6.2f}s")

    scores = score_dataset(detections, qrs_expert_list)
    output_path = REPORTS_DIR / "pt_training_baseline.csv"
    write_report(output_path, scores["records"], scores["aggregate"], elapsed_by_record)

    print()
    print("Per-record scores:")
    for row in scores["records"]:
        print(
            f"Rec {row['recording']:02d}: TP={row['tp']:6d} FP={row['fp']:5d} "
            f"FN={row['fn']:6d} Sens={row['sens']:.4f} "
            f"PPV={row['ppv']:.4f} F1={row['f1']:.4f}"
        )

    agg = scores["aggregate"]
    print()
    print(
        "Aggregate: "
        f"TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} "
        f"Sens={agg['sens']:.4f} PPV={agg['ppv']:.4f} F1={agg['f1']:.4f}"
    )
    print(f"Saved report: {output_path}")


def write_report(
    output_path: Path,
    rows: list[dict[str, int | float]],
    aggregate: dict[str, int | float],
    elapsed_by_record: list[float],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "recording",
                "tp",
                "fp",
                "fn",
                "sens",
                "ppv",
                "f1",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for row, elapsed in zip(rows, elapsed_by_record):
            writer.writerow(
                {
                    "recording": row["recording"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "sens": f"{row['sens']:.8f}",
                    "ppv": f"{row['ppv']:.8f}",
                    "f1": f"{row['f1']:.8f}",
                    "elapsed_seconds": f"{elapsed:.4f}",
                }
            )
        writer.writerow(
            {
                "recording": "aggregate",
                "tp": aggregate["tp"],
                "fp": aggregate["fp"],
                "fn": aggregate["fn"],
                "sens": f"{aggregate['sens']:.8f}",
                "ppv": f"{aggregate['ppv']:.8f}",
                "f1": f"{aggregate['f1']:.8f}",
                "elapsed_seconds": f"{sum(elapsed_by_record):.4f}",
            }
        )


if __name__ == "__main__":
    main()

