#!/usr/bin/env python3
"""End-to-end BMET3997/9997 baseline runner."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_submission import validate_submission_file
from src.evaluation import score_dataset, score_hrv_dataset
from src.hrv import hrv_for_recording, hrv_for_recordings
from src.io import (
    FS,
    HRV_KEYS,
    NUM_RECORDINGS,
    load_submission_template,
    load_test_data,
    load_training_data,
    save_submission,
)
from src.qrs_detector import detect_qrs
from src.reference import REFERENCE_HRV


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "train":
        run_train(args)
        return 0
    if args.mode == "test":
        return run_test(args)

    raise ValueError(f"Unknown mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--train-data", default="data/ProjectTrainData.mat")
    parser.add_argument("--test-data", default="data/ProjectTestData.mat")
    parser.add_argument("--template", default="data/ProjectTestDataAnalysis.mat")
    parser.add_argument("--group-number", type=int, default=0)
    parser.add_argument("--submission-number", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def run_train(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ecg_list, qrs_expert_list = load_training_data(args.train_data)
    qrs_detected, hrv_rows, elapsed = process_recordings(ecg_list, label="training")

    qrs_scores = score_dataset(qrs_detected, qrs_expert_list)
    hrv_predictions = _rows_to_hrv_arrays(hrv_rows)
    hrv_scores = score_hrv_dataset(REFERENCE_HRV, hrv_predictions)
    subset_scores = score_hrv_dataset(
        _subset_reference(REFERENCE_HRV, stop=20),
        _subset_predictions(hrv_predictions, stop=20),
    )

    qrs_path = out_dir / "train_qrs_baseline.csv"
    hrv_path = out_dir / "train_hrv_baseline.csv"
    write_qrs_report(qrs_path, qrs_scores["records"], qrs_scores["aggregate"], elapsed)
    write_hrv_report(hrv_path, hrv_scores, subset_scores)
    write_baseline_markdown(out_dir / "BASELINE.md", qrs_scores, hrv_scores, subset_scores)

    print()
    print("=== QRS detection (Pan-Tompkins baseline) ===")
    aggregate = qrs_scores["aggregate"]
    print(
        f"Aggregate Sens={aggregate['sens']:.4f}  "
        f"PPV={aggregate['ppv']:.4f}  F1={aggregate['f1']:.4f}"
    )
    worst = sorted(qrs_scores["records"], key=lambda row: row["f1"])[:5]
    worst_summary = [(row["recording"], round(row["f1"], 4)) for row in worst]
    print(f"Worst 5 recordings (by F1): {worst_summary}")

    print()
    print("=== HRV parameters (from PT detections vs expert reference) ===")
    for key in HRV_KEYS:
        row = hrv_scores[key]
        subset = subset_scores[key]
        print(
            f"{key:<11} MAPE={row['mape']:10.4f}% "
            f"(n_valid={row['n_valid']}, n_nan={row['n_nan']})  "
            f"records1-20={subset['mape']:8.4f}%"
        )

    print(f"Saved QRS report: {qrs_path}")
    print(f"Saved HRV report: {hrv_path}")
    print(f"Saved baseline summary: {out_dir / 'BASELINE.md'}")


def run_test(args: argparse.Namespace) -> int:
    out_dir = args.out_dir or Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)

    ecg_list = load_test_data(args.test_data)
    if len(ecg_list) != NUM_RECORDINGS:
        raise ValueError(f"Expected {NUM_RECORDINGS} test ECGs, got {len(ecg_list)}")

    template = load_submission_template(args.template)
    validate_template_placeholders(template)

    qrs_detected, hrv_rows, _ = process_recordings(ecg_list, label="test")
    hrv_predictions = _rows_to_hrv_arrays(hrv_rows)

    output_path = (
        out_dir
        / f"ProjectTestDataAnalysisGroup{args.group_number}Submission{args.submission_number}.mat"
    )
    save_submission(output_path, qrs_detected, hrv_predictions)

    failures = validate_submission_file(output_path)
    if failures:
        print(f"Generated submission failed validation: {output_path}")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Generated submission: {output_path}")
    print("Submission validation passed.")
    return 0


def process_recordings(
    ecg_list: list[np.ndarray], *, label: str
) -> tuple[list[np.ndarray], list[dict[str, float]], list[float]]:
    qrs_detected: list[np.ndarray] = []
    hrv_rows: list[dict[str, float]] = []
    elapsed: list[float] = []

    for idx, ecg in enumerate(ecg_list, start=1):
        start = time.perf_counter()
        qrs = detect_qrs(ecg, fs=FS)
        hrv = hrv_for_recording(qrs, fs=FS, ecg_len_samples=len(ecg))
        duration = time.perf_counter() - start

        qrs_detected.append(qrs)
        hrv_rows.append(hrv)
        elapsed.append(duration)
        print(
            f"Processed {label} recording {idx:02d}: "
            f"{len(qrs):6d} QRS in {duration:6.2f}s",
            flush=True,
        )

    return qrs_detected, hrv_rows, elapsed


def validate_template_placeholders(template: dict[str, list[np.ndarray] | np.ndarray]) -> None:
    qrs = template["QRS"]
    if not isinstance(qrs, list) or len(qrs) != NUM_RECORDINGS:
        raise ValueError("Template QRS must be a list of 35 placeholder arrays")
    for key in HRV_KEYS:
        values = np.asarray(template[key], dtype=np.float64).reshape(-1)
        if values.size != NUM_RECORDINGS:
            raise ValueError(f"Template {key} must have 35 values")
        if not np.all(np.isnan(values)):
            raise ValueError(f"Template {key} is expected to be all NaN before overwrite")


def write_qrs_report(
    path: Path,
    rows: list[dict[str, int | float]],
    aggregate: dict[str, int | float],
    elapsed: list[float],
) -> None:
    with path.open("w", newline="") as handle:
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
        for row, seconds in zip(rows, elapsed):
            writer.writerow(_format_qrs_row(row, seconds))
        writer.writerow(_format_qrs_row({"recording": "aggregate", **aggregate}, sum(elapsed)))


def write_hrv_report(
    path: Path,
    scores_all: dict[str, dict[str, float | int]],
    scores_first20: dict[str, dict[str, float | int]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "mape_all_percent",
                "n_valid_all",
                "n_nan_all",
                "mape_records_1_20_percent",
                "n_valid_records_1_20",
                "n_nan_records_1_20",
            ],
        )
        writer.writeheader()
        for key in HRV_KEYS:
            all_row = scores_all[key]
            subset_row = scores_first20[key]
            writer.writerow(
                {
                    "parameter": key,
                    "mape_all_percent": f"{all_row['mape']:.8f}",
                    "n_valid_all": all_row["n_valid"],
                    "n_nan_all": all_row["n_nan"],
                    "mape_records_1_20_percent": f"{subset_row['mape']:.8f}",
                    "n_valid_records_1_20": subset_row["n_valid"],
                    "n_nan_records_1_20": subset_row["n_nan"],
                }
            )


def write_baseline_markdown(
    path: Path,
    qrs_scores: dict[str, list[dict[str, int | float]] | dict[str, int | float]],
    hrv_scores: dict[str, dict[str, float | int]],
    subset_scores: dict[str, dict[str, float | int]],
) -> None:
    rows = qrs_scores["records"]
    aggregate = qrs_scores["aggregate"]
    catastrophic = [row for row in rows if row["f1"] < 0.5]
    worst_hrv = sorted(HRV_KEYS, key=lambda key: hrv_scores[key]["mape"], reverse=True)

    lines = [
        "# Baseline Report",
        "",
        "## Pipeline",
        "",
        "Pan-Tompkins QRS detection -> RR interval cleaning -> non-overlapping "
        "5-minute HRV windows -> seven HRV parameters.",
        "",
        "## Training QRS Metrics",
        "",
        f"- Aggregate Sensitivity: {aggregate['sens']:.4f}",
        f"- Aggregate PPV: {aggregate['ppv']:.4f}",
        f"- Aggregate F1: {aggregate['f1']:.4f}",
        "",
        "## Catastrophic QRS Failures",
        "",
    ]
    if catastrophic:
        for row in catastrophic:
            lines.append(
                f"- Recording {row['recording']}: F1={row['f1']:.4f}, "
                f"TP={row['tp']}, FP={row['fp']}, FN={row['fn']}"
            )
    else:
        lines.append("- None under F1 < 0.5.")

    lines.extend(
        [
            "",
            "## HRV Baseline MAPE",
            "",
            "| Parameter | All Records MAPE % | Records 1-20 MAPE % | n_valid | n_nan |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in HRV_KEYS:
        lines.append(
            f"| {key} | {hrv_scores[key]['mape']:.4f} | "
            f"{subset_scores[key]['mape']:.4f} | "
            f"{hrv_scores[key]['n_valid']} | {hrv_scores[key]['n_nan']} |"
        )

    lines.extend(
        [
            "",
            "## Known Weak Points",
            "",
            f"- Worst HRV parameter by all-record MAPE: {worst_hrv[0]} "
            f"({hrv_scores[worst_hrv[0]]['mape']:.4f}%).",
            "- The detector fails badly on a cluster of noisy/degraded recordings, "
            "so HRV errors from Pan-Tompkins detections are dominated by QRS failure, "
            "not the independently validated HRV implementation.",
            "- Subsequent improvement work should target post-detection refinement and "
            "the failure cluster before spending submissions.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _format_qrs_row(row: dict[str, int | float], elapsed_seconds: float) -> dict[str, str | int]:
    return {
        "recording": row["recording"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
        "sens": f"{row['sens']:.8f}",
        "ppv": f"{row['ppv']:.8f}",
        "f1": f"{row['f1']:.8f}",
        "elapsed_seconds": f"{elapsed_seconds:.4f}",
    }


def _rows_to_hrv_arrays(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in HRV_KEYS}


def _subset_reference(
    values: dict[str, np.ndarray], *, start: int = 0, stop: int
) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[start:stop], dtype=np.float64) for key, value in values.items()}


def _subset_predictions(
    values: dict[str, np.ndarray], *, start: int = 0, stop: int
) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[start:stop], dtype=np.float64) for key, value in values.items()}


if __name__ == "__main__":
    raise SystemExit(main())
