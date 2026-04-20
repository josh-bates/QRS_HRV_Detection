"""Slow Phase 05 integration checks.

Run directly when needed:

    python3 tests/test_pipeline_integration.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_pipeline(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            "src/run_pipeline.py",
            "--mode",
            "train",
            "--out-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Aggregate Sens=" in result.stdout

    qrs_rows = _read_csv(out_dir / "train_qrs_baseline.csv")
    aggregate = qrs_rows[-1]
    f1 = float(aggregate["f1"])
    assert 0.85 <= f1 <= 0.90

    hrv_rows = _read_csv(out_dir / "train_hrv_baseline.csv")
    avg_rr = next(row for row in hrv_rows if row["parameter"] == "avgRR")
    assert float(avg_rr["mape_records_1_20_percent"]) < 5.0


def test_test_pipeline_and_submission_validation(tmp_path: Path) -> None:
    out_dir = tmp_path / "submissions"
    output_path = out_dir / "ProjectTestDataAnalysisGroup0Submission0.mat"
    subprocess.run(
        [
            sys.executable,
            "src/run_pipeline.py",
            "--mode",
            "test",
            "--group-number",
            "0",
            "--submission-number",
            "0",
            "--out-dir",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/validate_submission.py", str(output_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    with __import__("tempfile").TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_train_pipeline(tmp_path)
        test_test_pipeline_and_submission_validation(tmp_path)
    print("Phase 05 pipeline integration tests passed.")

