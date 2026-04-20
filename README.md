# BMET3997 Processing

Python implementation workspace for the BMET3997/9997 major project:
single-lead overnight ECG QRS detection and HRV estimation.

## Assignment Summary

The final system must:

1. Load 35 training ECG recordings and expert QRS annotations from
   `ProjectTrainData.mat`.
2. Load 35 withheld-test ECG recordings from `ProjectTestData.mat`.
3. Detect QRS fiducial sample indices using a custom algorithm. QRS/HRV
   packages such as `neurokit2`, `biosppy`, `heartpy`, `ecgdetectors`,
   `pyhrv`, `hrvanalysis`, and WFDB QRS modules are not permitted.
4. Compute seven HRV outputs per recording from non-overlapping 5-minute
   windows: `avgRR`, `sdRR`, `RMSSD`, `pNN50`, `LF`, `HF`, and
   `LF_HFratio`.
5. Save a MATLAB submission file containing only `QRS` and the seven HRV
   arrays, preserving the template's `1 x 35` orientation and excluding `ECG`.

The full project brief is in
`assignment_brief/BMET3997_MajorProject_Brief.md`.

## Phase 01 Foundation

Implemented foundation files:

- `src/io.py`: data loading, template loading, and submission saving.
- `scripts/explore_data.py`: training-data sanity checks, recording-1 plot, and
  per-recording summary CSV.
- `scripts/check_template_roundtrip.py`: verifies that the submission template
  can be loaded and saved without adding/removing variables or changing values.
- `tests/test_io.py`: pytest coverage for the Phase 01 data contract.
- `requirements.txt`: permitted generic dependencies.

## Phase 02 Evaluation Harness

Implemented scoring files:

- `src/evaluation.py`: QRS matching, per-record and aggregate QRS metrics, HRV
  MAPE, and HRV dataset scoring.
- `tests/test_evaluation.py`: synthetic QRS/HRV tests plus the brief smoke
  check where Recording 1 expert detections are scored against themselves.
- `scripts/check_evaluation.py`: runnable Phase 02 smoke checks without needing
  `pytest`.

The QRS matcher uses an inclusive 50 ms tolerance (`<= 5` samples at 100 Hz),
counts are not one-to-one matched, and aggregate Sensitivity/PPV/F1 are
computed from total TP/FP/FN across records.

## Phase 03 Pan-Tompkins Baseline

Implemented detector files:

- `src/qrs_detector.py`: from-scratch Pan-Tompkins stages and `detect_qrs`.
- `scripts/run_pan_tompkins_training.py`: runs all 35 training recordings,
  scores them, and writes `reports/pt_training_baseline.csv`.
- `tests/test_detector.py`: synthetic no-QRS/QRS-train checks, Recording 1
  acceptance check, and aggregate baseline-range check.

## Phase 04 HRV Pipeline

Implemented HRV files:

- `src/hrv.py`: RR conversion, RR flagging, 5-minute windowing, time-domain
  HRV features, Welch LF/HF estimation, and recording-level aggregation.
- `tests/test_hrv.py`: RR, flagging, time-domain, window-validity, frequency
  sanity, and Recording 1 expert-QRS checks.
- `scripts/validate_hrv_on_expert.py`: compares HRV computed from expert
  training QRS annotations against the reference table from the assignment
  brief and writes `reports/hrv_validation_on_expert.csv`.

## Phase 05 Integration

Implemented end-to-end files:

- `src/run_pipeline.py`: train mode for baseline scoring and test mode for
  submission generation.
- `scripts/validate_submission.py`: strict submission `.mat` format and range
  validator.
- `tests/test_pipeline_integration.py`: slow on-demand integration checks.
- `src/reference.py`: shared expert HRV reference table.

The planned layout expects `.mat` files under `data/`. This checkout currently
has the supplied files under `resources/`, so the loaders support both:

- `data/ProjectTrainData.mat` or `resources/ProjectTrainData.mat`
- `data/ProjectTestData.mat` or `resources/ProjectTestData.mat`
- `data/ProjectTestDataAnalysis.mat` or
  `resources/ProjectTestDataAnalysis.mat`

## Usage

Install dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

Run the Phase 01 exploration checks:

```bash
python3 scripts/explore_data.py
```

Expected outputs:

- `reports/rec1_first14.png`
- `reports/training_data_summary.csv`

Run the submission-template format check:

```bash
python3 scripts/check_template_roundtrip.py
```

Run the Phase 02 scoring smoke checks:

```bash
python3 scripts/check_evaluation.py
python3 tests/test_evaluation.py
```

Run the Phase 03 detector smoke checks:

```bash
python3 tests/test_detector.py
```

Run the full training baseline:

```bash
python3 scripts/run_pan_tompkins_training.py
```

Run the Phase 04 HRV checks:

```bash
python3 tests/test_hrv.py
python3 scripts/validate_hrv_on_expert.py
```

Run the Phase 05 pipeline:

```bash
python3 src/run_pipeline.py --mode train --out-dir reports
python3 src/run_pipeline.py --mode test --group-number 0 --submission-number 0 --out-dir submissions
```

Validate a generated submission:

```bash
python3 scripts/validate_submission.py submissions/ProjectTestDataAnalysisGroup0Submission0.mat
```

If `pytest` is installed, run:

```bash
pytest
```

## Indexing Convention

QRS locations are stored and saved as 1-indexed MATLAB sample positions. When a
QRS position is used to index a NumPy ECG array, subtract one at the point of
use:

```python
ecg[qrs_samples - 1]
```
