# Phase 5 — End-to-End Integration & Submission Generator

## Goal
Tie Phases 1–4 into a single orchestration script that:

1. Processes all 35 **training** recordings end-to-end with Pan-Tompkins and scores them — producing per-record and aggregate QRS metrics plus per-parameter HRV MAPEs against the reference table. This is your baseline report.
2. Processes all 35 **test** recordings end-to-end and writes a correctly formatted `ProjectTestDataAnalysisGroup<N>Submission<M>.mat` file — ready to email to the lab tutor for a scored submission (or, more likely at this stage, ready for internal sweep testing before any scored submission is used).

After Phase 5, you have a working baseline. All subsequent improvement work — better detectors, post-detection refinement, robust HRV — lives on top of this scaffolding and benefits from the same score/MAPE harness.

## Preconditions
- Phases 1, 2, 3, 4 all complete with their acceptance criteria met.
- `ProjectTestData.mat` available (released Week 9).
- `ProjectTestDataAnalysis.mat` template available.

## Tasks

### 5.1 Implement `src/run_pipeline.py`

Two command-line entry points (or one with a `--mode` flag):

#### Mode `train` — baseline evaluation on training data
```
python src/run_pipeline.py --mode train \
    --train-data data/ProjectTrainData.mat \
    --out-dir reports/
```

Steps:
1. Load training data via `io.load_training_data`.
2. For each recording i ∈ 1..35:
   - Run `qrs_detector.detect_qrs(ecg_list[i], fs=100)` → detection indices.
   - Run `hrv.hrv_for_recording(detections, fs=100)` → 7-parameter dict.
   - Store both.
3. Score QRS detections against `qrs_expert_list` using `evaluation.score_dataset`. Save per-record rows and aggregate to `reports/train_qrs_baseline.csv`.
4. Compare HRV outputs to the hard-coded reference table (from brief §10) using `evaluation.score_hrv_dataset`. Save MAPE table to `reports/train_hrv_baseline.csv`.
5. Print a human-readable summary to stdout:
   ```
   === QRS detection (Pan-Tompkins baseline) ===
   Aggregate Sens=0.XXXX  PPV=0.XXXX  F1=0.XXXX
   Worst 5 recordings (by F1): [...]
   
   === HRV parameters (from PT detections vs expert reference) ===
   avgRR      MAPE = XX.X%   (n_valid=XX, n_nan=XX)
   sdRR       MAPE = XX.X%   ...
   ...
   ```

Expected numbers (from the brief):
- Aggregate F1 ≈ 0.87
- avgRR MAPE across all 35 ≈ 65,000 % (!) — driven by recordings 24–35
- avgRR MAPE across recordings 1–20 ≈ 1.98 %

**Also compute and report MAPE restricted to recordings 1–20** — this lets you separate "Pan-Tompkins broke" from "the HRV code is broken." The two subsets should agree on which layer of the pipeline is failing.

#### Mode `test` — generate submission file
```
python src/run_pipeline.py --mode test \
    --test-data data/ProjectTestData.mat \
    --template data/ProjectTestDataAnalysis.mat \
    --group-number 2 \
    --submission-number 1 \
    --out-dir submissions/
```

Steps:
1. Load test ECG via `io.load_test_data`. Assert length 35.
2. Load submission template via `io.load_submission_template`. Assert 7 HRV arrays all-NaN, QRS cell populated with placeholders.
3. For each recording:
   - Run detector → QRS indices (1-indexed).
   - Run HRV pipeline → 7-parameter dict.
4. Overwrite the template's `QRS` cell and HRV arrays with results.
5. Write to `submissions/ProjectTestDataAnalysisGroup<N>Submission<M>.mat` via `io.save_submission`.
6. **Validate the output file before declaring success** (see §5.2).

### 5.2 Submission-file validator `scripts/validate_submission.py`
A script that opens a generated submission .mat and verifies:

1. The file loads cleanly with `scipy.io.loadmat`.
2. Contains exactly these top-level variables (ignoring `__header__`, `__version__`, `__globals__`): `QRS`, `avgRR`, `sdRR`, `RMSSD`, `pNN50`, `LF`, `HF`, `LF_HFratio`. **No `ECG`**.
3. `QRS` is an object array of shape (1, 35) — **not** (35, 1) or (35,). Each entry is a 1-D integer array.
4. Each `QRS[i]` has length in the range [10000, 100000] (generous bounds around the brief's expected 20k–50k).
5. Each of the 7 HRV arrays has length 35 and contains no NaN (allow a small number of NaNs — say, up to 2 — if specific recordings had no valid windows, but flag this loudly to stdout; a working submission should ideally be NaN-free).
6. All HRV values are in plausible ranges:
   - avgRR: 300–2000 ms
   - sdRR: 5–500 ms
   - RMSSD: 5–500 ms
   - pNN50: 0–100 %
   - LF, HF: 10–50000 ms²
   - LF_HFratio: 0.01–100

Run this script at the end of Mode `test` automatically. If any check fails, print the failure and exit non-zero — do not quietly ship a bad submission.

### 5.3 Smoke test `tests/test_pipeline_integration.py`
A slow integration test (marked `@pytest.mark.slow` or similar) that:

1. Runs `run_pipeline.py --mode train` on the full training data.
2. Asserts aggregate F1 is in [0.85, 0.90].
3. Asserts avgRR MAPE restricted to records 1–20 is < 5%.
4. Runs `run_pipeline.py --mode test` with group/submission numbers 0/0 against a test-data fixture or the actual test data if available.
5. Runs `validate_submission.py` on the resulting file and asserts it exits 0.

Keep this test runnable on demand (not in the default `pytest` run) — it's slow but very high-signal for regressions.

### 5.4 Document the baseline in `reports/BASELINE.md`
A short markdown write-up summarising:
- The pipeline (Pan-Tompkins → RR/windows → 7 HRV params).
- Aggregate training metrics actually achieved.
- The list of "catastrophic" recordings where the detector broke, with per-record F1.
- Known-weak HRV parameters (whichever had the worst MAPE on the expert-QRS validation in Phase 4).

This file is the starting point for every subsequent improvement iteration. Future Claude Code runs will read this, pick a failure mode, and address it.

## Deliverables
- `src/run_pipeline.py` with both modes.
- `scripts/validate_submission.py`.
- `tests/test_pipeline_integration.py`.
- `reports/train_qrs_baseline.csv`, `reports/train_hrv_baseline.csv`, `reports/BASELINE.md`.
- `submissions/ProjectTestDataAnalysisGroup<N>Submission<M>.mat` (a **dry-run** submission file — do NOT email it to the tutor as a scored submission; save that for an improved pipeline).

## Acceptance Criteria
- `python src/run_pipeline.py --mode train` runs end-to-end and prints an aggregate F1 in [0.85, 0.90] and avgRR MAPE (records 1–20 only) < 5%.
- `python src/run_pipeline.py --mode test` produces a .mat file that passes `validate_submission.py`.
- The generated .mat file, when loaded back and inspected, has:
  - No `ECG` variable.
  - `QRS` cell array with shape (1, 35), not transposed.
  - All 7 HRV arrays populated (at most 2 NaNs across the board).
- A sceptical reading of `reports/BASELINE.md` makes it clear what's broken and what's working — i.e. the document identifies concrete targets for the next improvement phase.

## Why Stop Here

This is the handoff point to Claude Code for iterative improvement. After Phase 5, you have:

- A scorer you trust (Phase 2, tested).
- A detector that works on clean recordings and fails predictably on noisy ones (Phase 3).
- An HRV pipeline that's independently validated against expert annotations (Phase 4).
- An end-to-end runner that produces a submission file and tells you exactly where the detector is costing you points (Phase 5).

Everything downstream — better filtering, post-detection refinement, second-opinion detectors, cross-validation sweeps, hyperparameter search on the thresholding loop, dealing with the 24–35 failure cluster — is now a matter of swapping out `qrs_detector.detect_qrs` (or inserting a refinement stage after it) and re-running the same harness. The harness doesn't change.

Scored submissions to the tutor should begin only after at least one improvement iteration beyond Pan-Tompkins. Burning submission #1 of 5 on the raw Pan-Tompkins baseline is a waste — you know it will score ≈0.87 F1 with bad HRV MAPEs. Use submissions to validate improvements, not to confirm known baselines.

## Gotchas
- **Submission file shape.** `scipy.io.savemat` with a 1-D Python list wrapped as `np.array(lst, dtype=object)` can produce shape (35,) instead of (1, 35). Force the shape with `np.empty((1, 35), dtype=object); for i, x in enumerate(qrs_list): arr[0, i] = x`. Then verify by loading back and printing `qrs.shape` — it must be `(1, 35)`.
- **Template preservation.** The template contains placeholder QRS values `[100, 200, 300]`. If you accidentally save the template unchanged (because a dict update didn't take), the autograder will read 3 detections per recording and score catastrophically poorly — but the file will look "valid." The validator in §5.2 checks detection counts ≥ 10000 specifically to catch this class of bug.
- **1-indexed integer type.** Ensure `QRS[i]` contains `int64` (or at least integer dtype) values, not `float64`. Some MATLAB scorers are strict about types.
- **Don't email a baseline submission.** Submissions are rate-limited (max 5). Every scored submission should represent a real algorithmic improvement. A baseline validation run is for internal use only.
- **Don't forget records 24–35.** When looking at the training baseline, it's tempting to celebrate the 1–20 numbers. But the test set will include similarly-noisy recordings. Phases beyond 5 should focus first on the failure modes visible in records 24–35.
- **Run `run_pipeline.py --mode train` first, every time**, before the test mode. If the training baseline numbers have drifted, something broke in a dependency upgrade or refactor, and you don't want to discover that via a wasted scored submission.
