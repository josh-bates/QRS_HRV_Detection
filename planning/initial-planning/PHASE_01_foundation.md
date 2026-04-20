# Phase 1 — Foundation: Environment, Data Loading, Exploration

## Goal
Stand up a Python project that can load `ProjectTrainData.mat`, expose the 35 ECG recordings and their expert QRS annotations as clean Python objects, and reproduce the trivial sanity checks from the project brief (first 100 ECG samples of recording 1, recording-1 duration ≈ 8.2139 hours, first three QRS peaks of recording 1 at samples 40, 128, 213).

No algorithm work happens in this phase. This exists so every later phase starts from a known-good data-access layer.

## Preconditions
- `ProjectTrainData.mat` is available at `data/ProjectTrainData.mat`.
- Python 3.10+ available.

## Tasks

### 1.1 Environment
- Create `requirements.txt` with at minimum: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`.
- No HRV/QRS libraries — these are forbidden by the project rules. If a package's name suggests it handles ECG, HRV, heartbeats, or RR intervals, do not add it.

### 1.2 Implement `src/io.py`
Public functions:

- `load_training_data(path="data/ProjectTrainData.mat") -> (ecg_list, qrs_expert_list)`
  - Loads the .mat file with `scipy.io.loadmat`.
  - `ECG` arrives as an `object` ndarray of shape `(1, 35)` or `(35, 1)`; unwrap into a plain Python list of 1-D `np.float64` arrays, length 35.
  - `QRSexpert` arrives the same way; unwrap into a list of 1-D `np.int64` arrays, length 35. These are **1-indexed MATLAB sample positions** — keep them 1-indexed for now; convert only at the boundary where you index into NumPy arrays (subtract 1 at the point of use). Document this choice explicitly in the module docstring.

- `load_test_data(path="data/ProjectTestData.mat") -> ecg_list`
  - Returns a list of 35 ECG arrays. No QRS ground truth in test data.

- `load_submission_template(path="data/ProjectTestDataAnalysis.mat") -> dict`
  - Returns a dict with keys: `QRS` (list of 35 placeholder arrays), `avgRR`, `sdRR`, `RMSSD`, `pNN50`, `LF`, `HF`, `LF_HFratio` (each a 1-D array of length 35, initially NaN).

- `save_submission(path, qrs_list, hrv_dict)`
  - Writes a .mat file containing the `QRS` cell array and the 7 HRV arrays **only** — no `ECG`, no extra variables.
  - `QRS` must be saved as a MATLAB cell array **with the same orientation as the template** (do not transpose). Use `scipy.io.savemat` with `qrs_list` wrapped as an `object` ndarray of shape `(1, 35)` to match how MATLAB `cell(1,35)` round-trips.
  - HRV arrays should be plain `float64` numpy arrays of shape `(1, 35)` or `(35,)` — mirror the template's shape.

Constants to centralise:
- `FS = 100` (Hz)
- `ECG_UNIT = "uV"` (µV; 1 LSB = 5 µV per the brief, but ECG is already in µV so this is informational)
- `NUM_RECORDINGS = 35`

### 1.3 Sanity-check script `scripts/explore_data.py`
A throwaway script that prints/plots and asserts the following, exiting non-zero if anything mismatches:

1. `len(ecg_list) == 35` and `len(qrs_expert_list) == 35`.
2. `ecg_list[0][:100]` runs without error and is a float vector of length 100.
3. `qrs_expert_list[0][:3].tolist() == [40, 128, 213]`.
4. `len(ecg_list[0]) / (100 * 3600)` is approximately `8.2139` (assert within 1e-3).
5. A plot of `ECG[0][0:1250]` with the first 14 expert QRS peaks overlaid is saved to `reports/rec1_first14.png`. Visually, each green marker should sit on a QRS peak in the first ~12 seconds of data.

### 1.4 Quick per-recording stats (optional but useful)
Produce `reports/training_data_summary.csv` with one row per recording:
- recording index, length in samples, duration in hours, number of expert QRS peaks, mean heart rate (= `60 * n_qrs / duration_seconds`), min/max/mean ECG amplitude.

This CSV becomes a useful reference when diagnosing why later recordings (24–35) break the baseline.

## Deliverables
- `src/io.py` with the functions above, each with a docstring.
- `scripts/explore_data.py` that runs to completion and saves one PNG + one CSV.
- `requirements.txt`.

## Acceptance Criteria
- `python scripts/explore_data.py` exits 0.
- All four numerical assertions in §1.3 pass.
- The PNG visually matches the brief (QRS markers sit on peaks in ECG[0][0:1250]).
- `io.save_submission` can round-trip the **unmodified template** (read it, write it, read it again) without any value changes and without adding or removing variables. Verify with a one-off test.

## Gotchas
- **1-indexed vs 0-indexed:** MATLAB is 1-indexed. `QRSexpert{1}(1) == 40` means the 40th MATLAB sample, which is `ecg[39]` in Python. Pick one convention (recommend: keep QRS indices 1-indexed in Python to minimise mental translation when comparing to the brief, subtract 1 only when slicing NumPy arrays). Document this in `io.py`.
- **`scipy.io.loadmat` quirks:** nested cell arrays come back as nested `object` ndarrays; strings come back as `numpy.str_`. You will need to flatten with something like `[np.asarray(x).ravel() for x in raw["ECG"].flatten()]`.
- **Do not** import or add any package whose purpose is QRS detection or HRV analysis. If `requirements.txt` grows beyond the list in §1.1, pause and check.
- **Do not** transpose `QRS` in the save path. The autograder keys off exact shape.
- **Do not** include the `ECG` variable in anything written to `submissions/`.
