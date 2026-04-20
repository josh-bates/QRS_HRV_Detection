# Phase 2 — Evaluation Harness: QRS Scoring + HRV MAPE

## Goal
Build and **unit-test** the exact scoring code that the coordinator will use (or a faithful re-implementation of it). This must exist before the detector is written, because every later phase relies on "score it and see if the number matches the brief." Scoring bugs silently poison every downstream decision.

## Preconditions
- Phase 1 complete. Training data loads cleanly.

## Tasks

### 2.1 Implement `src/evaluation.py`

#### QRS scoring
- `match_qrs(qrs_detected, qrs_expert, tol_samples=12) -> (tp, fp, fn)`
  - Given two 1-D integer arrays of sample indices (1-indexed), count:
    - **TP:** an expert QRS is within ±`tol_samples` of at least one detection.
    - **FN:** an expert QRS is more than `tol_samples` from every detection.
    - **FP:** a detection is more than `tol_samples` from every expert QRS.
  - Tolerance of 12 samples = 120 ms at fs=100 Hz. Keep `tol_samples` configurable but default to 12.
  - **Important:** The brief's matching rule is "within-or-equal-to 120 ms." Use `<= tol` for TP/FN match, `> tol` for FP. No bipartite/one-to-one constraint is specified — a single detection counted as TP can in principle match multiple expert peaks (but with tol=12 at fs=100 Hz this almost never happens in practice; beat-to-beat distance is >>12 samples).

- `score_record(qrs_detected, qrs_expert, tol_samples=12) -> dict`
  - Returns `{tp, fp, fn, sens, ppv, f1}` for a single recording. Define `sens = tp/(tp+fn)`, `ppv = tp/(tp+fp)`, `f1 = 2·sens·ppv/(sens+ppv)`. Handle divide-by-zero gracefully (return 0.0).

- `score_dataset(qrs_detected_list, qrs_expert_list, tol_samples=12) -> dict`
  - Accumulates TP/FP/FN **across all recordings** (per the brief) before computing aggregate Sens/PPV/F1. Return per-record rows in a nested list **and** the aggregate summary.

#### HRV MAPE scoring
- `mape(ref, pred) -> float`
  - `100 * mean(|ref - pred| / |ref|)` over non-NaN pairs. The brief gives the formula as `(100/35) · Σ |...|` i.e. `mean(...) * 100`.
  - Document behaviour if `ref[i] == 0` (shouldn't happen for these parameters but defensive is good — skip or return inf for that element).
  - Document behaviour if `pred[i]` is NaN (a window-dropped recording). Either skip it (tell the caller how many were skipped) or count as 100% error — the brief is silent, but skipping is more defensible because the instrument specifies "ignore windows without enough valid RR" as legitimate behaviour. Go with: **skip NaNs in `pred`, report count of skipped**.

- `score_hrv_dataset(ref_dict, pred_dict) -> dict`
  - Returns `{param: {mape: ..., n_valid: ..., n_nan: ...}}` for each of the 7 HRV parameters.

### 2.2 Unit tests `tests/test_evaluation.py`

#### Test 2.2.1 — Synthetic QRS matching
- Build two tiny arrays:
  - `expert = [100, 200, 300, 400]`
  - `detected = [101, 205, 299, 500]` (three within tol, one far away; nothing near 400)
- Assert with `tol_samples=12`: TP=3, FN=1, FP=1.
- Run with `tol_samples=0`: should collapse to stricter counts.

#### Test 2.2.2 — Reproducing the tutorial Pan-Tompkins numbers on Record 1
This test requires the **Pan-Tompkins detections for Recording 1** as saved intermediate data. There are two options:
- **Option A (preferred):** Defer this test until Phase 3 is done and you have your own PT implementation; then add the assertion that on Recording 1 the tutorial baseline numbers reproduce: Sens ≈ 0.9999, PPV ≈ 0.9996, F1 ≈ 0.9997 (assert within 5e-4).
- **Option B:** Run the MATLAB reference `pan_tompkin.m` on Recording 1 once, export the detection vector to `tests/fixtures/pt_rec1_qrs.npy`, and assert the scorer produces the right F1 on that fixture. This proves the scorer is correct *independently* of your detector.

Do Option B if any MATLAB is available. Otherwise do Option A and flag in the test file that this test is effectively validated in Phase 3.

#### Test 2.2.3 — Synthetic HRV MAPE
- `ref = [100, 200, 400, 800]`, `pred = [110, 180, 420, 760]`.
- Expected MAPE: `mean(|10/100|, |20/200|, |20/400|, |40/800|) * 100 = mean(0.1, 0.1, 0.05, 0.05) * 100 = 7.5`.
- Assert within 1e-6.

#### Test 2.2.4 — NaN handling in MAPE
- `ref = [100, 200]`, `pred = [110, NaN]`.
- Expected: MAPE = 10.0, `n_valid = 1`, `n_nan = 1`.

### 2.3 Reproduce the tutorial scorer numerically (smoke test against brief)
Run the raw brief code (transliterated to Python) on Record 1 using the expert QRS as *both* detection and truth. Should yield Sens=PPV=F1=1.0, TP = len(expert), FP=FN=0. If this doesn't pass, the scorer is broken.

## Deliverables
- `src/evaluation.py` with `match_qrs`, `score_record`, `score_dataset`, `mape`, `score_hrv_dataset`.
- `tests/test_evaluation.py` — all tests passing.
- Optionally: `tests/fixtures/pt_rec1_qrs.npy` if using Option B.

## Acceptance Criteria
- `pytest tests/test_evaluation.py` — all green.
- Passing the expert QRS in as both inputs yields F1=1.0, FP=FN=0. Passing an empty detection list yields F1=0, TP=0, FN=N_expert.
- MAPE on the synthetic case equals 7.5 exactly.

## Gotchas
- **Per-record vs aggregate Sens/PPV/F1:** The brief computes aggregate metrics from accumulated TP/FP/FN totals — **not** as the mean of per-record F1s. Match that exactly. Your per-record F1s are for diagnostic use only.
- **Detection vector may be unsorted:** `match_qrs` must handle unsorted input. Sort internally or use broadcasting — don't assume monotonicity.
- **`np.any(abs(a - b) <= tol)` broadcasting cost:** For vectors with 50k entries on each side, the naive `abs(expert[j] - detected[:, None])` is ~2.5 billion comparisons. Use `numpy.searchsorted` on a sorted detected-array for O(N log M) matching, then check the nearest neighbour's distance. Write the naive version first for correctness; profile + swap only if tests are slow.
- **Integer vs float:** QRS indices are integers. Mixing with floats can cause subtle tolerance bugs. Cast to `np.int64` at the boundary.
- **The scorer is the ground truth of this whole project.** If it's wrong, nothing else matters. Spend the hour on tests here — it will save days later.
