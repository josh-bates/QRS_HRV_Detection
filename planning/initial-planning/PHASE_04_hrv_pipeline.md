# Phase 4 — HRV Pipeline: RR Intervals, Windowing, Seven HRV Parameters

## Goal
Given a list of QRS detection indices for each recording, compute the seven required HRV parameters by:
1. Deriving RR intervals from QRS positions.
2. Flagging physiologically implausible RR intervals.
3. Splitting the recording into non-overlapping 5-minute windows.
4. Dropping windows that don't have ≥4 minutes of unflagged RR intervals.
5. Computing each HRV parameter per window.
6. Averaging each parameter across retained windows to yield a single scalar per recording.

Validate against the expert reference table from the project brief (§10 of the master briefing) on the **training set using the expert QRS annotations as input**. When fed the expert QRS, this pipeline must reproduce the reference HRV values to within small tolerance — this proves the HRV code is correct *independently* of the detector.

## Preconditions
- Phases 1, 2, 3 complete.
- `QRSexpert` loads cleanly and the scoring harness works.

## Tasks

### 4.1 Implement `src/hrv.py`

#### RR derivation
- `rr_from_qrs(qrs_samples: np.ndarray, fs: int = 100) -> np.ndarray`
  - Returns RR intervals **in milliseconds**. `rr[i] = (qrs[i+1] - qrs[i]) / fs * 1000`.
  - Length is `len(qrs) - 1`.
  - Work in ms consistently throughout this module; do not mix s and ms.

#### RR flagging
- `flag_rr(rr_ms: np.ndarray, rr_min_ms: float = 250, rr_max_ms: float = 2000) -> np.ndarray`
  - Returns a boolean mask, `True` = **valid** (unflagged), `False` = flagged (drop).
  - Thresholds from the project brief: RR > 2000 ms or RR < 250 ms are physiologically implausible.

#### Window segmentation
- `segment_windows(qrs_samples: np.ndarray, rr_ms: np.ndarray, rr_valid_mask: np.ndarray, fs: int = 100, window_sec: float = 300) -> list[WindowSpec]`
  - Each `WindowSpec` contains:
    - `start_sample`, `end_sample` (inclusive range of ECG samples covered by the window)
    - `rr_values` (the RR intervals in this window — defined as RR intervals whose **second** QRS falls within the window, or equivalently whose start QRS falls within the window — pick one convention and document it)
    - `rr_valid_mask` (boolean, same length as `rr_values`)
    - `accumulated_valid_time_ms` = sum of valid RR values in the window
    - `is_valid_window` = `accumulated_valid_time_ms >= 240_000` (4 minutes)
  - Windows are non-overlapping, starting at sample 0. The last partial window at the tail of the recording is included if it accumulates ≥4 minutes of valid RR; otherwise discarded.

Document the convention clearly: "An RR interval belongs to window W if its first QRS (i.e. the one at index i in the QRS array, where RR[i] = QRS[i+1] - QRS[i]) falls into window W's sample range." This matches the tutorial's MATLAB snippet `RRseg = RR{1}(QRS{1}>=1 & QRS{1}<=5*100*60);` which indexes RR by QRS start position.

#### Time-domain parameters (per window)
- `avg_rr(rr_valid: np.ndarray) -> float` — `np.mean(rr_valid)` in ms.
- `sd_rr(rr_valid: np.ndarray) -> float` — `np.std(rr_valid, ddof=1)` in ms. (Use sample std with ddof=1; HRV convention.)
- `rmssd(rr_valid: np.ndarray) -> float` — `sqrt( mean( diff(rr_valid)**2 ) )` in ms. Watch: diff is computed on *valid consecutive* intervals only; if a flagged RR sits between two valid ones, the differences that cross the flagged one should be dropped. Simplest rule: only include `diff(rr_valid)` where **both** adjacent RR intervals were valid **and adjacent in the original sequence** (i.e. no flagged RR separated them). Use the full `rr` and `rr_valid_mask` arrays to implement this correctly.
- `pnn50(rr_valid: np.ndarray) -> float` — `100 * count( |diff| > 50 ) / count(diff)`. Same adjacency rule as RMSSD.

Implementation tip for adjacency: build `diffs = rr[1:] - rr[:-1]` and `diffs_valid = rr_valid_mask[1:] & rr_valid_mask[:-1]`. Then use `diffs[diffs_valid]` for RMSSD and pNN50.

#### Frequency-domain parameters (per window)
Bands:
- **LF**: 0.04 – 0.15 Hz
- **HF**: 0.15 – 0.40 Hz

Two common approaches; pick one and stick with it throughout this phase. Recommended: **interpolated Welch**, because it's straightforward in SciPy.

- `lf_hf_welch(rr_times_s: np.ndarray, rr_ms: np.ndarray, rr_valid_mask: np.ndarray, fs_interp: float = 4.0) -> tuple[float, float, float]`
  1. Build a time axis: `t_i = cumulative sum of RR / 1000` (seconds), aligned with the valid RR intervals only.
  2. Interpolate the RR(t) tachogram to a uniform time grid at `fs_interp = 4 Hz`. Use `scipy.interpolate.interp1d` with `kind='cubic'` or `'linear'`. If the window has gaps (runs of flagged RR), document how you bridge them — simplest is to interpolate straight across, accepting some distortion, or to skip segments where >10 s of flagged RR in a row.
  3. **Detrend** (subtract mean at minimum; linear detrend if you want to be more careful) — otherwise DC power leaks into LF.
  4. Welch PSD: `scipy.signal.welch(x, fs=fs_interp, nperseg=min(256, len(x)), detrend='linear')` → returns `(freqs, psd)`. PSD units: ms²/Hz.
  5. Integrate over LF and HF bands using the trapezoidal rule (`np.trapz(psd[mask], freqs[mask])`). Returns `LF`, `HF`, `LF/HF`.

Alternative (Lomb-Scargle, uneven sampling):
- `lf_hf_lomb(rr_times_s: np.ndarray, rr_ms: np.ndarray) -> tuple[float, float, float]` using `scipy.signal.lombscargle` on the detrended RR tachogram at frequencies spanning 0.01–0.5 Hz with fine resolution (e.g. 0.005 Hz). The normalization is subtle — verify against the reference table before trusting the output.

**Validation knob:** after implementing whichever method, tune:
- Interpolation method (linear vs cubic)
- Detrending (none, mean-subtract, linear)
- Welch `nperseg`
- Interpolation rate (2 Hz, 4 Hz)
against the expert reference table (§4.3 below). These choices change LF/HF numerically by 10–30% easily; don't skip validation.

#### Per-window HRV bundle
- `hrv_for_window(window: WindowSpec) -> dict | None`
  - Returns `{avgRR, sdRR, RMSSD, pNN50, LF, HF, LF_HFratio}` if `window.is_valid_window`, else `None`.

#### Per-recording HRV aggregation
- `hrv_for_recording(qrs_samples: np.ndarray, fs: int = 100) -> dict`
  - Runs RR derivation, flagging, windowing.
  - Computes HRV for each valid window.
  - Averages each parameter across valid windows (simple arithmetic mean, ignoring the `None` windows).
  - Returns the 7-key dict, with NaN for any parameter that had no valid windows.

### 4.2 Script `scripts/validate_hrv_on_expert.py`
The key validation. This script feeds the **expert** QRS annotations through the HRV pipeline and compares against the expert reference table (§10 of master briefing). If the HRV code is correct, this should produce very low MAPEs.

Steps:
1. Load training data and expert QRS (Phase 1).
2. For each recording i in 1..35: `hrv_result[i] = hrv_for_recording(qrs_expert_list[i])`.
3. Hard-code the 35 × 7 reference values from the brief's Table in §12 of the original assignment (or §10 of the master briefing). Store as a dict of numpy arrays.
4. Compute MAPE for each parameter using `evaluation.score_hrv_dataset` from Phase 2.
5. Print a table with one row per parameter: reference MAPE, # valid records, # NaN records.

### 4.3 Validation targets (acceptance thresholds)

When fed expert QRS, the MAPEs should be roughly:

| Parameter | Expected MAPE (ideal) | Acceptance ceiling |
|---|---|---|
| avgRR | < 0.5 % | < 2 % |
| sdRR | < 3 % | < 10 % |
| RMSSD | < 3 % | < 10 % |
| pNN50 | < 5 % (absolute terms tricky — small values) | < 20 % |
| LF | < 10 % | < 30 % |
| HF | < 10 % | < 30 % |
| LF_HFratio | < 15 % | < 40 % |

`avgRR` is the strictest because it's basically `mean(diff(QRS))` — any MAPE above 1% means there's a bug in windowing or averaging. Use it as the first assertion.

Frequency-domain targets are looser because the exact spectral method used to build the reference isn't specified in the brief, so a perfect match is impossible. Aim for consistent behaviour — if LF/HF ratios are consistently 50% off in the same direction, there's probably a factor-of-2 bug (e.g. one-sided vs two-sided spectrum).

### 4.4 Unit tests `tests/test_hrv.py`

#### Test 4.4.1 — RR derivation
- `qrs = [100, 200, 350, 500]` at fs=100 Hz → `rr_ms = [1000, 1500, 1500]`.

#### Test 4.4.2 — Flagging
- `rr_ms = [500, 3000, 1000, 100]` with default thresholds → mask `[True, False, True, False]`.

#### Test 4.4.3 — Time-domain parameters on a clean synthetic RR series
- RR = [800, 820, 780, 810, 790] ms.
- `avgRR` = 800.0
- `sdRR` = `np.std([800,820,780,810,790], ddof=1)` ≈ 15.81
- RMSSD = `sqrt(mean([(20)², (-40)², (30)², (-20)²]))` ≈ 28.28
- pNN50 = 0% (no diffs exceed 50 ms)

#### Test 4.4.4 — pNN50 with known crossings
- RR = [800, 870, 810, 750, 820] → diffs = [70, -60, -60, 70] → 4/4 exceed 50 → pNN50 = 100%.

#### Test 4.4.5 — Window validity
- Construct a 7-minute recording (42,000 samples at fs=100). With 4.5 minutes of valid beats and 2.5 minutes of flagged beats, the single window should be marked invalid (since it needs ≥4 min of valid *and* the window is 5 min long — wait: a 5-min window over a 7-min recording leaves a 2-min tail that should be discarded; the first window has the 4.5 valid, which should pass the ≥4-min threshold and be marked valid). Adjust the synthetic data to hit edge cases on both sides of the threshold.

#### Test 4.4.6 — Frequency-domain sanity
- A synthetic RR series that is 1000 ms constant + sinusoid at 0.25 Hz with amplitude 30 ms, 5 minutes long. Expected: HF band dominates, LF ≈ 0, LF/HF ≈ 0 (or very small). Exact power value depends on method; assert HF/LF > 10.

#### Test 4.4.7 — End-to-end on Recording 1 expert QRS
- Run `hrv_for_recording(qrs_expert_list[0])`.
- Assert `abs(result['avgRR'] - 992) / 992 < 0.02` (reference from §10 table).

## Deliverables
- `src/hrv.py`.
- `tests/test_hrv.py` — all passing.
- `scripts/validate_hrv_on_expert.py` — outputs a MAPE table to stdout and saves `reports/hrv_validation_on_expert.csv`.

## Acceptance Criteria
- All unit tests pass.
- `validate_hrv_on_expert.py` produces:
  - `avgRR` MAPE < 2 %
  - `sdRR`, `RMSSD` MAPE < 10 %
  - `pNN50` MAPE < 20 %
  - `LF`, `HF` MAPE < 30 %
  - `LF_HFratio` MAPE < 40 %
- Every recording produces non-NaN values when fed expert QRS (there's enough valid data in each).

## Gotchas
- **Unit consistency.** The brief lists HRV in ms and ms²; the expert table is in ms. Keep RR in ms throughout. If you accidentally mix s and ms, avgRR will be off by 1000× and LF/HF by 10⁶×.
- **Adjacency for diff-based features.** RMSSD and pNN50 must not cross flagged RRs. Naïvely running `np.diff(rr[valid])` on the filtered array *re-glues* RRs that weren't actually consecutive in the recording and produces wrong numbers.
- **The 4-minute rule uses accumulated valid *time*, not count of valid RRs.** Ten RRs of 1000 ms each = 10 s of valid time, not 10 valid things. This matters most on noisy recordings where you might have many short (invalid) RRs.
- **Detrending before Welch is essential.** Without it, the DC component will inflate the LF band by 100× and your LF values will look nothing like the reference.
- **`np.std(..., ddof=1)` (sample) vs `ddof=0` (population).** HRV convention is sample std (ddof=1). Check which one the reference used if your `sdRR` MAPE is consistently ~3-5% off with no other explanation.
- **pNN50 convention.** Some formulations count pairs where `|Δ| > 50`, others `|Δ| ≥ 50`. The brief says "more than 50 ms" — use `>`.
- **LF/HF band edges.** The brief does **not** specify LF/HF band frequencies. The 0.04–0.15 / 0.15–0.40 Hz values used here are the Task Force of ESC/NASPE 1996 standard. If LF/HF MAPEs are stubbornly high after tuning, consider whether the coordinator used different band edges — but start with the standard.
- **BMET9997 causality:** RR computation and HRV calculation are inherently batch operations on completed 5-minute windows. Causality doesn't constrain them per se — only the QRS detector. So Phase 4 is shared between cohorts.
