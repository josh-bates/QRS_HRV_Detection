# Phase 3 — Pan-Tompkins QRS Detector (From-Scratch Python Implementation)

## Goal
Implement the classic Pan & Tompkins (1985) QRS detection algorithm in Python, from scratch, without using any QRS-detection library. The implementation must reproduce (within a small tolerance) the baseline performance figures the project brief obtained from MATLAB's `pan_tompkin.m`:

- **Recording 1:** Sens ≈ 0.9999, PPV ≈ 0.9996, F1 ≈ 0.9997
- **All 35 training recordings aggregated:** Sens ≈ 0.7818, PPV ≈ 0.9863, F1 ≈ 0.8722

Hitting exactly the MATLAB numbers is not required — small differences in filter design and threshold initialisation are expected. F1 within ±0.01 of the reference on Recording 1 and within ±0.02 on the aggregate is acceptable.

This detector is the baseline. It is deliberately chosen because the brief documents its exact baseline performance, making it the easiest end-to-end sanity check. The final submitted algorithm will evolve beyond it in later work.

## Preconditions
- Phases 1 and 2 complete.
- `evaluation.score_dataset` tested and trusted.

## Tasks

### 3.1 Reference for the algorithm
The algorithm is in the public record: *Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng. 1985; BME-32(3):230–236.* Re-implement from scratch — do **not** import a library that packages it.

### 3.2 Implement `src/qrs_detector.py`

Main entry point:
```
detect_qrs(ecg: np.ndarray, fs: int = 100) -> np.ndarray
```
Returns a 1-D `np.int64` array of **1-indexed** QRS sample positions (to match the `QRSexpert` convention from Phase 1).

The standard pipeline, at fs=100 Hz:

#### Stage A — Band-pass filter (≈5–15 Hz)
Suppresses low-frequency baseline wander, T-wave energy, and high-frequency noise / muscle artefact. Options:
- A cascade of two Butterworth filters: high-pass at 5 Hz (order 1 or 2) then low-pass at 15 Hz (order 1 or 2), implemented via `scipy.signal.butter` + `sosfiltfilt` (bidirectional — BMET3997) or `sosfilt` (causal — BMET9997).
- Or Pan & Tompkins's original cascaded integer-coefficient filters. Butterworth is simpler and fine for this purpose.

At fs=100 Hz the Nyquist is 50 Hz, so 15 Hz is comfortably below it. Note that the original Pan-Tompkins was designed for fs=200 Hz; the fact we're at 100 Hz means the pass-band is relatively wider. Expect slightly degraded behaviour versus the original, but the brief's reference numbers were obtained at 100 Hz too.

#### Stage B — Derivative
Highlight steep slopes (QRS has the steepest slope in a cardiac cycle). Use the 5-point derivative from the paper, rescaled for fs=100 Hz:
```
y[n] = (1/8) * (2 x[n] + x[n-1] - x[n-3] - 2 x[n-4])
```
(Implement as a small FIR via `scipy.signal.lfilter` with b = [2, 1, 0, -1, -2]/8, a = [1].) For BMET3997 this can be run with a zero-phase wrapper; for BMET9997, plain `lfilter`.

#### Stage C — Squaring
Element-wise `y = y**2`. Makes everything positive and emphasises larger derivative values non-linearly. This is the "non-linear transformation" block in the brief's diagram.

#### Stage D — Moving-window integration
Moving average over a window approximately matching QRS width (~150 ms). At fs=100 Hz, window = 15 samples. Implement as `np.convolve(squared, np.ones(15)/15, mode='same')` or `scipy.signal.lfilter(np.ones(15)/15, [1], squared)`. For BMET9997, use `lfilter` (causal); for BMET3997 either works.

#### Stage E — Adaptive thresholding + peak detection (the hard part)
This is where most implementations differ. A compact, faithful version of the Pan-Tompkins logic:

1. Find local maxima of the integrated signal. A fiducial candidate is a local max separated by at least the refractory period (~200 ms = 20 samples at fs=100 Hz) from the previous candidate.
2. Maintain two running estimates:
   - `SPKI` — running estimate of signal peak amplitude on the integrated signal.
   - `NPKI` — running estimate of noise peak amplitude on the integrated signal.
3. Threshold:
   - `THRESHOLD_I1 = NPKI + 0.25 * (SPKI - NPKI)`
   - `THRESHOLD_I2 = 0.5 * THRESHOLD_I1`
4. For each candidate peak value `PEAKI`:
   - If `PEAKI > THRESHOLD_I1`: it's a signal peak. `SPKI = 0.125 * PEAKI + 0.875 * SPKI`. Record the detection.
   - Else: it's a noise peak. `NPKI = 0.125 * PEAKI + 0.875 * NPKI`.
5. **Searchback:** If no QRS is detected for longer than `1.66 * (current RR average)`, go back to the interval and accept the largest peak that exceeded `THRESHOLD_I2` as a QRS; update SPKI with a more aggressive weight (0.25 instead of 0.125). Maintain a running RR average of the last 8 accepted intervals.
6. **Twave discrimination:** If a candidate comes within 360 ms (36 samples) of the previous QRS, check the slope against the previous QRS slope — if slope is less than half, reject as T-wave. Optional for the first pass; include if needed to hit Rec-1 numbers.
7. **Initialisation:** For the first 2 seconds (200 samples), estimate SPKI as the max of the integrated signal in that window, NPKI as the mean. Begin the adaptive loop after that.

Return the detected peak indices. **Compensate for filter group delay** — the peaks detected on the integrated signal are shifted right by approximately half the moving-window width (≈75 ms = 7.5 samples) plus any filter delays. Either shift indices left by the known delay at the end, or use a local-max-finder on the band-passed signal around each integrated peak to snap to the true R-peak location. The latter is more robust and recommended.

### 3.3 Optional second stage — R-peak snapping
After the PT core returns peaks on the filtered+integrated signal, for each detection:
1. Search a ±100 ms window on the original (or band-passed) ECG.
2. Snap the detection to the local maximum absolute amplitude within that window.

This consistently improves Sens and PPV by a couple of tenths of a percent at little cost.

### 3.4 Smoke-test script `scripts/run_pan_tompkins_training.py`
- Load training data (Phase 1).
- Run `detect_qrs` on all 35 recordings.
- Score with `evaluation.score_dataset` (Phase 2) against `QRSexpert`.
- Print per-record TP/FP/FN/Sens/PPV/F1 and the aggregate.
- Save results to `reports/pt_training_baseline.csv`.

### 3.5 Unit tests `tests/test_detector.py`

#### Test 3.5.1 — Synthetic sinusoid
A pure 1 Hz sinusoid sampled at 100 Hz over 30 seconds with amplitude 1 mV shouldn't trigger many QRS detections (no sharp slopes). Assert the number of detections is < 5% of the expected heart-rate count (just a sanity check, not a precise test).

#### Test 3.5.2 — Synthetic QRS train
Generate a synthetic "ECG": a train of raised cosines (or triangle peaks) at 1 Hz with added 0.01 mV Gaussian noise, 100 seconds long. Expect ~100 detections, F1 > 0.95 against the known peak positions.

#### Test 3.5.3 — Recording 1 of training data
Run on `ecg_list[0]`, score against `qrs_expert_list[0]`. Assert:
- F1 ≥ 0.99
- Sens ≥ 0.99
- PPV ≥ 0.99

If any of these fail, the detector is broken — do not proceed to Phase 4.

#### Test 3.5.4 — Aggregate on all 35 training recordings
Assert aggregate F1 ≥ 0.85 across all 35 recordings (a forgiving lower bound around the reference 0.8722). If you exceed 0.90, something is probably wrong — the reference really is that bad because late recordings break catastrophically, and your implementation shouldn't be magically robust.

## Deliverables
- `src/qrs_detector.py` with `detect_qrs`, and the component functions (filter stage, derivative, square, integrate, threshold loop) as separate callable helpers for later reuse.
- `tests/test_detector.py` — all passing.
- `scripts/run_pan_tompkins_training.py` — produces `reports/pt_training_baseline.csv`.
- A short section in the module docstring listing which recordings the detector failed catastrophically on (expect roughly: 24, 25, 26, 27, 29, 30, 31, 34, 35 based on the brief's `avgRR` table) — useful context for Phase 4 and future improvement.

## Acceptance Criteria
- All tests pass.
- `scripts/run_pan_tompkins_training.py` runs to completion in < 10 minutes on a normal laptop.
- `reports/pt_training_baseline.csv` shows aggregate F1 between **0.85 and 0.90** (not higher — if it's higher you're probably over-fitting the threshold logic to the clean records; the noisy records are *supposed* to fail here).

## Gotchas
- **fs=100 Hz, not 200 Hz:** Every magic number from the original Pan-Tompkins paper that's expressed in samples needs re-scaling. Constants in seconds/ms are portable; constants in samples are not. Always compute `int(0.15 * fs)` rather than hard-coding 30.
- **Group delay compensation:** Easy to forget. Symptom: your detections are consistently ~5–10 samples late, so records with tight tolerance look worse than they should. R-peak snapping (§3.3) fixes this.
- **Refractory period of 200 ms:** With fs=100 Hz, that's exactly 20 samples. Without it, the integrated signal's broad plateau produces multiple peaks per QRS → inflated FP count.
- **Initial threshold learning:** If you initialise SPKI=NPKI=0, the first few seconds will pick up tons of FPs. Warm-start from a 2-second calibration window.
- **Causality for BMET9997:** Use `sosfilt` (causal forward filter) not `sosfiltfilt` (zero-phase bidirectional). Searchback beyond the current sample is not allowed — only searchback within already-observed samples. This is fine because the searchback look *backwards* in time; just make sure no lookahead is used anywhere.
- **Don't use `scipy.signal.find_peaks`'s advanced options to over-tune this.** Pan-Tompkins is defined by its two-threshold adaptive loop; using `find_peaks` for local-max enumeration is fine but the decision logic is yours.
- **Don't spend >2 days chasing Rec-1 F1=0.9997 exactly.** The point of this phase is a working baseline, not a perfect replica of the MATLAB code. If you're at 0.995 on Rec 1 and 0.86 aggregate, move on.
