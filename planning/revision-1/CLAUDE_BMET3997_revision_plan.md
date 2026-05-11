# BMET3997/9997 Causal QRS + HRV Revision Plan for Claude

## Goal

Upgrade the current codebase to a stronger **fully causal** ECG pipeline that materially improves:

- aggregate QRS **Sensitivity / PPV / F1**
- QRS **timing accuracy at ±50 ms**
- HRV MAPE, especially **RMSSD, pNN50, LF, HF, LF/HF**

Do **not** rewrite the whole project. Keep the current structure and submission flow, but replace the weak parts with a better causal detector + refinement layer and a more accurate HRV implementation.

This plan is optimized for the current codebase, not for a greenfield project.

---

## Non-negotiable constraints

1. **Causal only**
   - No `filtfilt`
   - No reverse-time passes
   - No use of future samples for QRS decisions
   - Small fixed delays are acceptable if they come from trailing windows or one-sample peak confirmation

2. **No prohibited libraries**
   - No QRS detection libraries
   - No HRV packages
   - General NumPy / SciPy / pandas / matplotlib are fine

3. **Preserve project compatibility**
   - Keep 1-indexed MATLAB QRS outputs
   - Keep `.mat` input/output compatibility
   - Keep the submission writer and validator working
   - Keep `run_pipeline.py` usable with train/test modes

4. **Internal metric target**
   - Optimize QRS under **±5 samples = ±50 ms**
   - Do not relax the scorer to make metrics look better

---

## Important context from the current code

The current code already has a reasonable baseline:
- causal Pan-Tompkins-style filtering, derivative, squaring, trailing integration
- adaptive thresholds
- searchback / T-wave logic
- R-peak snapping
- windowed HRV calculation
- train/test runner and submission writer

However, it has several issues that are likely blocking major gains:

### A. Current detector is not strictly causal
`qrs_detector.py` still uses whole-record statistics:
- subtracts the **whole-record mean**
- normalizes by **whole-record max abs**
- derivative is also normalized by **whole-record max abs**

This must be removed.

### B. Timing refinement is too weak for a ±50 ms target
The current candidate stream is based heavily on the integrated signal, which is broad. The current snap-to-peak step uses the **largest absolute extremum in a backward window**, which can land on the wrong deflection or noise.

### C. One proposal stream is doing too much
The current integrated-domain candidate generation is carrying detection, rejection, and timing all at once. That is fragile on noisy records.

### D. HRV error is not just a QRS problem
The current MAPE pattern strongly suggests that some HRV error is caused by the **HRV estimator itself**, not only beat errors:
- `avgRR` is already very good on records 1–20
- but `RMSSD`, `pNN50`, `LF`, `HF`, and `LF/HF` remain much worse

This means the next revision must improve both:
1. beat detection / timing
2. HRV estimation details

---

## High-level strategy

Implement a **hybrid causal pipeline** with three priorities:

### Priority 1 — Fix HRV calibration against expert QRS first
Before changing the detector heavily, validate and improve `hrv.py` using the **expert training QRS**.
This is critical because there is no point improving QRS if the HRV implementation itself is mismatched to the reference table.

### Priority 2 — Convert the detector into proposal + refinement
Use a **high-sensitivity Stage A** to propose candidate beats, then a **Stage B causal refinement layer** to:
- accept or reject candidates
- retime accepted beats more accurately
- recover likely missed beats in long gaps

### Priority 3 — Add signal-quality-aware logic
Noisy segments should not be treated like clean segments.
Use simple causal SQI-style features to tighten acceptance in bad segments and prevent false positives from poisoning RR and HRV.

---

## What Claude should implement

## Phase 1 — Fix `hrv.py` before detector changes

### Objective
Make sure the HRV implementation itself is aligned with the assignment reference values when using expert QRS.

### Required work

1. Create a script:
   - `scripts/validate_hrv_variants.py`

2. This script must compute the seven HRV outputs using **expert QRS** and compare against the provided training reference table.

3. Implement and compare at least these spectral variants:
   - current Lomb–Scargle approach
   - Lomb–Scargle using **better RR timing placement**:
     - interval start time
     - interval midpoint time
     - cumulative beat time / beat-centered timing
   - interpolated tachogram at **4 Hz** followed by **Welch PSD**

4. Keep the same assignment-required bands:
   - LF: 0.04–0.15 Hz
   - HF: 0.15–0.40 Hz

5. Pick the spectral method that gives the **lowest training-set MAPE on expert QRS**, especially for:
   - LF
   - HF
   - LF/HF

6. Keep the existing valid-RR logic unless a change is clearly justified:
   - valid RR in [250, 2000] ms
   - drop windows with < 4 min valid RR

7. Verify the time-domain metrics too:
   - `avgRR`
   - `sdRR`
   - `RMSSD`
   - `pNN50`

### Notes
- Do not use any HRV package
- Implement Lomb–Scargle / interpolation / Welch manually with SciPy primitives only
- This phase must happen **before** heavy detector refactoring

### Expected result
A more faithful `hrv.py` that reduces HRV MAPE **even when using perfect expert QRS**.

---

## Phase 2 — Make QRS preprocessing strictly causal

### Objective
Remove all future leakage and avoid morphology-distorting normalization.

### Required changes in `src/qrs_detector.py`

1. **Remove whole-record demeaning**
   - Do not subtract `np.mean(ecg_array)`
   - Prefer one of:
     - no explicit demeaning
     - slow causal baseline tracker (EMA baseline removal)

2. **Remove whole-record normalization**
   - Do not divide the filtered or derivative signals by global max abs
   - Do not normalize every sample by a running envelope unless clearly needed

3. **Use running signal/noise levels only for thresholds**
   - Preserve amplitude information for morphology and timing refinement
   - Track adaptive scale with causal envelopes, but use them in thresholding / scoring, not to warp the signal itself

4. **Document the effective decision delay**
   - derivative stage
   - trailing integration
   - one-sample local-max confirmation
   - any retiming delay

### Recommended implementation choice
Use:
- optional slow EMA baseline removal
- causal band-pass filtering
- no global normalization
- adaptive thresholds based on running signal/noise estimates

---

## Phase 3 — Replace single-stream detection with dual causal proposals

### Objective
Improve sensitivity on hard recordings without exploding false positives.

### Required design

Create a candidate proposal structure that stores, per candidate:
- proposal time
- raw ECG local peak index and amplitude
- bandpassed local peak index and amplitude
- integrated peak value
- local slope / derivative energy
- local width estimate
- recent RR context
- local signal-quality features

### Stage A should produce candidates from **two lightweight causal streams**

#### Stream 1 — upgraded Pan-Tompkins energy stream
Keep the current Pan-Tompkins backbone, but tune it for **high sensitivity**, not final precision.

#### Stream 2 — bandpass/slope-based second opinion
Add a second causal proposal stream built from:
- local maxima in bandpassed ECG
- gated by slope / derivative / short-window energy
- its own adaptive threshold
- refractory handling

This second stream should have a different failure mode from the integrated stream.

### Proposal fusion
- merge nearby candidates from both streams if they are within a small tolerance
- preserve the strongest timing candidate
- keep enough metadata so Stage B can know which stream proposed the beat

### Important
Do **not** add a heavy ML model here.
Keep the detector deterministic, causal, and debuggable.

---

## Phase 4 — Add a real Stage B refinement layer

### Objective
Turn candidate events into better final QRS detections.

Create:
- `src/qrs_refinement.py`
- `src/qrs_features.py`

### Stage B must process candidates sequentially in time and decide:
- accept
- reject
- retime
- recover a likely missed beat in a recent long gap

### Required candidate features

For each candidate, compute only **causal** features from current/past samples.

#### Morphology / amplitude features
- raw peak amplitude
- bandpassed peak amplitude
- local peak prominence
- absolute slope / derivative energy
- integrated energy
- QRS-width proxy
- amplitude ratio to recent accepted beats
- polarity consistency with recent accepted beats

#### RR / rhythm features
- time since previous accepted beat
- ratio to recent median RR
- ratio to recent regular RR mean
- whether the candidate splits an implausibly long gap into two plausible intervals
- refractory violation severity

#### Signal-quality features
- local QRS-band energy vs low-frequency energy
- local baseline wander estimate
- template correlation with recent accepted beats
- local noise floor estimate
- recent false-positive density

### Stage B rules

Implement simple sequential rules, not a black-box classifier:

1. **False-positive rejection**
   Reject candidates that are weak, noisy, inconsistent with recent morphology, and implausible in RR context.

2. **Missed-beat recovery**
   When a gap is much larger than recent RR, search causally inside the past gap for the best missed-beat candidate using the stored proposal history.

3. **Template-guided acceptance**
   Maintain a small rolling QRS template from recent high-confidence accepted beats and use normalized correlation as one feature.

4. **Signal-quality-aware thresholding**
   Tighten acceptance when local quality is poor; be more permissive when local quality is high.

5. **Hard cap on refractory violations**
   Avoid duplicate detections around the same beat.

---

## Phase 5 — Replace the current snap-to-peak logic with polarity-aware timing refinement

### Objective
Improve timing enough to materially help the ±50 ms scorer and RR-derived HRV.

### Why
The current code snaps to the **largest absolute extremum** in a backward window, which can choose the wrong wave component.

### New timing refinement

Implement a dedicated `refine_r_peak_time()` step that:

1. Estimates the dominant QRS polarity from recent high-confidence accepted beats:
   - mostly positive
   - mostly negative
   - mixed / uncertain

2. Searches only within a **short causal backward window** around the candidate

3. Chooses the best local peak using a fused score:
   - polarity-consistent raw amplitude
   - bandpassed amplitude
   - local slope
   - local prominence
   - closeness to proposal center

4. Rejects obviously late/early shifts that are inconsistent with recent timing

5. Returns the final 1-indexed MATLAB QRS sample

### Important
Keep this retiming step causal.
No future samples may be used.

---

## Phase 6 — Add signal-quality-aware HRV protection

### Objective
Prevent a few bad detections from disproportionately damaging `RMSSD`, `pNN50`, `LF`, `HF`, and `LF/HF`.

### Required logic

1. Track per-beat confidence:
   - high confidence
   - medium confidence
   - low confidence

2. Track per-window quality:
   - fraction of beats that were corrected or recovered
   - fraction of low-confidence beats
   - local SQI summary
   - amount of valid RR time remaining

3. For HRV windows:
   - continue to obey the assignment-validity rule
   - additionally suppress windows that are dominated by unstable / low-confidence beats if this improves training-set agreement
   - do **not** over-prune windows aggressively

### Goal
Especially reduce MAPE for:
- `RMSSD`
- `pNN50`
- `LF`
- `HF`
- `LF/HFratio`

---

## Phase 7 — Build the right diagnostics and ablations

### Objective
Make tuning evidence-based instead of trial-and-error.

### Create

- `src/qrs_debug.py`
- `scripts/run_revision_training.py`
- `scripts/ablation_revision.py`

### Training script outputs must include

1. Aggregate QRS:
   - TP / FP / FN
   - Sens / PPV / F1

2. Per-record QRS summary:
   - TP / FP / FN
   - Sens / PPV / F1
   - number of recovered beats
   - number of rejected candidates

3. Timing diagnostics:
   - distribution of matched detection timing errors
   - median absolute timing error
   - counts of early vs late detections
   - proportion within ±1, ±2, ±3, ±5 samples

4. HRV diagnostics:
   - MAPE for all seven outputs
   - full 35-record MAPE
   - records 1–20 MAPE
   - expert-QRS HRV validation summary

5. Hard-case ranking:
   - worst recordings by F1
   - worst recordings by LF/HF error
   - worst recordings by RMSSD / pNN50 error

### Required ablations
At minimum compare:
- current baseline
- causal-cleaned baseline
- + dual proposals
- + refinement
- + polarity-aware retiming
- + HRV spectral update
- + SQI gating

---

## Files Claude should add or modify

### Modify
- `src/qrs_detector.py`
- `src/hrv.py`
- `scripts/run_pipeline.py` (only if needed to call the upgraded pipeline)

### Add
- `src/qrs_refinement.py`
- `src/qrs_features.py`
- `src/qrs_debug.py`
- `scripts/validate_hrv_variants.py`
- `scripts/run_revision_training.py`
- `scripts/ablation_revision.py`

Optional:
- `scripts/tune_revision.py`

---

## Recommended implementation order

1. Validate and improve `hrv.py` using expert QRS
2. Remove non-causal global statistics from `qrs_detector.py`
3. Add the second causal proposal stream
4. Add Stage B refinement
5. Add polarity-aware timing refinement
6. Add SQI / confidence logic
7. Run ablations and keep only changes that improve training metrics

---

## Acceptance criteria

The revision is successful only if it does all of the following:

1. Stays fully causal
2. Uses no prohibited QRS / HRV libraries
3. Preserves valid submission output format
4. Improves aggregate QRS F1 over the current baseline
5. Improves timing accuracy under the **±50 ms** tolerance
6. Improves HRV MAPE, especially:
   - `RMSSD`
   - `pNN50`
   - `LF`
   - `HF`
   - `LF_HFratio`
7. Produces clear training diagnostics showing which components helped

---

## Explicit do-not-do list

- Do not use `filtfilt`
- Do not use reverse-time cleanup
- Do not use future lookahead for beat decisions
- Do not keep whole-record mean/max normalization
- Do not rely on a single integrated-signal threshold stream
- Do not use a heavy neural network or external pretrained detector
- Do not rewrite the project I/O or submission format unnecessarily
- Do not optimize only record 1 or only clean recordings
- Do not change the scorer tolerance away from ±5 samples

---

## Short implementation summary

Upgrade the current code into a **causal hybrid detector + calibrated HRV pipeline**:
1. first fix HRV spectral estimation against expert QRS,
2. then replace the current single-stream Pan-Tompkins detector with a dual-stream causal proposal system,
3. add a sequential refinement layer for false-positive rejection, missed-beat recovery, and polarity-aware retiming,
4. and protect HRV from noisy/low-confidence segments using simple SQI-aware window logic.

This is the best trade-off between accuracy, causality, assignment compliance, and implementation complexity for the current codebase.
