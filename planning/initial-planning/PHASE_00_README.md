# Phased Plan — Pan-Tompkins Baseline for BMET3997/9997 Major Project

This folder contains **five sequential planning documents** that guide construction of a complete, submittable baseline for the QRS-detection + HRV-estimation project. The target end state is:

> A Python implementation that, given `ProjectTrainData.mat` and `ProjectTestData.mat`, runs a from-scratch **Pan-Tompkins** QRS detector, computes the seven HRV parameters in non-overlapping 5-minute windows, validates performance against the expert reference table on the training set, and emits a correctly formatted `ProjectTestDataAnalysisGroup<N>Submission<M>.mat` file ready for submission.

Once this baseline is running and producing scores, subsequent improvement passes (post-detection refinement, better detectors, robust HRV) will happen iteratively in Claude Code.

## Scope Rules Observed Throughout

- **No QRS or HRV library** (`neurokit2`, `biosppy`, `heartpy`, `ecgdetectors`, `pyhrv`, `hrvanalysis`, WFDB QRS modules — all forbidden). Pan-Tompkins is re-implemented from scratch; it's an algorithm, not a library.
- **SciPy / NumPy / matplotlib / pandas / scipy.io are fine** for generic DSP, I/O, and plotting.
- **Causality:** default to bidirectional processing (BMET3997). If implementing for BMET9997, see the causality notes inside Phase 3 and Phase 4.

## Phase Index

| # | Phase | Produces | Approx. effort |
|---|---|---|---|
| 1 | **Foundation** — env setup, data loading, exploration | `io.py`, loaded sanity-check notebook | Low |
| 2 | **Evaluation Harness** — QRS scoring + HRV MAPE | `evaluation.py`, unit-tested on Rec 1 | Medium |
| 3 | **Pan-Tompkins Detector** — from-scratch Python implementation | `qrs_detector.py` reproducing tutorial baselines | High |
| 4 | **HRV Pipeline** — RR, windowing, 7 HRV parameters | `hrv.py` matching reference table to within tolerance | High |
| 5 | **Integration & Submission Generator** — end-to-end on train + test | `run_pipeline.py`, `ProjectTestDataAnalysisGroup<N>Submission<M>.mat` | Medium |

Each phase document has: **Goal → Preconditions → Tasks → Deliverables → Acceptance Criteria → Gotchas**. Execute them in order. Do **not** start a later phase before the previous phase's acceptance criteria are met — each phase's tests are what lets the next phase trust its inputs.

## Target Baseline Numbers (from project brief)

After Phase 5 finishes, running the baseline on **training** data should reproduce numbers close to:

- Pan-Tompkins on Recording 1 alone: **Sens ≈ 0.9999, PPV ≈ 0.9996, F1 ≈ 0.9997**
- Pan-Tompkins on all 35 training recordings: **Sens ≈ 0.7818, PPV ≈ 0.9863, F1 ≈ 0.8722**
- `avgRR` MAPE across all 35: **≈ 65,962 %** (catastrophic — driven by records 24–35 where the detector breaks)
- `avgRR` MAPE across records 1–20 only: **≈ 1.98 %**

Reproducing these numbers (±small tolerance) is the proof that the harness and baseline are wired correctly. The horrible aggregate MAPE is the **point** of the baseline — it defines the problem the subsequent Claude Code iterations will solve.

## Suggested Repository Layout

```
project/
├── data/
│   ├── ProjectTrainData.mat         # provided
│   ├── ProjectTestData.mat          # provided (Week 9)
│   └── ProjectTestDataAnalysis.mat  # provided template
├── src/
│   ├── io.py              # Phase 1
│   ├── evaluation.py      # Phase 2
│   ├── qrs_detector.py    # Phase 3  (Pan-Tompkins)
│   ├── hrv.py             # Phase 4
│   └── run_pipeline.py    # Phase 5
├── tests/
│   ├── test_evaluation.py
│   ├── test_detector.py
│   └── test_hrv.py
├── submissions/
│   └── ProjectTestDataAnalysisGroup<N>Submission<M>.mat
├── reports/
│   └── train_baseline_report.csv    # per-record metrics from Phase 5
├── requirements.txt
└── README.md
```

Claude Code should treat this layout as a suggestion, not a mandate — but the names referenced in later phases (`io.load_training_data`, `evaluation.score_qrs`, etc.) assume this structure.
