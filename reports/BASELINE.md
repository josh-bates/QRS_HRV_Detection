# Baseline Report

## Pipeline

Pan-Tompkins QRS detection -> RR interval cleaning -> non-overlapping 5-minute HRV windows -> seven HRV parameters.

## Training QRS Metrics

- Aggregate Sensitivity: 0.9929
- Aggregate PPV: 0.9911
- Aggregate F1: 0.9920

## Catastrophic QRS Failures

- None under F1 < 0.5.

## HRV Baseline MAPE

| Parameter | All Records MAPE % | Records 1-20 MAPE % | n_valid | n_nan |
|---|---:|---:|---:|---:|
| avgRR | 0.4660 | 0.2147 | 35 | 0 |
| sdRR | 8.3348 | 5.4028 | 35 | 0 |
| RMSSD | 18.4003 | 17.2941 | 35 | 0 |
| pNN50 | 34.1022 | 38.0263 | 35 | 0 |
| LF | 23.5395 | 14.9771 | 35 | 0 |
| HF | 33.4477 | 40.2136 | 35 | 0 |
| LF_HFratio | 18.2573 | 21.7538 | 35 | 0 |

## Known Weak Points

- Worst HRV parameter by all-record MAPE: pNN50 (34.1022%).
- The detector fails badly on a cluster of noisy/degraded recordings, so HRV errors from Pan-Tompkins detections are dominated by QRS failure, not the independently validated HRV implementation.
- Subsequent improvement work should target post-detection refinement and the failure cluster before spending submissions.
