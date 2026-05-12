# Baseline Report

## Pipeline

Pan-Tompkins QRS detection -> RR interval cleaning -> non-overlapping 5-minute HRV windows -> seven HRV parameters.

## Training QRS Metrics

- Aggregate Sensitivity: 0.9952
- Aggregate PPV: 0.9948
- Aggregate F1: 0.9950

## Catastrophic QRS Failures

- None under F1 < 0.5.

## HRV Baseline MAPE

| Parameter | All Records MAPE % | Records 1-20 MAPE % | n_valid | n_nan |
|---|---:|---:|---:|---:|
| avgRR | 0.2291 | 0.1134 | 35 | 0 |
| sdRR | 2.9083 | 2.4793 | 35 | 0 |
| RMSSD | 6.9052 | 7.6138 | 35 | 0 |
| pNN50 | 5.6019 | 6.9187 | 35 | 0 |
| LF | 9.2065 | 7.9796 | 35 | 0 |
| HF | 12.9407 | 16.8792 | 35 | 0 |
| LF_HFratio | 13.0407 | 11.6601 | 35 | 0 |

## Known Weak Points

- Worst HRV parameter by all-record MAPE: LF_HFratio (13.0407%).
- The detector fails badly on a cluster of noisy/degraded recordings, so HRV errors from Pan-Tompkins detections are dominated by QRS failure, not the independently validated HRV implementation.
- Subsequent improvement work should target post-detection refinement and the failure cluster before spending submissions.
