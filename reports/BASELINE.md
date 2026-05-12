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
| avgRR | 0.2289 | 0.1130 | 35 | 0 |
| sdRR | 2.9178 | 2.4699 | 35 | 0 |
| RMSSD | 6.9491 | 7.6915 | 35 | 0 |
| pNN50 | 6.1184 | 7.2071 | 35 | 0 |
| LF | 9.2857 | 8.0053 | 35 | 0 |
| HF | 13.4196 | 17.2750 | 35 | 0 |
| LF_HFratio | 14.6120 | 15.7890 | 35 | 0 |

## Known Weak Points

- Worst HRV parameter by all-record MAPE: LF_HFratio (14.6120%).
- The detector fails badly on a cluster of noisy/degraded recordings, so HRV errors from Pan-Tompkins detections are dominated by QRS failure, not the independently validated HRV implementation.
- Subsequent improvement work should target post-detection refinement and the failure cluster before spending submissions.
