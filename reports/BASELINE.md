# Baseline Report

## Pipeline

Pan-Tompkins QRS detection -> RR interval cleaning -> non-overlapping 5-minute HRV windows -> seven HRV parameters.

## Training QRS Metrics

- Aggregate Sensitivity: 0.7866
- Aggregate PPV: 0.9875
- Aggregate F1: 0.8757

## Catastrophic QRS Failures

- Recording 3: F1=0.0000, TP=0, FP=5, FN=34253
- Recording 24: F1=0.0000, TP=0, FP=442, FN=24613
- Recording 26: F1=0.0000, TP=0, FP=806, FN=27668
- Recording 29: F1=0.0000, TP=0, FP=1987, FN=28083
- Recording 30: F1=0.0093, TP=131, FP=84, FN=27781
- Recording 31: F1=0.0019, TP=26, FP=34, FN=27978
- Recording 34: F1=0.0000, TP=0, FP=1259, FN=31169
- Recording 35: F1=0.0001, TP=1, FP=727, FN=23955

## HRV Baseline MAPE

| Parameter | All Records MAPE % | Records 1-20 MAPE % | n_valid | n_nan |
|---|---:|---:|---:|---:|
| avgRR | 7.3401 | 0.1914 | 31 | 4 |
| sdRR | 22.9377 | 5.1807 | 31 | 4 |
| RMSSD | 74.4452 | 17.3033 | 31 | 4 |
| pNN50 | 211.0578 | 39.9264 | 31 | 4 |
| LF | 46.5530 | 33.6509 | 31 | 4 |
| HF | 171.6467 | 37.0584 | 31 | 4 |
| LF_HFratio | 31.8025 | 21.4657 | 31 | 4 |

## Known Weak Points

- Worst HRV parameter by all-record MAPE: pNN50 (211.0578%).
- The detector fails badly on a cluster of noisy/degraded recordings, so HRV errors from Pan-Tompkins detections are dominated by QRS failure, not the independently validated HRV implementation.
- Subsequent improvement work should target post-detection refinement and the failure cluster before spending submissions.
