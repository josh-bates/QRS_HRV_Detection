# BMET3997/9997 – Physiological Signal Sensing and Processing
## Major Project: HRV Parameters – Frequency Domain Methods

---

## Measurement of HRV

Heart rate variability (HRV) is a measure of the variation in time between each heartbeat. It can be measured using the ECG.

---

## Steps for the HRV Calculation

1. QRS detection
2. Identifying the non-normal beats
3. Calculate the RR intervals between the successive normal beats
4. Calculate the HRV parameters

**Issue:** Some HRV methods assume that the RR intervals are contiguous (i.e. no gaps). Ectopic beats cause issues.

---

## HRV Time Domain Features

| Parameter | Calculation |
|---|---|
| Mean RR-interval | Average value of the RR intervals |
| SDNN | Standard deviation of the RR intervals |
| Serial correlation of RR intervals | Correlation of the RR intervals with the RR intervals shifted by fixed number of beats |
| NN50 | Number of pairs of adjacent RR-intervals where the first differs from the second by more than 50 ms |
| pNN50 | NN50 divided by the total number of RR-intervals × 100 |
| SDSD | Standard deviation of the differences between adjacent RR-intervals |
| RMSSD | Square root of the mean of the squares of differences between adjacent RR-intervals |
| HRV triangular index | Total number of all RR intervals divided by the maximum of the density distribution of the RRs |

### MATLAB Code (RR vector scaled in seconds, e.g. `RR=[0.85, 0.90, 1.00, 0.98, 1.1...]`)

```matlab
meanRR = mean(RR);
stdRR = std(RR);

L = length(RR);
temp = xcorr(RR, 'coeff');
Corr = temp(L+1);

NN50 = sum(diff(RR) > 0.05 | diff(RR) < -0.05);
pNN50 = NN50 / (length(RR));
SDSD = std(diff(RR));
RMSSD = sqrt(mean(diff(RR).^2));

[~, count] = mode(RR);
TriI = length(RR) / count;
```

### Python Code

```python
import numpy as np
from scipy import stats

meanRR = np.mean(RR)
stdRR = np.std(RR)

NN50 = np.sum((np.diff(RR) > 0.05) | (np.diff(RR) < -0.05))
pNN50 = NN50 / len(RR)
SDSD = np.std(np.diff(RR))
RMSSD = np.sqrt(np.mean(np.diff(RR)**2))

mode_value, count = stats.mode(RR)
TriI = len(RR) / count[0]
```

---

## Ectopic Beats and Noise

- **Important:** HRV is normal-normal beat variability.
- Ideally, remove ectopic beats first as they can heavily influence HRV.
- The choice of ectopic detector can itself influence HRV.
- Each ectopic beat affects **2 RR intervals**, creating a problem of missing values.

**Example effect on standard deviation:**
- Uncorrected RR intervals (ms): 856, 652, 1235, 870, 905, 632 → St. dev. = 218 ms
- Corrected (ectopic removed): 856, –, –, 870, 905, – → St. dev. = 25 ms

### Handling Non-Normal Beats in MATLAB

Each non-normal beat affects two RR intervals. A convenient approach is to assign the QRS times associated with non-normal beats to `NaN`.

```matlab
QRS = [1, 1.95, 2.23, 3.94, 4.87];  % Time of each QRS in seconds

% Process including the ectopic beat
RR = diff(QRS);
meanRR = mean(RR);
stdRR = std(RR);
% Result: meanRR = 0.9675, stdRR = 0.5847

% Process ignoring the ectopic beat (QRS3)
QRS(3) = nan;
RR = diff(QRS);
meanRR = mean(RR, 'omitnan');
stdRR = std(RR, 'omitnan');
% Result: meanRR = 0.9400, stdRR = 0.0141
% Much smaller standard deviation as the very long and very short RR intervals caused by the ectopic beat are ignored
```

---

## HRV Frequency Domain Features

| Parameter | Calculation |
|---|---|
| VLF | Power in the very-low-frequency range: ≤0.04 Hz (short-term); 0.003–0.04 Hz (24-hour) |
| LF | Power in the low-frequency range: 0.04–0.15 Hz |
| HF | Power in the high-frequency range: 0.15–0.4 Hz |
| TP | Total power ≤0.4 Hz; equal to the variance of NN intervals over the selected segment |
| LF/HF | Ratio of LF to HF power |

---

## Calculating PSD of RR Intervals

QRS points are unevenly sampled, so the RR time series is also unevenly sampled. The DFT cannot be directly applied to the RR time series as it assumes evenly sampled data.

---

## Method 1: Resample RR Time Series

Resample the time series evenly then apply the DFT.

**Advantages:**
- Conceptually easy

**Disadvantages:**
- Resampled values depend on the chosen interpolation method
- The series can have sudden changes (e.g. at t=6 seconds), causing unwanted high-frequency components in the PSD

**Method not recommended.**

---

## Method 2a: Spectrum of RR-Intervals (Interval-Based Periodogram)

Find the DFT of the RR interval **series** (not the RR interval time series).

**Advantages:**
- No ambiguity about the method

**Disadvantages:**
- The spectrum cannot be directly interpreted in terms of frequency
- Cannot handle missing values

### Calculating the Periodogram

Given the RR interval sequence `RR`, compute:

$$|\mathcal{F}(RR - \text{mean}(RR))|^2$$

The x-axis units are **cycles per average RR-interval**.

```matlab
% MATLAB
P = abs(fft(RR)).^2;
```

```python
# Python
P = np.abs(np.fft.fft(RR))**2
```

Only the lower half of the spectrum is needed, as the upper half is symmetrical.

### Converting X-Axis from Cycles per Average Heart Interval to Hertz

Divide x-axis bin points by the average RR-interval.

**Example:** If average RR-interval = 0.8 seconds, then the x-axis values 0, 0.1, 0.2, 0.3, 0.4, 0.5 (cycles/interval) become 0, 0.125, 0.25, 0.375, 0.5, 0.675 Hz.

### MATLAB Example: Major Project Record 5

The ECG signal is divided into 1-minute segments, a PSD is calculated for each segment, and the PSDs are averaged.

```matlab
load ProjectTrainData.mat
i = 5;
RR{5} = diff(QRSexpert{5}) / 100;
EpochLen = 100 * 60;  % 1 minute segments at 100 Hz

close all
for j = 1:length(ECG{5}) / EpochLen
    SegmentStart = (j-1) * EpochLen + 1;
    SegmentEnd = j * EpochLen;
    RRseg = RR{i}(QRSexpert{i} >= SegmentStart & QRSexpert{i} <= SegmentEnd);

    if ~any(isnan(RRseg))
        % Zero-pads RR interval sequence to 256 points so all segment PSDs are same length
        PSD(j,:) = abs(fft(RRseg - mean(RRseg), 256)).^2;
    else
        % If any NaNs in the segment, cannot calculate a spectrum for that segment
        PSD(j, 1:256) = nan;
    end
end

% Average spectrum in cycles per interval
PSDavg = mean(PSD, 1, 'omitnan');
clf; plot([0:255]/256, PSDavg)
hold on; plot([0:255]/256, PSDavg, '.')
xlabel('cycles per average heart interval'); ylabel('Power (relative units)')

% Average spectrum (lower half) in Hertz
figure; plot([0:127]/256 / mean(RR{i}, 'omitnan'), PSDavg(1:128))
hold on; plot([0:127]/256 / mean(RR{i}, 'omitnan'), PSDavg(1:128), '.')
xlabel('Hertz'); ylabel('Power (relative units)')
title(['Average heart interval = ', num2str(mean(RR{i}, 'omitnan')), ' seconds'])
```

---

## Method 2b: Lomb-Scargle Periodogram

The Lomb periodogram does not use the DFT. Instead, it fits sinusoids to the data via least squares.

**Advantages:**
- Does not require evenly sampled data
- Handles missing data
- Properly deals with the uneven sampling inherently involved in RR series

**Disadvantages:**
- More computationally intensive: O(N²)

**Equivalence note:** The Lomb-Scargle periodogram is equivalent to the standard periodogram when:
- Data is evenly sampled
- The signal has zero mean
- There is no missing data
- Non-weighted least squares is used

**Implementation:**
- In MATLAB: use the `plomb` function
- In Python: use `LombScargle` from `astropy.timeseries`

```python
from astropy.timeseries import LombScargle
```

### MATLAB Example: Lomb-Scargle for Major Project Data

```matlab
load ProjectTrainData.mat
i = 5;
RR{5} = diff(QRSexpert{5}) / 100;
EpochLen = 100 * 60;

clear PSD
figure
for j = 1:length(ECG{i}) / EpochLen
    SegmentStart = (j-1) * EpochLen + 1;
    SegmentEnd = j * EpochLen;
    RRseg = RR{i}(QRSexpert{i} >= SegmentStart & QRSexpert{i} <= SegmentEnd);
    QRSseg = QRSexpert{i}(QRSexpert{i} > (j-1)*EpochLen & QRSexpert{i} <= j*EpochLen) / 100;  % QRS times in seconds

    % Remove instances of identical QRS detection times
    idx = find(diff(QRSseg) == 0);
    QRSseg(idx) = [];
    RRseg(idx) = [];

    if sum(RRseg) > 30  % Need minimum coverage of 30 seconds before calculating spectrum
        PSD(j,:) = plomb(RRseg - mean(RRseg, 'omitnan'), QRSseg, 'psd', 0.5*[0:99]/100);  % 100 bins between 0 and 0.5 Hz
    else
        PSD(j,:) = nan(1, 100);
    end
end

% Average spectrum in Hz
PSDavg = mean(PSD, 1, 'omitnan');
clf; plot(0.5*[0:99]/100, PSDavg)
hold on; plot(0.5*[0:99]/100, PSDavg, '.')
xlabel('Hz'); ylabel('Power (relative units)')
```

---

## Summary of Spectrum Methods

| Method | Handles Missing Data | Evenly Sampled Required | Notes |
|---|---|---|---|
| Resample + DFT | No | Yes (after resampling) | Not recommended; interpolation artefacts |
| Interval-based periodogram | No | N/A (uses interval series) | Simple; x-axis needs rescaling to Hz |
| Lomb-Scargle periodogram | Yes | No | Preferred for unevenly sampled/missing data; O(N²) cost |
