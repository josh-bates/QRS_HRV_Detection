"""MATLAB data I/O helpers for the BMET3997/9997 ECG major project.

The project data is supplied as MATLAB ``.mat`` files containing cell arrays:

* ``ProjectTrainData.mat`` has ``ECG`` and ``QRSexpert`` cells, each with 35
  recordings.
* ``ProjectTestData.mat`` has ``ECG`` only.
* ``ProjectTestDataAnalysis.mat`` is the submission template with ``QRS`` and
  seven HRV arrays.

Important indexing convention
-----------------------------
QRS sample locations are kept as **1-indexed MATLAB sample positions** inside
this Python project. This matches the brief, the expert annotations, and the
required submission format. Convert to 0-indexed NumPy positions only at the
point where an ECG array is indexed, e.g. ``ecg[qrs_samples - 1]``.

No QRS-detection or HRV-analysis libraries are used here. This module only
does generic file loading, shape conversion, validation, and saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat, savemat


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

FS = 100
"""Sampling frequency in Hz."""

ECG_UNIT = "uV"
"""ECG amplitude unit. The brief states 1 LSB = 5 uV; data is already in uV."""

NUM_RECORDINGS = 35
"""Number of ECG recordings in both the training and test datasets."""

HRV_KEYS = ("avgRR", "sdRR", "RMSSD", "pNN50", "LF", "HF", "LF_HFratio")
"""Required scalar HRV outputs per recording."""

TEMPLATE_KEYS = ("QRS", *HRV_KEYS)
"""Variables that are allowed in the submission file."""


def load_training_data(
    path: str | Path = "data/ProjectTrainData.mat",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load training ECG recordings and expert QRS annotations.

    Parameters
    ----------
    path:
        Path to ``ProjectTrainData.mat``. The planned repository layout uses
        ``data/``; this checkout currently stores provided files in
        ``resources/``, so the default path falls back there if needed.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``ecg_list`` contains 35 1-D ``float64`` arrays in microvolts.
        ``qrs_expert_list`` contains 35 1-D ``int64`` arrays of 1-indexed
        MATLAB sample positions.
    """

    mat = loadmat(_resolve_data_path(path, "ProjectTrainData.mat"))
    _require_keys(mat, ("ECG", "QRSexpert"))

    ecg_list = _unwrap_matlab_cell_vector(mat["ECG"], dtype=np.float64, name="ECG")
    qrs_expert_list = _unwrap_matlab_cell_vector(
        mat["QRSexpert"], dtype=np.int64, name="QRSexpert"
    )
    _validate_recording_count(ecg_list, "ECG")
    _validate_recording_count(qrs_expert_list, "QRSexpert")
    return ecg_list, qrs_expert_list


def load_test_data(path: str | Path = "data/ProjectTestData.mat") -> list[np.ndarray]:
    """Load test ECG recordings.

    Parameters
    ----------
    path:
        Path to ``ProjectTestData.mat``. As with ``load_training_data``, the
        default planned ``data/`` path falls back to ``resources/`` in this
        checkout.

    Returns
    -------
    list[np.ndarray]
        35 1-D ``float64`` ECG arrays in microvolts.
    """

    mat = loadmat(_resolve_data_path(path, "ProjectTestData.mat"))
    _require_keys(mat, ("ECG",))

    ecg_list = _unwrap_matlab_cell_vector(mat["ECG"], dtype=np.float64, name="ECG")
    _validate_recording_count(ecg_list, "ECG")
    return ecg_list


def load_submission_template(
    path: str | Path = "data/ProjectTestDataAnalysis.mat",
) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Load the MATLAB submission template into plain Python structures.

    The returned dictionary has exactly the required submission variables:
    ``QRS`` as a list of 35 placeholder arrays, and each HRV key as a 1-D
    ``float64`` vector of length 35. The template values are preserved, which
    makes this function suitable for round-trip format tests.
    """

    mat = loadmat(_resolve_data_path(path, "ProjectTestDataAnalysis.mat"))
    _require_keys(mat, TEMPLATE_KEYS)

    qrs_list = _unwrap_matlab_cell_vector(mat["QRS"], dtype=np.int64, name="QRS")
    _validate_recording_count(qrs_list, "QRS")

    template: dict[str, list[np.ndarray] | np.ndarray] = {"QRS": qrs_list}
    for key in HRV_KEYS:
        values = np.asarray(mat[key], dtype=np.float64).reshape(-1)
        if values.size != NUM_RECORDINGS:
            raise ValueError(
                f"{key} must contain {NUM_RECORDINGS} values, got {values.size}"
            )
        template[key] = values

    return template


def save_submission(
    path: str | Path,
    qrs_list: Iterable[np.ndarray | Iterable[int]],
    hrv_dict: dict[str, np.ndarray | Iterable[float]],
) -> None:
    """Save QRS detections and HRV outputs in the required MATLAB format.

    The output file contains only:

    * ``QRS`` as a MATLAB cell array with shape ``1 x 35``.
    * ``avgRR``, ``sdRR``, ``RMSSD``, ``pNN50``, ``LF``, ``HF``, and
      ``LF_HFratio`` as ``1 x 35`` ``float64`` row vectors.

    No ``ECG`` variable or diagnostic variables are written.
    """

    qrs_cells = _make_qrs_cell_array(qrs_list)
    payload: dict[str, np.ndarray] = {"QRS": qrs_cells}

    for key in HRV_KEYS:
        if key not in hrv_dict:
            raise KeyError(f"Missing HRV output '{key}'")
        payload[key] = _as_row_vector(hrv_dict[key], key)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, payload, do_compression=True)


def _resolve_data_path(path: str | Path, filename: str) -> Path:
    """Resolve a data file path, supporting both planned and current layouts."""

    candidate = Path(path)
    repo_candidate = _PROJECT_ROOT / candidate if not candidate.is_absolute() else candidate
    fallback = _PROJECT_ROOT / "resources" / filename

    for option in (candidate, repo_candidate, fallback):
        if option.exists():
            return option

    raise FileNotFoundError(
        f"Could not find {filename}. Checked '{candidate}', "
        f"'{repo_candidate}', and '{fallback}'."
    )


def _require_keys(mat: dict, required_keys: Iterable[str]) -> None:
    missing = [key for key in required_keys if key not in mat]
    if missing:
        raise KeyError(f"MAT file is missing required variable(s): {missing}")


def _unwrap_matlab_cell_vector(
    cell_array: np.ndarray, *, dtype: np.dtype, name: str
) -> list[np.ndarray]:
    """Convert a MATLAB cell vector loaded by SciPy into a list of 1-D arrays."""

    if not isinstance(cell_array, np.ndarray) or cell_array.dtype != object:
        raise TypeError(f"{name} must be a MATLAB cell array loaded as dtype object")

    values: list[np.ndarray] = []
    for index, cell in enumerate(cell_array.reshape(-1), start=1):
        array = np.asarray(cell, dtype=dtype).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} cell {index} is empty")
        values.append(array)

    return values


def _validate_recording_count(values: list[np.ndarray], name: str) -> None:
    if len(values) != NUM_RECORDINGS:
        raise ValueError(f"{name} must contain {NUM_RECORDINGS} recordings")


def _make_qrs_cell_array(
    qrs_list: Iterable[np.ndarray | Iterable[int]],
) -> np.ndarray:
    qrs_values = [np.asarray(qrs, dtype=np.int64).reshape(-1) for qrs in qrs_list]
    _validate_recording_count(qrs_values, "QRS")

    qrs_cells = np.empty((1, NUM_RECORDINGS), dtype=object)
    for idx, qrs in enumerate(qrs_values):
        qrs_cells[0, idx] = qrs
    return qrs_cells


def _as_row_vector(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != NUM_RECORDINGS:
        raise ValueError(
            f"{name} must contain {NUM_RECORDINGS} values, got {array.size}"
        )
    return array.reshape(1, NUM_RECORDINGS)
