from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.io import (
    FS,
    HRV_KEYS,
    NUM_RECORDINGS,
    TEMPLATE_KEYS,
    load_submission_template,
    load_training_data,
    save_submission,
)


def test_training_data_brief_sanity_checks():
    ecg_list, qrs_expert_list = load_training_data()

    assert FS == 100
    assert len(ecg_list) == NUM_RECORDINGS
    assert len(qrs_expert_list) == NUM_RECORDINGS
    assert ecg_list[0][:100].shape == (100,)
    assert ecg_list[0][:100].dtype == np.float64
    assert qrs_expert_list[0][:3].tolist() == [40, 128, 213]
    assert np.isclose(len(ecg_list[0]) / (FS * 3600), 8.2139, atol=1e-3)


def test_submission_template_roundtrip(tmp_path):
    template = load_submission_template()
    output_path = tmp_path / "ProjectTestDataAnalysis_roundtrip.mat"

    save_submission(output_path, template["QRS"], {key: template[key] for key in HRV_KEYS})
    rewritten = loadmat(output_path)

    assert {key for key in rewritten if not key.startswith("__")} == set(TEMPLATE_KEYS)
    assert rewritten["QRS"].shape == (1, NUM_RECORDINGS)
    for key in HRV_KEYS:
        assert rewritten[key].shape == (1, NUM_RECORDINGS)


def test_submission_excludes_ecg(tmp_path):
    qrs_list = [np.array([100, 200, 300], dtype=np.int64) for _ in range(NUM_RECORDINGS)]
    hrv_dict = {key: np.full(NUM_RECORDINGS, np.nan) for key in HRV_KEYS}
    output_path = tmp_path / "submission.mat"

    save_submission(output_path, qrs_list, hrv_dict)
    mat = loadmat(output_path)

    assert "ECG" not in mat
    assert Path(output_path).exists()
