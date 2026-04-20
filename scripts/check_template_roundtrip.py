#!/usr/bin/env python3
"""Verify that the submission template can be saved without format drift."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import HRV_KEYS, TEMPLATE_KEYS, load_submission_template, save_submission


def main() -> None:
    template_path = _resolve_template_path()
    original = loadmat(template_path)
    template = load_submission_template(template_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "ProjectTestDataAnalysis_roundtrip.mat"
        save_submission(output_path, template["QRS"], {key: template[key] for key in HRV_KEYS})
        rewritten = loadmat(output_path)

    original_keys = _public_keys(original)
    rewritten_keys = _public_keys(rewritten)
    expected_keys = set(TEMPLATE_KEYS)
    assert set(original_keys) == expected_keys, original_keys
    assert set(rewritten_keys) == expected_keys, rewritten_keys

    assert original["QRS"].shape == rewritten["QRS"].shape
    for idx in range(original["QRS"].shape[1]):
        np.testing.assert_array_equal(
            np.asarray(original["QRS"][0, idx]).reshape(-1),
            np.asarray(rewritten["QRS"][0, idx]).reshape(-1),
        )

    for key in HRV_KEYS:
        assert original[key].shape == rewritten[key].shape
        np.testing.assert_array_equal(original[key], rewritten[key])

    print("Submission template round-trip check passed.")
    print(f"Template variables: {', '.join(TEMPLATE_KEYS)}")
    print(f"QRS shape: {rewritten['QRS'].shape}")


def _resolve_template_path() -> Path:
    planned = PROJECT_ROOT / "data" / "ProjectTestDataAnalysis.mat"
    fallback = PROJECT_ROOT / "resources" / "ProjectTestDataAnalysis.mat"
    return planned if planned.exists() else fallback


def _public_keys(mat: dict) -> list[str]:
    return [key for key in mat if not key.startswith("__")]


if __name__ == "__main__":
    main()
