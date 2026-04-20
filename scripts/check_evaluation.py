#!/usr/bin/env python3
"""Phase 02 scoring smoke checks against the project brief."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import mape, match_qrs, score_record
from src.io import load_training_data


def main() -> None:
    _, qrs_expert_list = load_training_data()
    rec1_expert = qrs_expert_list[0]

    perfect = score_record(rec1_expert, rec1_expert)
    assert perfect["tp"] == len(rec1_expert)
    assert perfect["fp"] == 0
    assert perfect["fn"] == 0
    assert perfect["sens"] == 1.0
    assert perfect["ppv"] == 1.0
    assert perfect["f1"] == 1.0

    empty = score_record(np.array([], dtype=np.int64), rec1_expert)
    assert empty["tp"] == 0
    assert empty["fp"] == 0
    assert empty["fn"] == len(rec1_expert)
    assert empty["f1"] == 0.0

    synthetic_tp_fp_fn = match_qrs(
        qrs_detected=np.array([101, 205, 299, 500]),
        qrs_expert=np.array([100, 200, 300, 400]),
        tol_samples=5,
    )
    assert synthetic_tp_fp_fn == (3, 1, 1)

    synthetic_mape = mape(
        ref=np.array([100, 200, 400, 800], dtype=float),
        pred=np.array([110, 180, 420, 760], dtype=float),
    )
    assert np.isclose(synthetic_mape, 7.5)

    print("Phase 02 evaluation smoke checks passed.")
    print(
        "Record 1 expert-as-detection: "
        f"TP={perfect['tp']} FP={perfect['fp']} FN={perfect['fn']} "
        f"Sens={perfect['sens']:.4f} PPV={perfect['ppv']:.4f} F1={perfect['f1']:.4f}"
    )
    print("Synthetic QRS counts: TP=3 FP=1 FN=1")
    print("Synthetic HRV MAPE: 7.5")


if __name__ == "__main__":
    main()
