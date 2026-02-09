from __future__ import annotations

import pickle
from pathlib import Path

from visual_servo_tracker.jacobian_modeling.build import build_jacobian
from visual_servo_tracker.jacobian_modeling.evaluate import evaluate_jacobian
from visual_servo_tracker.jacobian_modeling.optimize import optimize_target


def test_offline_pipeline_minimal(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "samples"
    outputs_root = tmp_path / "outputs"

    jacobian_path = build_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version="circle_red_green",
        sample_gap=2,
        reset=True,
    )
    assert jacobian_path.exists()

    with jacobian_path.open("rb") as file_obj:
        jacobian = pickle.load(file_obj)
    assert jacobian.shape[0] == 2

    normal_half, normal_full = evaluate_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version="circle_red_green",
        eval_type="normal",
        radius=2,
    )
    assert normal_half.exists()
    assert normal_full.exists()

    opt_result = optimize_target(
        data_root=data_root,
        outputs_root=outputs_root,
        version="circle_red_green",
        max_opt=2,
        iterations=20,
        learning_rate=0.01,
    )
    assert Path(opt_result["result_path"]).exists()

    opt_half, opt_full = evaluate_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version="circle_red_green",
        eval_type="optimized",
        radius=2,
    )
    assert opt_half.exists()
    assert opt_full.exists()
