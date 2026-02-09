from __future__ import annotations

import pickle
from pathlib import Path

from visual_servo_tracker.jacobian_modeling.build import build_jacobian


def test_jacobian_shape(tmp_path: Path) -> None:
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

    with jacobian_path.open("rb") as file_obj:
        jacobian = pickle.load(file_obj)

    assert jacobian.ndim == 2
    assert jacobian.shape[0] == 2
