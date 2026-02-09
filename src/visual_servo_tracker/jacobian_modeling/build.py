from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def calculate_rgb_difference(target_image: np.ndarray, sample_image: np.ndarray) -> np.ndarray:
    diff = target_image.astype(np.float32) - sample_image.astype(np.float32)
    return diff.reshape(-1, 1)


def _append_pickle(filepath: Path, new_data: np.ndarray) -> np.ndarray:
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("wb") as file_obj:
            pickle.dump(new_data, file_obj)
        return new_data

    with filepath.open("rb") as file_obj:
        old_data = pickle.load(file_obj)

    merged = np.hstack((old_data, new_data))
    with filepath.open("wb") as file_obj:
        pickle.dump(merged, file_obj)
    return merged


def _overwrite_pickle(filepath: Path, data: np.ndarray) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as file_obj:
        pickle.dump(data, file_obj)


def least_squares_block(B: np.ndarray, q: np.ndarray) -> np.ndarray:
    B_T = B.T
    B_T_B = np.dot(B_T, B)
    try:
        inv_B_T_B = np.linalg.inv(B_T_B)
    except np.linalg.LinAlgError:
        LOGGER.warning("B^T*B is singular; fallback to pseudo-inverse for stability.")
        inv_B_T_B = np.linalg.pinv(B_T_B)
    B_inv = np.dot(inv_B_T_B, B_T)
    p = np.dot(q, B_inv)
    return p


def build_jacobian(
    data_root: str | Path,
    outputs_root: str | Path,
    version: str,
    sample_gap: int,
    reset: bool = True,
) -> Path:
    if sample_gap <= 0:
        raise ValueError("sample_gap must be positive")

    data_root = Path(data_root)
    outputs_root = Path(outputs_root)

    sample_folder = data_root / version / "gap"
    target_path = data_root / version / "goal" / "0_0.jpg"
    jacobian_folder = outputs_root / "jacobian" / version

    if reset and jacobian_folder.exists():
        shutil.rmtree(jacobian_folder)

    jacobian_folder.mkdir(parents=True, exist_ok=True)

    target_image = cv2.imread(str(target_path))
    if target_image is None:
        raise FileNotFoundError(f"Target image not found: {target_path}")

    gap_list = [(sample_gap, 0), (-sample_gap, 0), (0, sample_gap), (0, -sample_gap)]

    for dx, dy in gap_list:
        sample_path = sample_folder / f"{dx}_{dy}.jpg"
        sample_image = cv2.imread(str(sample_path))
        if sample_image is None:
            raise FileNotFoundError(f"Sample image not found: {sample_path}")

        rgb_difference = calculate_rgb_difference(target_image, sample_image)
        theta = np.array([dx, dy]).reshape(-1, 1)

        B = _append_pickle(jacobian_folder / "RGB.pkl", rgb_difference)
        q = _append_pickle(jacobian_folder / "theta.pkl", theta)
        jacobian = least_squares_block(B, q)
        _overwrite_pickle(jacobian_folder / "jacobian.pkl", jacobian)

    LOGGER.info("Jacobian built at %s", jacobian_folder / "jacobian.pkl")
    return jacobian_folder / "jacobian.pkl"
