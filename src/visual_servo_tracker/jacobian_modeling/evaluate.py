from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def calculate_rgb_difference(target_image: np.ndarray, sample_image: np.ndarray) -> np.ndarray:
    diff = target_image.astype(np.float32) - sample_image.astype(np.float32)
    return diff.reshape(-1, 1)


def evaluate_jacobian(
    data_root: str | Path,
    outputs_root: str | Path,
    version: str,
    eval_type: str,
    radius: int,
) -> tuple[Path, Path]:
    if eval_type not in {"normal", "optimized", "wide"}:
        raise ValueError("eval_type must be one of: normal, optimized, wide")
    if radius <= 0:
        raise ValueError("radius must be positive")

    data_root = Path(data_root)
    outputs_root = Path(outputs_root)

    sample_folder = data_root / version / "gap"
    jacobian_path = outputs_root / "jacobian" / version / "jacobian.pkl"

    if eval_type == "optimized":
        target_image_path = outputs_root / "optimized" / version / "result.jpg"
        graph_name = f"{version}_optimized_graph"
    else:
        target_image_path = data_root / version / "goal" / "0_0.jpg"
        graph_name = f"{version}_{eval_type}_graph"

    if not jacobian_path.exists():
        raise FileNotFoundError(f"Jacobian not found: {jacobian_path}")

    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    with jacobian_path.open("rb") as file_obj:
        jacobian = pickle.load(file_obj)

    graph_dir = outputs_root / "graph" / graph_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    files = sorted([f for f in os.listdir(sample_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for current_radius in range(1, radius + 1):
        true_values: list[list[float]] = []
        estimated_values: list[list[float]] = []

        for file_name in files:
            dx, dy = map(int, os.path.splitext(file_name)[0].split("_"))
            if not (
                dx == current_radius
                or dy == current_radius
                or dx == -current_radius
                or dy == -current_radius
            ):
                continue

            sample_image = cv2.imread(str(sample_folder / file_name))
            if sample_image is None:
                continue

            rgb_diff = calculate_rgb_difference(target_image, sample_image)
            predict = np.dot(jacobian, rgb_diff)

            true_values.append([dx, dy])
            estimated_values.append([dx - float(predict[0][0]), dy - float(predict[1][0])])

        if not true_values:
            LOGGER.warning("No data points found for radius=%s", current_radius)
            continue

        true_arr = np.array(true_values)
        estimated_arr = np.array(estimated_values)

        plt.figure(figsize=(8, 8))
        plt.scatter(true_arr[:, 0], true_arr[:, 1], color="blue", label="True Values", s=100)
        plt.scatter(estimated_arr[:, 0], estimated_arr[:, 1], color="red", label="Estimated Values", s=100)

        for i in range(len(true_arr)):
            plt.arrow(
                true_arr[i, 0],
                true_arr[i, 1],
                estimated_arr[i, 0] - true_arr[i, 0],
                estimated_arr[i, 1] - true_arr[i, 1],
                head_width=0.1,
                head_length=0.1,
                fc="gray",
                ec="gray",
            )

        plt.xlabel("X (pixel)")
        plt.ylabel("Y (pixel)")
        plt.title(f"Move Prediction for radius = {current_radius}")
        plt.legend()
        plt.grid(True)
        plt.xlim(-current_radius - 5, current_radius + 5)
        plt.ylim(-current_radius - 5, current_radius + 5)

        out_path = graph_dir / f"plot_{graph_name}_{current_radius}.png"
        plt.savefig(out_path)
        plt.close()
        saved_paths.append(out_path)

    if not saved_paths:
        raise RuntimeError("No graph images were generated.")

    half_idx = max(1, radius // 2)
    half_path = graph_dir / f"plot_{graph_name}_{half_idx}.png"
    full_path = graph_dir / f"plot_{graph_name}_{radius}.png"

    if not half_path.exists():
        half_path = saved_paths[0]
    if not full_path.exists():
        full_path = saved_paths[-1]

    LOGGER.info("Evaluation graph generated at %s", graph_dir)
    return half_path, full_path
