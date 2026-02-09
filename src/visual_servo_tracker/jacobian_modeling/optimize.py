from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def calculate_rgb_difference(target_image: np.ndarray, sample_image: np.ndarray) -> np.ndarray:
    diff = target_image.astype(np.float64) - sample_image.astype(np.float64)
    return diff.reshape(-1, 1)


def compute_loss(true_dx: float, true_dy: float, estimated_dx: float, estimated_dy: float) -> float:
    return (true_dx - estimated_dx) ** 2 + (true_dy - estimated_dy) ** 2


def optimize_target(
    data_root: str | Path,
    outputs_root: str | Path,
    version: str,
    max_opt: int,
    iterations: int = 1000,
    learning_rate: float | None = None,
) -> dict[str, object]:
    if max_opt <= 0:
        raise ValueError("max_opt must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    data_root = Path(data_root)
    outputs_root = Path(outputs_root)

    sample_folder = data_root / version / "gap"
    jacobian_path = outputs_root / "jacobian" / version / "jacobian.pkl"
    target_path = data_root / version / "goal" / "0_0.jpg"

    with jacobian_path.open("rb") as file_obj:
        jacobian = pickle.load(file_obj)

    real_target_image = cv2.imread(str(target_path))
    if real_target_image is None:
        raise FileNotFoundError(f"Target image not found: {target_path}")

    target_height, target_width = real_target_image.shape[:2]
    target_image = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    initial_target_image = target_image.copy()

    save_interval = max(1, iterations // 10)
    best_loss = float("inf")
    best_target_image: np.ndarray | None = None
    best_learning_rate: float | None = None
    best_transitions_image: np.ndarray | None = None

    if learning_rate is None:
        learning_rates = [1 / (10**i) for i in range(10)]
    else:
        learning_rates = [learning_rate]

    for lr in learning_rates:
        LOGGER.info("Testing learning rate: %s", lr)
        target_image = initial_target_image.copy()
        transitions: list[np.ndarray] = []

        for i in range(iterations):
            if i % 4 == 0:
                dx, dy = -max_opt, 0
            elif i % 4 == 1:
                dx, dy = max_opt, 0
            elif i % 4 == 2:
                dx, dy = 0, max_opt
            else:
                dx, dy = 0, -max_opt

            sample_path = sample_folder / f"{dx}_{dy}.jpg"
            sample_image = cv2.imread(str(sample_path))
            if sample_image is None:
                raise FileNotFoundError(f"Sample image not found: {sample_path}")

            rgb_diff = calculate_rgb_difference(target_image, sample_image)
            predict = np.dot(jacobian, rgb_diff)
            diff = np.array([dx - predict[0][0], dy - predict[1][0]], dtype=np.float64)

            j_t = jacobian.T
            delta = np.dot(j_t, diff).reshape(target_image.shape)
            target_image = (target_image.astype(np.float64) + delta * lr).clip(0, 255).astype(np.uint8)

            loss = compute_loss(dx, dy, float(predict[0][0]), float(predict[1][0]))

            # Keep previous behavior: choose best in the final 10% iterations.
            if i >= iterations - max(1, iterations // 10) and loss < best_loss:
                best_loss = loss
                best_target_image = target_image.copy()
                best_learning_rate = lr
                best_transitions_image = np.hstack(transitions) if transitions else None

            if i % save_interval == 0:
                transitions.append(cv2.resize(target_image, (100, 50)))
                LOGGER.info("lr=%s iteration=%s loss=%s", lr, i, loss)

    if best_target_image is None:
        LOGGER.warning("No best target selected in tail window; fallback to the latest image.")
        best_target_image = target_image.copy()
        best_learning_rate = learning_rates[-1]

    optimize_dir = outputs_root / "optimized" / version
    optimize_dir.mkdir(parents=True, exist_ok=True)

    result_path = optimize_dir / "result.jpg"
    transition_path = optimize_dir / "transition.jpg"

    cv2.imwrite(str(result_path), best_target_image)
    if best_transitions_image is not None:
        cv2.imwrite(str(transition_path), best_transitions_image)

    LOGGER.info("Optimization complete. best_lr=%s best_loss=%s", best_learning_rate, best_loss)

    return {
        "initial_image": initial_target_image,
        "best_image": best_target_image,
        "transition_image": best_transitions_image,
        "best_learning_rate": best_learning_rate,
        "best_loss": best_loss,
        "result_path": result_path,
        "transition_path": transition_path if best_transitions_image is not None else None,
    }
