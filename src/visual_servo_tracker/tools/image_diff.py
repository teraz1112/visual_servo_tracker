from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def calculate_rgb_difference(image1_path: str | Path, image2_path: str | Path) -> np.ndarray:
    image1 = cv2.imread(str(image1_path))
    image2 = cv2.imread(str(image2_path))

    if image1 is None:
        raise FileNotFoundError(f"Image not found: {image1_path}")
    if image2 is None:
        raise FileNotFoundError(f"Image not found: {image2_path}")

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    diff = cv2.subtract(image1, image2)
    return np.maximum(diff, 0)


def show_rgb_difference(image1_path: str | Path, image2_path: str | Path) -> None:
    diff = calculate_rgb_difference(image1_path, image2_path)
    image1 = cv2.imread(str(image1_path))
    image2 = cv2.imread(str(image2_path))

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("Difference", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
