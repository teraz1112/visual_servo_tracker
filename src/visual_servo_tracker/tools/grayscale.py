from __future__ import annotations

from pathlib import Path

import cv2


def convert_to_grayscale(input_image_path: str | Path, output_image_path: str | Path) -> Path:
    image = cv2.imread(str(input_image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {input_image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), gray)
    return output_path
