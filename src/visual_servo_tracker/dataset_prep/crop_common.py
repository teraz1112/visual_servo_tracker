from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def save_shifted_images(
    image_bgr,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
    version_root: Path,
    max_gap: int,
) -> None:
    gap_dir = version_root / "gap"
    gap_dir.mkdir(parents=True, exist_ok=True)

    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    for g in range(1, max_gap + 1):
        for dx in (-g, 0, g):
            for dy in (-g, 0, g):
                box = (start_x + dx, start_y + dy, start_x + dx + width, start_y + dy + height)
                output_path = gap_dir / f"{dx}_{dy}.jpg"
                pil_image.crop(box).save(output_path)


def save_goal_image(image_bgr, x1: int, y1: int, x2: int, y2: int, version_root: Path) -> Path:
    goal_dir = version_root / "goal"
    goal_dir.mkdir(parents=True, exist_ok=True)
    goal_path = goal_dir / "0_0.jpg"
    cropped = image_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Cropped goal image is empty. Check ROI selection.")
    cv2.imwrite(str(goal_path), cropped)
    return goal_path


def load_image(input_path: str | Path):
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {input_path}")
    return image
