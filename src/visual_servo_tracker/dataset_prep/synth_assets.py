from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.strip()
    if not color.startswith("#") or len(color) != 7:
        raise ValueError("Color must be like #RRGGBB")
    return tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (4, 2, 0))


def generate_spiral_video(
    out_path: str | Path,
    shape: str,
    bg_color: str,
    shape_color: str,
    width: int = 1000,
    height: int = 700,
    shape_size: int = 100,
    fps: int = 30,
    duration_sec: int = 10,
) -> Path:
    shape = shape.lower().strip()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg_bgr = _hex_to_bgr(bg_color)
    shape_bgr = _hex_to_bgr(shape_color)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    total_frames = fps * duration_sec
    center_x, center_y = width // 2, height // 2
    max_radius = min(width, height) // 2 - shape_size // 2 - 100
    frequency = 0.05

    for i in range(total_frames):
        t = i / total_frames
        radius = max_radius * math.sin(math.pi * t)
        angle = 2 * math.pi * frequency * i
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = bg_bgr

        if shape == "circle":
            cv2.circle(frame, (x, y), shape_size // 2, shape_bgr, -1)
        elif shape == "square":
            cv2.rectangle(frame, (x - shape_size // 2, y - shape_size // 2), (x + shape_size // 2, y + shape_size // 2), shape_bgr, -1)
        elif shape == "rectangle":
            rect_h = shape_size // 2
            cv2.rectangle(frame, (x - shape_size // 2, y - rect_h // 2), (x + shape_size // 2, y + rect_h // 2), shape_bgr, -1)
        else:
            writer.release()
            raise ValueError("shape must be circle/square/rectangle")

        writer.write(frame)

    writer.release()
    return out_path


def generate_center_image(
    out_path: str | Path,
    shape: str,
    bg_color: str,
    shape_color: str,
    width: int = 500,
    height: int = 500,
    shape_size: int = 100,
) -> Path:
    shape = shape.lower().strip()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    left = (width - shape_size) // 2
    top = (height - shape_size) // 2
    right = left + shape_size
    bottom = top + shape_size

    if shape == "circle":
        draw.ellipse([left, top, right, bottom], fill=shape_color)
    elif shape == "square":
        draw.rectangle([left, top, right, bottom], fill=shape_color)
    elif shape == "rectangle":
        rect_h = shape_size // 2
        top = (height - rect_h) // 2
        bottom = top + rect_h
        draw.rectangle([left, top, right, bottom], fill=shape_color)
    else:
        raise ValueError("shape must be circle/square/rectangle")

    image.save(out_path)
    return out_path
