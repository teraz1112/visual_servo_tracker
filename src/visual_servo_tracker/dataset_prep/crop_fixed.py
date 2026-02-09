from __future__ import annotations

from pathlib import Path

import cv2

from .crop_common import load_image, save_goal_image, save_shifted_images


class _PointState:
    def __init__(self) -> None:
        self.start_x = -1
        self.start_y = -1
        self.selected = False

    def on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.selected = True


def prepare_dataset_fixed(
    input_image_path: str | Path,
    data_root: str | Path,
    version: str,
    width: int,
    height: int,
    max_gap: int,
) -> Path:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    image = load_image(input_image_path)
    clone = image.copy()
    state = _PointState()

    window_name = "Select top-left, s:save, q:quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, state.on_mouse)

    while True:
        temp = clone.copy()
        if state.selected:
            cv2.rectangle(
                temp,
                (state.start_x, state.start_y),
                (state.start_x + width, state.start_y + height),
                (49, 255, 120),
                2,
            )

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and state.selected:
            x1, y1 = state.start_x, state.start_y
            x2, y2 = x1 + width, y1 + height
            version_root = Path(data_root) / version
            goal_path = save_goal_image(image, x1, y1, x2, y2, version_root)
            save_shifted_images(image, x1, y1, width, height, version_root, max_gap)
            cv2.destroyAllWindows()
            return goal_path

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Dataset preparation cancelled by user.")
