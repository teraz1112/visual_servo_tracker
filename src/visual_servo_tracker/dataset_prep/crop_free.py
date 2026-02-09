from __future__ import annotations

from pathlib import Path

import cv2

from .crop_common import load_image, save_goal_image, save_shifted_images


class _RectState:
    def __init__(self) -> None:
        self.is_drawing = False
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1
        self.bounding_box_set = False

    def on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_x, self.start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.end_x, self.end_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.end_x, self.end_y = x, y
            self.bounding_box_set = True


def prepare_dataset_free(
    input_image_path: str | Path,
    data_root: str | Path,
    version: str,
    max_gap: int,
) -> Path:
    image = load_image(input_image_path)
    clone = image.copy()
    state = _RectState()

    window_name = "Select ROI (drag), s:save, q:quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, state.on_mouse)

    while True:
        temp = clone.copy()
        if state.is_drawing or state.bounding_box_set:
            cv2.rectangle(
                temp,
                (state.start_x, state.start_y),
                (state.end_x, state.end_y),
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and state.bounding_box_set:
            x1, y1 = min(state.start_x, state.end_x), min(state.start_y, state.end_y)
            x2, y2 = max(state.start_x, state.end_x), max(state.start_y, state.end_y)
            version_root = Path(data_root) / version
            goal_path = save_goal_image(image, x1, y1, x2, y2, version_root)
            save_shifted_images(image, x1, y1, x2 - x1, y2 - y1, version_root, max_gap)
            cv2.destroyAllWindows()
            return goal_path

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Dataset preparation cancelled by user.")
