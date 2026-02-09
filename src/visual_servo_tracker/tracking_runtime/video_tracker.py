from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class _RoiState:
    def __init__(self) -> None:
        self.defined = False
        self.x = 0
        self.y = 0

    def on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x, self.y = x, y
            self.defined = True


def _calculate_rgb_difference(target_image: np.ndarray, sample_image: np.ndarray) -> np.ndarray:
    diff = target_image.astype(np.float32) - sample_image.astype(np.float32)
    return diff.reshape(-1, 1)


def run_video_tracking(
    video_path: str | Path,
    target_image_path: str | Path,
    jacobian_path: str | Path,
    iterations_per_frame: int = 10,
) -> None:
    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    with Path(jacobian_path).open("rb") as file_obj:
        jacobian = pickle.load(file_obj)

    roi_h, roi_w = target_image.shape[:2]
    expected_width = roi_w * roi_h * 3
    if jacobian.shape[1] != expected_width:
        raise ValueError(
            f"Jacobian shape mismatch. expected second dim={expected_width}, got={jacobian.shape[1]}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 1

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Unable to read the first frame from the video.")

    roi_state = _RoiState()
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", roi_state.on_mouse)

    x = 0
    y = 0
    while True:
        view = frame.copy()
        if roi_state.defined:
            cv2.rectangle(view, (roi_state.x, roi_state.y), (roi_state.x + roi_w, roi_state.y + roi_h), (255, 159, 40), 2)
        cv2.putText(view, "Click ROI origin, press 's' start, 'q' quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Tracking", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and roi_state.defined:
            x, y = roi_state.x, roi_state.y
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        for _ in range(iterations_per_frame):
            x = max(0, min(x, max(0, w - roi_w)))
            y = max(0, min(y, max(0, h - roi_h)))
            sample_image = frame[y : y + roi_h, x : x + roi_w]
            if sample_image.shape[:2] != (roi_h, roi_w):
                break
            rgb_difference = _calculate_rgb_difference(target_image, sample_image)
            predicted = np.dot(jacobian, rgb_difference)
            x -= int(predicted[0])
            y -= int(predicted[1])

        x = max(0, min(x, max(0, w - roi_w)))
        y = max(0, min(y, max(0, h - roi_h)))

        sample_image = frame[y : y + roi_h, x : x + roi_w]
        cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)
        if sample_image.size:
            cv2.imshow("Sample", sample_image)
        cv2.imshow("Target", target_image)

        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    LOGGER.info("Video tracking completed.")
