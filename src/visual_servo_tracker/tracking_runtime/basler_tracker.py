from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk

LOGGER = logging.getLogger(__name__)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _fit_scale(img_w: int, img_h: int, max_w: int, max_h: int) -> float:
    if img_w == 0 or img_h == 0:
        return 1.0
    return min(max_w / img_w, max_h / img_h, 1.0)


def _safe_move(name: str, x: int, y: int, w: int, h: int, scr_w: int, scr_h: int) -> None:
    cv2.moveWindow(name, _clamp(x, 0, max(0, scr_w - w)), _clamp(y, 0, max(0, scr_h - h)))


def _rgb_diff(target: np.ndarray, sample: np.ndarray) -> np.ndarray:
    diff = target.astype(np.float32) - sample.astype(np.float32)
    return diff.reshape(-1, 1)


class _RoiState:
    def __init__(self) -> None:
        self.defined = False
        self.start = False
        self.x = 0
        self.y = 0
        self.scale = 1.0
        self.roi_w = 0
        self.roi_h = 0

    def on_mouse(self, event, x, y, flags, param) -> None:
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            fw = param["fw"]
            fh = param["fh"]
            ox = int(x / max(self.scale, 1e-6))
            oy = int(y / max(self.scale, 1e-6))
            self.x = _clamp(ox, 0, max(0, fw - self.roi_w))
            self.y = _clamp(oy, 0, max(0, fh - self.roi_h))
            self.defined = True


def run_basler_tracking(
    target_image_path: str | Path,
    jacobian_path: str | Path,
    iterations_per_frame: int = 10,
    camera_fps: int = 200,
) -> None:
    try:
        from pypylon import pylon
    except ImportError as exc:
        raise ImportError(
            "pypylon is required for Basler tracking. Install with pip install pypylon"
        ) from exc

    target = cv2.imread(str(target_image_path))
    if target is None:
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    with Path(jacobian_path).open("rb") as file_obj:
        jacobian = pickle.load(file_obj)

    tk_root = tk.Tk()
    tk_root.withdraw()
    scr_w = tk_root.winfo_screenwidth()
    scr_h = tk_root.winfo_screenheight()

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.GainAuto.SetValue("Continuous")
    camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(camera_fps)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed

    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if not grab.GrabSucceeded():
        raise RuntimeError("Failed to get initial frame from camera")
    frame = converter.Convert(grab).GetArray()
    fh, fw = frame.shape[:2]

    roi_h, roi_w = target.shape[:2]
    if roi_w > fw or roi_h > fh:
        scale = _fit_scale(roi_w, roi_h, fw, fh)
        target = cv2.resize(target, (int(roi_w * scale), int(roi_h * scale)))
        roi_h, roi_w = target.shape[:2]

    expected_width = roi_w * roi_h * 3
    if jacobian.ndim != 2 or jacobian.shape[0] != 2 or jacobian.shape[1] != expected_width:
        raise ValueError(
            f"Jacobian shape invalid. expected=(2, {expected_width}), got={jacobian.shape}"
        )

    margin = 16
    main_max_w = int(scr_w * 0.60)
    main_max_h = int(scr_h * 0.95)
    side_max_w = scr_w - (margin * 3 + main_max_w)
    side_max_h_each = int((scr_h - margin * 3) / 2)

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sample", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Target", cv2.WINDOW_NORMAL)

    roi_state = _RoiState()
    roi_state.roi_w = roi_w
    roi_state.roi_h = roi_h
    cv2.setMouseCallback("Tracking", roi_state.on_mouse, param={"fw": fw, "fh": fh})

    while not roi_state.start:
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            break
        frame = converter.Convert(grab).GetArray()
        fh, fw = frame.shape[:2]

        roi_state.scale = _fit_scale(fw, fh, main_max_w, main_max_h)
        disp = cv2.resize(frame, (int(fw * roi_state.scale), int(fh * roi_state.scale)))

        if roi_state.defined:
            rx = int(roi_state.x * roi_state.scale)
            ry = int(roi_state.y * roi_state.scale)
            rw = int(roi_w * roi_state.scale)
            rh = int(roi_h * roi_state.scale)
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 159, 40), 2)

        cv2.putText(disp, "Click ROI origin, 's' start, 'q' quit", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Tracking", disp)
        _safe_move("Tracking", margin, margin, disp.shape[1], disp.shape[0], scr_w, scr_h)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and roi_state.defined:
            roi_state.start = True
        elif key == ord("q"):
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()
            return

    x, y = roi_state.x, roi_state.y
    target_scale = _fit_scale(roi_w, roi_h, side_max_w, side_max_h_each)
    target_disp = cv2.resize(target, (int(roi_w * target_scale), int(roi_h * target_scale)))
    cv2.imshow("Target", target_disp)

    while camera.IsGrabbing() and roi_state.start:
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            break

        frame = converter.Convert(grab).GetArray()
        fh, fw = frame.shape[:2]

        for _ in range(iterations_per_frame):
            x = _clamp(x, 0, max(0, fw - roi_w))
            y = _clamp(y, 0, max(0, fh - roi_h))
            sample = frame[y : y + roi_h, x : x + roi_w]
            if sample.shape[:2] != (roi_h, roi_w):
                break
            delta = jacobian @ _rgb_diff(target, sample)
            x -= int(delta[0][0])
            y -= int(delta[1][0])

        x = _clamp(x, 0, max(0, fw - roi_w))
        y = _clamp(y, 0, max(0, fh - roi_h))

        scale = _fit_scale(fw, fh, main_max_w, main_max_h)
        disp = cv2.resize(frame, (int(fw * scale), int(fh * scale)))

        rx, ry = int(x * scale), int(y * scale)
        rw, rh = int(roi_w * scale), int(roi_h * scale)
        cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

        cv2.imshow("Tracking", disp)
        _safe_move("Tracking", margin, margin, disp.shape[1], disp.shape[0], scr_w, scr_h)

        sample = frame[y : y + roi_h, x : x + roi_w]
        sample_scale = _fit_scale(roi_w, roi_h, side_max_w, side_max_h_each)
        sample_disp = cv2.resize(sample, (int(roi_w * sample_scale), int(roi_h * sample_scale)))
        cv2.imshow("Sample", sample_disp)

        right_x = margin * 2 + min(disp.shape[1], main_max_w)
        _safe_move("Sample", right_x, margin, sample_disp.shape[1], sample_disp.shape[0], scr_w, scr_h)
        _safe_move("Target", right_x, margin * 2 + side_max_h_each, target_disp.shape[1], target_disp.shape[0], scr_w, scr_h)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    LOGGER.info("Basler tracking completed.")
