from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .config import ensure_output_layout, load_config
from .dataset_prep.crop_fixed import prepare_dataset_fixed
from .dataset_prep.crop_free import prepare_dataset_free
from .jacobian_modeling.build import build_jacobian
from .jacobian_modeling.evaluate import evaluate_jacobian
from .jacobian_modeling.optimize import optimize_target
from .tools.grayscale import convert_to_grayscale
from .tools.image_diff import show_rgb_difference
from .tracking_runtime.basler_tracker import run_basler_tracking
from .tracking_runtime.video_tracker import run_video_tracking

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _load_runtime_config(config_path: str | None) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path else _default_config_path()
    cfg = load_config(path)
    ensure_output_layout(cfg["paths"]["outputs_root"])
    return cfg


def _run_offline_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    data_root = cfg["paths"]["data_root"]
    outputs_root = cfg["paths"]["outputs_root"]
    version = cfg["experiment"]["version"]

    jacobian_path = build_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version=version,
        sample_gap=int(cfg["jacobian"]["range"]),
        reset=True,
    )

    normal_half, normal_full = evaluate_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version=version,
        eval_type="normal",
        radius=int(cfg["evaluation"]["radius"]),
    )

    opt_result = optimize_target(
        data_root=data_root,
        outputs_root=outputs_root,
        version=version,
        max_opt=int(cfg["optimization"]["max_offset"]),
        iterations=int(cfg["optimization"]["iterations"]),
        learning_rate=cfg["optimization"].get("learning_rate"),
    )

    opt_half, opt_full = evaluate_jacobian(
        data_root=data_root,
        outputs_root=outputs_root,
        version=version,
        eval_type="optimized",
        radius=int(cfg["evaluation"]["radius"]),
    )

    return {
        "jacobian_path": jacobian_path,
        "normal_half": normal_half,
        "normal_full": normal_full,
        "opt_result": opt_result,
        "opt_half": opt_half,
        "opt_full": opt_full,
    }


def _resolve_tracking_video_inputs(
    cfg: dict[str, Any],
    version: str,
    video_override: str | None,
    target_override: str | None,
    jacobian_override: str | None,
) -> tuple[Path, Path, Path]:
    data_root = Path(cfg["paths"]["data_root"])
    outputs_root = Path(cfg["paths"]["outputs_root"])
    tracking_cfg = cfg.get("tracking", {})

    default_video = data_root / "videos" / f"{version}_sample.mp4"
    default_target = data_root / version / "goal" / "0_0.jpg"
    default_jacobian = outputs_root / "jacobian" / version / "jacobian.pkl"

    video_path = (
        Path(video_override).resolve()
        if video_override
        else Path(tracking_cfg.get("video_path") or default_video).resolve()
    )
    target_path = (
        Path(target_override).resolve()
        if target_override
        else Path(tracking_cfg.get("target_image") or default_target).resolve()
    )
    jacobian_path = (
        Path(jacobian_override).resolve()
        if jacobian_override
        else Path(tracking_cfg.get("jacobian_path") or default_jacobian).resolve()
    )
    return video_path, target_path, jacobian_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual servo tracker CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser("offline-run", help="Run Jacobian build + evaluate + optimize pipeline")
    offline.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")

    offline_track = sub.add_parser(
        "offline-track-video",
        help="Run offline pipeline and then start video tracking",
    )
    offline_track.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config")
    offline_track.add_argument("--video", type=str, default=None, help="Video path (optional, fallback to config)")
    offline_track.add_argument("--target", type=str, default=None, help="Target image path (optional, fallback to config)")
    offline_track.add_argument("--jacobian", type=str, default=None, help="Jacobian path (optional, fallback to config)")
    offline_track.add_argument("--iterations-per-frame", type=int, default=None)

    prep_fixed = sub.add_parser("prep-fixed", help="Prepare dataset by fixed ROI size")
    prep_fixed.add_argument("--config", type=str, default=None)
    prep_fixed.add_argument("--input-image", type=str, required=True)
    prep_fixed.add_argument("--width", type=int, required=True)
    prep_fixed.add_argument("--height", type=int, required=True)
    prep_fixed.add_argument("--max-gap", type=int, required=True)
    prep_fixed.add_argument("--version", type=str, default=None)

    prep_free = sub.add_parser("prep-free", help="Prepare dataset by free rectangle ROI")
    prep_free.add_argument("--config", type=str, default=None)
    prep_free.add_argument("--input-image", type=str, required=True)
    prep_free.add_argument("--max-gap", type=int, required=True)
    prep_free.add_argument("--version", type=str, default=None)

    build = sub.add_parser("build-jacobian", help="Build jacobian from dataset")
    build.add_argument("--config", type=str, default=None)
    build.add_argument("--version", type=str, default=None)
    build.add_argument("--range", dest="sample_gap", type=int, default=None)

    evaluate = sub.add_parser("evaluate", help="Generate jacobian evaluation plots")
    evaluate.add_argument("--config", type=str, default=None)
    evaluate.add_argument("--version", type=str, default=None)
    evaluate.add_argument("--type", dest="eval_type", choices=["normal", "optimized", "wide"], default="normal")
    evaluate.add_argument("--radius", type=int, default=None)

    optimize = sub.add_parser("optimize", help="Optimize target image")
    optimize.add_argument("--config", type=str, default=None)
    optimize.add_argument("--version", type=str, default=None)
    optimize.add_argument("--max-opt", type=int, default=None)
    optimize.add_argument("--iterations", type=int, default=None)
    optimize.add_argument("--learning-rate", type=float, default=None)

    track_video = sub.add_parser("track-video", help="Track object in a video")
    track_video.add_argument("--config", type=str, default=None)
    track_video.add_argument("--video", type=str, default=None, help="Video path (optional, fallback to config)")
    track_video.add_argument("--target", type=str, default=None, help="Target image path (optional, fallback to config)")
    track_video.add_argument("--jacobian", type=str, default=None, help="Jacobian path (optional, fallback to config)")
    track_video.add_argument("--iterations-per-frame", type=int, default=None)

    track_basler = sub.add_parser("track-basler", help="Track object with Basler camera")
    track_basler.add_argument("--config", type=str, default=None)
    track_basler.add_argument("--target", type=str, required=True)
    track_basler.add_argument("--jacobian", type=str, required=True)
    track_basler.add_argument("--iterations-per-frame", type=int, default=None)

    grayscale = sub.add_parser("grayscale", help="Convert image to grayscale")
    grayscale.add_argument("--input", type=str, required=True)
    grayscale.add_argument("--output", type=str, required=True)

    diff = sub.add_parser("image-diff", help="Display image difference")
    diff.add_argument("--image1", type=str, required=True)
    diff.add_argument("--image2", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "grayscale":
        out = convert_to_grayscale(args.input, args.output)
        LOGGER.info("Grayscale image saved: %s", out)
        return 0

    if args.command == "image-diff":
        show_rgb_difference(args.image1, args.image2)
        return 0

    cfg = _load_runtime_config(getattr(args, "config", None))
    data_root = cfg["paths"]["data_root"]
    outputs_root = cfg["paths"]["outputs_root"]
    version = args.version if hasattr(args, "version") and args.version else cfg["experiment"]["version"]

    if args.command == "offline-run":
        result = _run_offline_pipeline(cfg)
        LOGGER.info("Offline pipeline completed.")
        LOGGER.info("Jacobian: %s", result["jacobian_path"])
        LOGGER.info("Optimized result: %s", result["opt_result"]["result_path"])
        LOGGER.info(
            "Best learning rate: %s / Best loss: %s",
            result["opt_result"]["best_learning_rate"],
            result["opt_result"]["best_loss"],
        )
        return 0

    if args.command == "offline-track-video":
        result = _run_offline_pipeline(cfg)
        iter_pf = (
            args.iterations_per_frame
            if args.iterations_per_frame is not None
            else int(cfg["tracking"]["iterations_per_frame"])
        )
        video_path, target_path, jacobian_path = _resolve_tracking_video_inputs(
            cfg,
            version=cfg["experiment"]["version"],
            video_override=args.video,
            target_override=args.target,
            jacobian_override=args.jacobian,
        )
        LOGGER.info("Offline pipeline completed. Starting video tracking...")
        LOGGER.info("video=%s target=%s jacobian=%s", video_path, target_path, jacobian_path)
        run_video_tracking(video_path, target_path, jacobian_path, iterations_per_frame=iter_pf)
        LOGGER.info("Video tracking completed.")
        LOGGER.info("Optimized result: %s", result["opt_result"]["result_path"])
        return 0

    if args.command == "prep-fixed":
        goal = prepare_dataset_fixed(
            input_image_path=args.input_image,
            data_root=data_root,
            version=version,
            width=args.width,
            height=args.height,
            max_gap=args.max_gap,
        )
        LOGGER.info("Prepared dataset goal image: %s", goal)
        return 0

    if args.command == "prep-free":
        goal = prepare_dataset_free(
            input_image_path=args.input_image,
            data_root=data_root,
            version=version,
            max_gap=args.max_gap,
        )
        LOGGER.info("Prepared dataset goal image: %s", goal)
        return 0

    if args.command == "build-jacobian":
        sample_gap = args.sample_gap if args.sample_gap is not None else int(cfg["jacobian"]["range"])
        out = build_jacobian(data_root, outputs_root, version, sample_gap=sample_gap, reset=True)
        LOGGER.info("Jacobian built: %s", out)
        return 0

    if args.command == "evaluate":
        radius = args.radius if args.radius is not None else int(cfg["evaluation"]["radius"])
        half, full = evaluate_jacobian(data_root, outputs_root, version, args.eval_type, radius)
        LOGGER.info("Evaluation plots: %s / %s", half, full)
        return 0

    if args.command == "optimize":
        max_opt = args.max_opt if args.max_opt is not None else int(cfg["optimization"]["max_offset"])
        iterations = args.iterations if args.iterations is not None else int(cfg["optimization"]["iterations"])
        learning_rate = args.learning_rate if args.learning_rate is not None else cfg["optimization"].get("learning_rate")
        result = optimize_target(data_root, outputs_root, version, max_opt=max_opt, iterations=iterations, learning_rate=learning_rate)
        LOGGER.info("Optimized image saved: %s", result["result_path"])
        return 0

    if args.command == "track-video":
        iter_pf = (
            args.iterations_per_frame
            if args.iterations_per_frame is not None
            else int(cfg["tracking"]["iterations_per_frame"])
        )
        video_path, target_path, jacobian_path = _resolve_tracking_video_inputs(
            cfg,
            version=version,
            video_override=args.video,
            target_override=args.target,
            jacobian_override=args.jacobian,
        )
        LOGGER.info("video=%s target=%s jacobian=%s", video_path, target_path, jacobian_path)
        run_video_tracking(video_path, target_path, jacobian_path, iterations_per_frame=iter_pf)
        return 0

    if args.command == "track-basler":
        iter_pf = (
            args.iterations_per_frame
            if args.iterations_per_frame is not None
            else int(cfg["tracking"]["iterations_per_frame"])
        )
        run_basler_tracking(args.target, args.jacobian, iterations_per_frame=iter_pf)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
