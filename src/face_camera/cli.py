"""Small CLI; configuration and processing logic live in independent modules."""

import argparse
import logging
import os
import signal
import sys
from contextlib import closing
from pathlib import Path
from threading import Event

# Reduce native FFmpeg output, which may otherwise print camera credentials.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2  # noqa: E402

from face_camera.camera import frames  # noqa: E402
from face_camera.config import Settings, load_settings  # noqa: E402
from face_camera.models import download_models  # noqa: E402
from face_camera.recognition import FaceEngine, annotate  # noqa: E402
from face_camera.snapshots import SnapshotWriter  # noqa: E402

LOGGER = logging.getLogger(__name__)


def run(settings: Settings, max_frames: int | None = None) -> int:
    if not settings.headless:
        if sys.platform.startswith("linux") and not (
            os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
        ):
            raise ValueError("No display available; use --headless or configure WSLg/X11")
        if hasattr(cv2, "currentUIFramework") and not cv2.currentUIFramework():
            raise ValueError("OpenCV has no GUI backend; use --headless")
    engine = FaceEngine(settings)
    engine.load_gallery(settings.known_faces_dir)
    writer = SnapshotWriter(settings.output_dir, settings.snapshot_cooldown)
    stop = Event()
    previous_handlers = {}
    processed = 0
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        with closing(frames(settings, stop)) as stream:
            for frame in stream:
                detections = engine.recognize(frame)
                writer.save(frame, detections)
                processed += 1
                if not settings.headless:
                    cv2.imshow("Face Camera - Q to exit", annotate(frame, detections))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if max_frames is not None and processed >= max_frames:
                    break
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not settings.headless:
            cv2.destroyAllWindows()
    LOGGER.info("Finished after %d frames", processed)
    return 0


def camera_source(value: str) -> str | int:
    return int(value) if value.isdecimal() else value


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local face recognition with YuNet and SFace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download-models", help="Download and verify ONNX models")
    download.add_argument("--models-dir", type=Path, default=Path("models"))
    start = subparsers.add_parser("run", help="Recognize faces in a camera stream or video file")
    start.add_argument("--config", type=Path)
    start.add_argument("--source", type=camera_source)
    display = start.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true", default=None)
    display.add_argument("--show", dest="headless", action="store_false")
    start.add_argument("--max-frames", type=positive_int, help="Stop after N frames")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "download-models":
            download_models(args.models_dir)
            return 0
        path = args.config
        if path is None and Path("config.toml").is_file():
            path = Path("config.toml")
        settings = load_settings(path, source=args.source, headless=args.headless)
        return run(settings, args.max_frames)
    except KeyboardInterrupt:
        return 130
    except cv2.error:
        LOGGER.error("OpenCV processing failed; check models, input images and GUI support")
        return 1
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
