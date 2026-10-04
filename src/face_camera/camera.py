"""Video lifecycle, bounded reconnection and predictable resource cleanup."""

import logging
from collections.abc import Iterator
from threading import Event

import cv2

from face_camera.config import Settings
from face_camera.recognition import Frame

LOGGER = logging.getLogger(__name__)


def open_capture(settings: Settings) -> cv2.VideoCapture:
    if isinstance(settings.source, str) and "://" in settings.source:
        return cv2.VideoCapture(
            settings.source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                settings.timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                settings.timeout_ms,
            ],
        )
    return cv2.VideoCapture(settings.source)


def frames(settings: Settings, stop: Event) -> Iterator[Frame]:
    is_file = isinstance(settings.source, str) and "://" not in settings.source
    failures = 0
    while not stop.is_set():
        capture = open_capture(settings)
        try:
            if capture.isOpened():
                LOGGER.info("Video source connected")
                read_any = False
                while not stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        if is_file:
                            if not read_any:
                                raise OSError("Video file contains no decodable frames")
                            LOGGER.info("Video file ended")
                            return
                        break
                    read_any = True
                    failures = 0
                    yield frame
            elif is_file:
                raise OSError("Cannot open video file; check its path and codec")
        finally:
            capture.release()
        if stop.is_set():
            return
        if failures >= settings.reconnect_attempts:
            raise OSError("Camera unavailable; reconnection attempts exhausted")
        failures += 1
        LOGGER.warning(
            "Video unavailable; reconnecting (%d/%d)", failures, settings.reconnect_attempts
        )
        stop.wait(settings.reconnect_delay)
