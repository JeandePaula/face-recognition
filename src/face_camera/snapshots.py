"""Save original frames when a label appears, with bounded snapshot frequency."""

import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2

from face_camera.recognition import Detection, Frame

LOGGER = logging.getLogger(__name__)


class SnapshotWriter:
    def __init__(self, directory: Path, cooldown: float) -> None:
        self.directory = directory
        self.cooldown = cooldown
        self.previous: set[str | None] = set()
        self.last_saved: dict[str | None, float] = {}
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, frame: Frame, detections: list[Detection]) -> list[Path]:
        now = time.monotonic()
        current = {detection.name for detection in detections}
        saved = []
        saved_names = set()
        for name in sorted(current, key=lambda value: (value is not None, value or "")):
            if (
                name in self.previous
                or now - self.last_saved.get(name, float("-inf")) < self.cooldown
            ):
                continue
            safe_name = re.sub(r"[^\w-]+", "_", name or "unknown", flags=re.ASCII).strip("_")[:60]
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
            kind = "known" if name is not None else "unknown"
            path = self.directory / f"{kind}-{safe_name}-{timestamp}-{uuid4().hex[:8]}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Could not write snapshot: {path}")
            self.last_saved[name] = now
            saved_names.add(name)
            saved.append(path)
            LOGGER.info("Saved snapshot: %s", path.name)
        # Suppressed reappearances remain pending until the cooldown has elapsed.
        self.previous = current & (self.previous | saved_names)
        return saved
