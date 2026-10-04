"""YuNet detection and SFace recognition; all inference stays on this machine."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from face_camera.config import Settings
from face_camera.models import require_models

LOGGER = logging.getLogger(__name__)
type Frame = NDArray[np.uint8]
type Feature = NDArray[np.float32]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Detection:
    name: str | None
    box: tuple[int, int, int, int]  # x, y, width, height in original frame coordinates
    similarity: float | None


def best_match(
    feature: Feature, gallery: list[tuple[str, Feature]], threshold: float
) -> tuple[str | None, float | None]:
    """Compare normalized embeddings by cosine similarity, keeping the best match."""
    if not gallery:
        return None, None
    scores = [float(np.dot(feature.ravel(), candidate.ravel())) for _, candidate in gallery]
    index = int(np.argmax(scores))
    score = scores[index]
    return (gallery[index][0] if score >= threshold else None), score


class FaceEngine:
    def __init__(self, settings: Settings) -> None:
        detector, recognizer = require_models(settings.models_dir)
        self.settings = settings
        self.detector = cv2.FaceDetectorYN.create(
            str(detector), "", (320, 320), settings.detection_threshold
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        self.gallery: list[tuple[str, Feature]] = []

    def detect(self, frame: Frame, scale: float = 1.0) -> Feature:
        height, width = frame.shape[:2]
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = cv2.resize(frame, size) if size != (width, height) else frame
        self.detector.setInputSize(size)
        _, faces = self.detector.detect(small)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        faces = faces.copy()
        # Scale boxes AND all five landmarks, using actual rounded dimensions.
        faces[:, [0, 2, 4, 6, 8, 10, 12]] *= width / size[0]
        faces[:, [1, 3, 5, 7, 9, 11, 13]] *= height / size[1]
        return faces

    def feature(self, frame: Frame, face: Feature) -> Feature:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned).copy().ravel()
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The recognizer produced an invalid face embedding")
        return feature / norm

    def load_gallery(self, directory: Path) -> int:
        self.gallery.clear()
        directory.mkdir(parents=True, exist_ok=True)
        # One image named Alice.jpg, or multiple images inside Alice/.
        paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in paths:
            relative = path.relative_to(directory)
            name = path.stem if len(relative.parts) == 1 else relative.parts[0]
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Cannot read gallery image: {path}")
            faces = self.detect(frame)
            if len(faces) != 1:
                raise ValueError(
                    f"Gallery image must contain exactly one face: {path} ({len(faces)})"
                )
            self.gallery.append((name, self.feature(frame, faces[0])))
        LOGGER.info("Loaded %d reference images", len(self.gallery))
        if not self.gallery:
            LOGGER.warning("Gallery is empty; all faces will be marked Unknown")
        return len(self.gallery)

    def recognize(self, frame: Frame) -> list[Detection]:
        result = []
        height, width = frame.shape[:2]
        for face in self.detect(frame, self.settings.resize_factor):
            name, score = best_match(
                self.feature(frame, face), self.gallery, self.settings.match_threshold
            )
            x, y, w, h = face[:4]
            left, top = max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))
            right, bottom = (
                min(width, max(left + 1, round(x + w))),
                min(height, max(top + 1, round(y + h))),
            )
            result.append(Detection(name, (left, top, right - left, bottom - top), score))
        return result


def annotate(frame: Frame, detections: list[Detection]) -> Frame:
    result = frame.copy()
    for detection in detections:
        x, y, width, height = detection.box
        color = (70, 210, 80) if detection.name is not None else (60, 80, 240)
        label = detection.name if detection.name is not None else "Unknown"
        # Hershey fonts only support ASCII; original Unicode names remain in the gallery.
        label = label.encode("ascii", errors="replace").decode()
        cv2.rectangle(result, (x, y), (x + width, y + height), color, 2)
        cv2.putText(result, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return result
