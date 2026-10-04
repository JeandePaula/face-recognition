"""Explicit, atomic downloads of pinned OpenCV Zoo models, verified with SHA-256."""

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)
REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"


@dataclass(frozen=True)
class Model:
    directory: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}"
            f"/models/{self.directory}/{self.filename}"
        )


DETECTOR = Model(
    "face_detection_yunet",
    "face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
RECOGNIZER = Model(
    "face_recognition_sface",
    "face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
MODELS = (DETECTOR, RECOGNIZER)


def verified(path: Path, model: Model) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == model.sha256


def require_models(directory: Path) -> tuple[Path, Path]:
    for model in MODELS:
        if not verified(directory / model.filename, model):
            raise ValueError(
                f"Missing or corrupted model: {directory / model.filename}. "
                "Run face-camera download-models --models-dir <directory> first."
            )
    return directory / DETECTOR.filename, directory / RECOGNIZER.filename


def download_models(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        target = directory / model.filename
        if verified(target, model):
            LOGGER.info("Model already verified: %s", model.filename)
            continue
        LOGGER.info("Downloading %s", model.filename)
        temporary = None
        try:
            with (
                urlopen(model.url, timeout=30) as response,
                tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as output,
            ):
                temporary = Path(output.name)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if not verified(temporary, model):
                raise ValueError(f"SHA-256 verification failed for {model.filename}")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
