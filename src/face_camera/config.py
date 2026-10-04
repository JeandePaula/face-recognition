"""Validated TOML configuration, with paths relative to the configuration file."""

import math
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    source: str | int = 0
    known_faces_dir: Path = Path("known_faces")
    models_dir: Path = Path("models")
    output_dir: Path = Path("detected_faces_output")
    headless: bool = True
    resize_factor: float = 0.5
    detection_threshold: float = 0.9
    match_threshold: float = 0.363
    snapshot_cooldown: float = 10.0
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if (
            type(self.source) not in (str, int)
            or (isinstance(self.source, int) and self.source < 0)
            or (isinstance(self.source, str) and not self.source.strip())
        ):
            raise ValueError("source must be a nonempty URL/path or a nonnegative camera index")
        if type(self.headless) is not bool:
            raise ValueError("headless must be true or false")
        for key, low, high in (
            ("resize_factor", 0, 1),
            ("detection_threshold", 0, 1),
            ("match_threshold", 0, 1),
        ):
            value = getattr(self, key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low < value <= high
            ):
                raise ValueError(f"{key} must be greater than {low} and at most {high}")
        for key in ("snapshot_cooldown", "reconnect_delay"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        for key, minimum in (("reconnect_attempts", 0), ("timeout_ms", 1)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")


def load_settings(path: Path | None = None, **overrides: object) -> Settings:
    data = {}
    base = Path.cwd()
    if path is not None:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
        base = path.resolve().parent
    allowed = {field.name for field in fields(Settings)}
    unknown = data.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    source = os.getenv("FACE_SOURCE") or os.getenv("IP_CAMERA_URL")
    if source:
        data["source"] = int(source) if source.isdecimal() else source
    if "DISABLE_GUI" in os.environ:
        value = os.environ["DISABLE_GUI"].lower()
        if value not in ("true", "false"):
            raise ValueError("DISABLE_GUI must be true or false")
        data["headless"] = value == "true"
    data.update({key: value for key, value in overrides.items() if value is not None})
    for key in ("known_faces_dir", "models_dir", "output_dir"):
        value = data.get(key, str(getattr(Settings(), key)))
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{key} must be a nonempty path")
        data[key] = (base / Path(value).expanduser()).resolve()
    source = data.get("source", 0)
    if isinstance(source, str) and source.strip() and "://" not in source:
        data["source"] = str((base / Path(source).expanduser()).resolve())
    return Settings(**data)
