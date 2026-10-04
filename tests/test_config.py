from pathlib import Path

import pytest

from face_camera.config import Settings, load_settings


def test_defaults_are_headless(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.headless
    assert settings.source == 0
    assert settings.models_dir == tmp_path / "models"


def test_relative_paths_and_precedence(tmp_path, monkeypatch):
    path = tmp_path / "camera.toml"
    path.write_text('source = "video.mp4"\nheadless = false\nmodels_dir = "weights"')
    settings = load_settings(path)
    assert settings.source == str(tmp_path / "video.mp4")
    assert settings.models_dir == tmp_path / "weights"
    monkeypatch.setenv("IP_CAMERA_URL", "http://legacy/video")
    monkeypatch.setenv("FACE_SOURCE", "http://preferred/video")
    monkeypatch.setenv("DISABLE_GUI", "true")
    assert load_settings(path).source == "http://preferred/video"
    settings = load_settings(path, source=2, headless=False)
    assert settings.source == 2
    assert not settings.headless


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": -1},
        {"source": True},
        {"source": ""},
        {"source": []},
        {"headless": "false"},
        {"resize_factor": 0},
        {"resize_factor": 2},
        {"match_threshold": float("nan")},
        {"detection_threshold": True},
        {"snapshot_cooldown": -1},
        {"reconnect_delay": float("inf")},
        {"reconnect_attempts": 1.2},
        {"reconnect_attempts": False},
        {"timeout_ms": 0},
    ],
)
def test_reject_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_unknown_config_key_is_error(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("typo = true")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_settings(path)


def test_missing_explicit_config_is_error():
    with pytest.raises(FileNotFoundError):
        load_settings(Path("missing-config.toml"))


def test_reject_non_path_value(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("models_dir = 42")
    with pytest.raises(ValueError, match="models_dir"):
        load_settings(path)


def test_invalid_environment_boolean(monkeypatch):
    monkeypatch.setenv("DISABLE_GUI", "maybe")
    with pytest.raises(ValueError, match="DISABLE_GUI"):
        load_settings()


@pytest.mark.parametrize("source", ["", "   "])
def test_empty_source_is_rejected_before_path_resolution(tmp_path, source):
    path = tmp_path / "settings.toml"
    path.write_text(f'source = "{source}"')
    with pytest.raises(ValueError, match="source"):
        load_settings(path)
