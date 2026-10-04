import numpy as np
import pytest


@pytest.fixture
def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ("FACE_SOURCE", "IP_CAMERA_URL", "DISABLE_GUI"):
        monkeypatch.delenv(key, raising=False)
