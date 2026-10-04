import hashlib
import io

import pytest

from face_camera import models


@pytest.fixture
def model(monkeypatch):
    model = models.Model("test", "test.onnx", hashlib.sha256(b"valid").hexdigest())
    monkeypatch.setattr(models, "MODELS", (model,))
    return model


def test_download_verify_reuse_and_repair(tmp_path, model, monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(b"valid")

    monkeypatch.setattr(models, "urlopen", fetch)
    models.download_models(tmp_path)
    models.download_models(tmp_path)
    assert len(calls) == 1
    target = tmp_path / model.filename
    target.write_bytes(b"corrupt")
    models.download_models(tmp_path)
    assert len(calls) == 2
    assert models.verified(target, model)
    assert not list(tmp_path.glob("*.part"))


def test_checksum_failure_keeps_original_and_removes_temp(tmp_path, model, monkeypatch):
    target = tmp_path / model.filename
    target.write_bytes(b"original")
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: io.BytesIO(b"invalid"))
    with pytest.raises(ValueError, match="SHA-256"):
        models.download_models(tmp_path)
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.part"))


def test_missing_model_has_actionable_error(tmp_path, model):
    with pytest.raises(ValueError, match="download-models"):
        models.require_models(tmp_path)
