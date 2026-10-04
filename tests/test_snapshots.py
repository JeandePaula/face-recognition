import cv2
import pytest

from face_camera.recognition import Detection
from face_camera.snapshots import SnapshotWriter


def detected(name="Ana"):
    return Detection(name, (0, 0, 10, 10), 0.9)


def test_arrival_disappearance_and_pending_cooldown(tmp_path, frame, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("face_camera.snapshots.time.monotonic", lambda: clock[0])
    writer = SnapshotWriter(tmp_path, 10)
    paths = writer.save(frame, [detected()])
    assert len(paths) == 1
    assert cv2.imread(str(paths[0])).shape == frame.shape
    clock[0] = 20
    assert writer.save(frame, [detected()]) == []  # A continuous presence is not periodic.
    writer.save(frame, [])
    assert len(writer.save(frame, [detected()])) == 1
    clock[0] = 21
    writer.save(frame, [])
    assert writer.save(frame, [detected()]) == []
    clock[0] = 30
    assert len(writer.save(frame, [detected()])) == 1  # Pending reappearance is saved.


def test_unknown_group_is_distinct_from_person_named_unknown(tmp_path, frame):
    writer = SnapshotWriter(tmp_path, 0)
    paths = writer.save(frame, [detected(None), detected(None), detected("Unknown")])
    assert len(paths) == 2
    assert any(path.name.startswith("known-") for path in paths)
    assert any(path.name.startswith("unknown-") for path in paths)


def test_safe_unique_filenames(tmp_path, frame):
    writer = SnapshotWriter(tmp_path, 0)
    first = writer.save(frame, [detected("../../person : /\\?")])[0]
    writer.save(frame, [])
    second = writer.save(frame, [detected("../../person : /\\?")])[0]
    assert first.parent == second.parent == tmp_path
    assert first != second


def test_failed_write_is_error_and_can_be_retried(tmp_path, frame, monkeypatch):
    writer = SnapshotWriter(tmp_path, 0)
    real_write = cv2.imwrite
    monkeypatch.setattr(cv2, "imwrite", lambda *_: False)
    with pytest.raises(OSError, match="Could not write"):
        writer.save(frame, [detected()])
    assert writer.last_saved == {}
    monkeypatch.setattr(cv2, "imwrite", real_write)
    assert len(writer.save(frame, [detected()])) == 1
