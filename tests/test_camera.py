from contextlib import closing
from threading import Event
from unittest.mock import Mock

import pytest

from face_camera.camera import frames, open_capture
from face_camera.config import Settings


def capture(reads=(), opened=True):
    result = Mock()
    result.isOpened.return_value = opened
    result.read.side_effect = reads
    return result


def test_video_eof_does_not_reconnect(frame, monkeypatch):
    video = capture([(True, frame), (False, None)])
    factory = Mock(return_value=video)
    monkeypatch.setattr("face_camera.camera.open_capture", factory)
    assert len(list(frames(Settings(source="movie.avi"), Event()))) == 1
    factory.assert_called_once()
    video.release.assert_called_once()


def test_network_reconnect_and_close(frame, monkeypatch):
    first = capture([(True, frame), (False, None)])
    second = capture([(True, frame)])
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr("face_camera.camera.open_capture", factory)
    with closing(
        frames(Settings(source="http://camera/video", reconnect_delay=0), Event())
    ) as stream:
        assert next(stream) is frame
        assert next(stream) is frame
    assert factory.call_count == 2
    first.release.assert_called_once()
    second.release.assert_called_once()


def test_retries_are_bounded(monkeypatch):
    attempts = [capture(opened=False) for _ in range(3)]
    factory = Mock(side_effect=attempts)
    monkeypatch.setattr("face_camera.camera.open_capture", factory)
    with pytest.raises(OSError, match="exhausted"):
        list(frames(Settings(reconnect_attempts=2, reconnect_delay=0), Event()))
    assert factory.call_count == 3
    for attempt in attempts:
        attempt.release.assert_called_once()


def test_success_resets_retry_budget(frame, monkeypatch):
    attempts = [
        capture(opened=False),
        capture([(True, frame), (False, None)]),
        capture([(True, frame)]),
    ]
    factory = Mock(side_effect=attempts)
    monkeypatch.setattr("face_camera.camera.open_capture", factory)
    with closing(frames(Settings(reconnect_attempts=1, reconnect_delay=0), Event())) as stream:
        assert next(stream) is frame
        assert next(stream) is frame
    assert factory.call_count == 3


def test_read_exception_releases_capture(monkeypatch):
    video = capture()
    video.read.side_effect = RuntimeError("read failed")
    monkeypatch.setattr("face_camera.camera.open_capture", lambda _: video)
    with pytest.raises(RuntimeError, match="read failed"):
        list(frames(Settings(), Event()))
    video.release.assert_called_once()


def test_stop_during_reconnect_does_not_open_again(monkeypatch):
    stop = Event()
    video = capture(opened=False)
    factory = Mock(return_value=video)
    monkeypatch.setattr("face_camera.camera.open_capture", factory)
    monkeypatch.setattr(stop, "wait", lambda _: stop.set())
    assert list(frames(Settings(), stop)) == []
    factory.assert_called_once()


@pytest.mark.parametrize("opened", [True, False])
def test_unreadable_file_is_error(opened, monkeypatch):
    video = capture([(False, None)], opened=opened)
    monkeypatch.setattr("face_camera.camera.open_capture", lambda _: video)
    with pytest.raises(OSError):
        list(frames(Settings(source="bad.avi"), Event()))
    video.release.assert_called_once()


def test_network_timeout_parameters(monkeypatch):
    import cv2

    factory = Mock()
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    open_capture(Settings(source="rtsp://camera/video", timeout_ms=1234))
    factory.assert_called_once_with(
        "rtsp://camera/video",
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            1234,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            1234,
        ],
    )
