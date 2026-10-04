"""Real inference and video tests; never download personal images implicitly."""

import os
import shutil
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import pytest

from face_camera.camera import frames
from face_camera.cli import main
from face_camera.config import Settings
from face_camera.models import MODELS
from face_camera.recognition import FaceEngine

pytestmark = pytest.mark.integration
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture
def engine():
    if not all((MODELS_DIR / model.filename).is_file() for model in MODELS):
        pytest.skip("Run face-camera download-models first")
    return FaceEngine(Settings(models_dir=MODELS_DIR))


def write_video(path, frame):
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (width, height))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(frame)
    finally:
        writer.release()


def test_real_detector_on_blank_frame(engine):
    assert engine.recognize(np.zeros((480, 640, 3), dtype=np.uint8)) == []


def test_real_headless_video_and_eof(engine, tmp_path):
    video = tmp_path / "blank.avi"
    write_video(video, np.zeros((240, 320, 3), dtype=np.uint8))
    config = tmp_path / "config.toml"
    config.write_text(f'models_dir = "{MODELS_DIR.as_posix()}"\n')
    assert main(["run", "--config", str(config), "--source", str(video)]) == 0
    assert list((tmp_path / "detected_faces_output").iterdir()) == []


def test_real_face_enrollment_recognition_and_snapshot(engine, tmp_path):
    path = os.environ.get("FACE_TEST_IMAGE")
    if not path:
        pytest.skip("Set FACE_TEST_IMAGE to a local single-face photo")
    frame = cv2.imread(path)
    assert frame is not None
    unknown = engine.recognize(frame)
    assert len(unknown) == 1
    assert unknown[0].name is None
    gallery = tmp_path / "known_faces"
    gallery.mkdir()
    shutil.copyfile(path, gallery / f"Test Person{Path(path).suffix}")
    assert engine.load_gallery(gallery) == 1
    results = engine.recognize(frame)
    assert len(results) == 1
    assert results[0].name == "Test Person"
    assert results[0].similarity > engine.settings.match_threshold
    video = tmp_path / "face.avi"
    write_video(video, frame)
    config = tmp_path / "config.toml"
    config.write_text(f'models_dir = "{MODELS_DIR.as_posix()}"\n')
    assert main(["run", "--config", str(config), "--source", str(video)]) == 0
    snapshots = list((tmp_path / "detected_faces_output").glob("known-Test_Person-*.jpg"))
    assert len(snapshots) == 1
    assert cv2.imread(str(snapshots[0])).shape == frame.shape


def test_real_mjpeg_stream_reconnects_after_disconnect():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    jpeg = encoded.tobytes()
    connections = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            connections.append(True)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for _ in range(3):
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = Settings(
        source=f"http://127.0.0.1:{server.server_port}/video",
        reconnect_delay=0,
        reconnect_attempts=1,
        timeout_ms=1000,
    )
    try:
        with closing(frames(settings, Event())) as stream:
            for _ in range(5):
                assert next(stream).shape == frame.shape
        assert len(connections) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
