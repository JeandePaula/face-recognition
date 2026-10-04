from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from face_camera.config import Settings
from face_camera.recognition import Detection, FaceEngine, annotate, best_match


def test_best_match_uses_highest_score_and_threshold():
    gallery = [("first", np.array([0.6, 0.8])), ("best", np.array([1.0, 0.0]))]
    assert best_match(np.array([1.0, 0.0]), gallery, 0.9) == ("best", 1.0)
    assert best_match(np.array([-1.0, 0.0]), gallery, 0.9)[0] is None
    assert best_match(np.array([1.0, 0.0]), [], 0.9) == (None, None)


def test_boxes_and_landmarks_use_actual_resize_dimensions():
    engine = FaceEngine.__new__(FaceEngine)
    engine.detector = Mock()
    face = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0.99]], dtype=np.float32)
    engine.detector.detect.return_value = (1, face)
    frame = np.zeros((101, 203, 3), dtype=np.uint8)
    result = engine.detect(frame, 0.5)[0]
    engine.detector.setInputSize.assert_called_once_with((102, 50))
    np.testing.assert_allclose(result[:14:2], face[0, :14:2] * 203 / 102)
    np.testing.assert_allclose(result[1:14:2], face[0, 1:14:2] * 101 / 50)
    assert result[14] == face[0, 14]


@pytest.mark.parametrize("count", [0, 2])
def test_gallery_rejects_ambiguous_or_missing_faces(tmp_path, frame, count):
    cv2.imwrite(str(tmp_path / "Ana.jpg"), frame)
    engine = FaceEngine.__new__(FaceEngine)
    engine.gallery = []
    engine.detect = Mock(return_value=np.zeros((count, 15)))
    with pytest.raises(ValueError, match="exactly one face"):
        engine.load_gallery(tmp_path)


def test_gallery_reads_name_from_directory_and_accepts_multiple_references(tmp_path, frame):
    (tmp_path / "Ana").mkdir()
    for filename in ("one.jpg", "two.PNG"):
        cv2.imwrite(str(tmp_path / "Ana" / filename), frame)
    engine = FaceEngine.__new__(FaceEngine)
    engine.gallery = []
    engine.detect = Mock(return_value=np.zeros((1, 15)))
    engine.feature = Mock(return_value=np.array([1.0]))
    assert engine.load_gallery(tmp_path) == 2
    assert [name for name, _ in engine.gallery] == ["Ana", "Ana"]


def test_gallery_rejects_corrupt_image(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    engine = FaceEngine.__new__(FaceEngine)
    engine.gallery = []
    with pytest.raises(ValueError, match="Cannot read"):
        engine.load_gallery(tmp_path)


def test_recognition_clamps_boxes(frame):
    engine = FaceEngine.__new__(FaceEngine)
    engine.settings = Settings()
    engine.gallery = []
    engine.detect = Mock(return_value=np.array([[-5, -7, 999, 999] + [0] * 11]))
    engine.feature = Mock(return_value=np.array([1.0]))
    detection = engine.recognize(frame)[0]
    assert detection.box == (0, 0, 160, 120)
    assert detection.name is None


def test_annotation_does_not_change_original(frame):
    original = frame.copy()
    result = annotate(frame, [Detection("Ana", (20, 20, 40, 40), 0.9)])
    np.testing.assert_array_equal(original, frame)
    assert np.any(result != frame)
