import os
import shutil
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from module_train_face import train_face_recognition

def setup_test_environment():
    os.makedirs("my_faces", exist_ok=True)
    os.makedirs(os.path.join("my_faces", "person1"), exist_ok=True)
    os.makedirs(os.path.join("my_faces", "person2"), exist_ok=True)

    dummy_face = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    for i in range(2):
        cv2.imwrite(os.path.join("my_faces", "person1", f"face_{i}.jpg"), dummy_face)
        cv2.imwrite(os.path.join("my_faces", "person2", f"face_{i}.jpg"), dummy_face)

def teardown_test_environment():
    shutil.rmtree("my_faces", ignore_errors=True)

@patch('module_train_face.FaceNet')
@patch('module_train_face.cv2.CascadeClassifier')
@patch('module_train_face.os.path.exists', return_value=True)
def test_train_face_recognition_success(mock_exists, mock_detector, mock_face_net):
    setup_test_environment()
    mock_detector.return_value.detectMultiScale.return_value = [(0, 0, 100, 100)]
    mock_embedder = MagicMock()
    mock_embedder.embeddings.return_value = [np.random.rand(512)]
    mock_face_net.return_value = mock_embedder

    def fake_listdir(path):
        if "person1" in path or "person2" in path:
            return ["face_0.jpg", "face_1.jpg"]
        return ["person1", "person2"]

    with patch('module_train_face.os.listdir', side_effect=fake_listdir), \
         patch('module_train_face.os.path.isdir', return_value=True), \
         patch('sklearn.calibration.CalibratedClassifierCV.fit', return_value=None), \
         patch('sklearn.calibration.CalibratedClassifierCV.score', return_value=1.0):
        result = train_face_recognition()

    teardown_test_environment()
    assert result == 0
