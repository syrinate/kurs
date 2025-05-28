import shutil
import pytest
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
from module_capture_faces import capture_faces


@pytest.fixture
def cleanup():
    yield
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")


@patch('module_capture_faces.cv2.VideoCapture')
@patch('module_capture_faces.cv2.imwrite')
def test_capture_faces_success(mock_imwrite, mock_video_capture, cleanup):
    mock_capture = MagicMock()
    mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_capture

    with patch('module_capture_faces.cv2.CascadeClassifier') as mock_detector:
        mock_detector.return_value.detectMultiScale.return_value = [(50, 50, 100, 100)]
        capture_faces("test_output")

    assert mock_imwrite.call_count == 20
    mock_capture.release.assert_called_once()


@patch('module_capture_faces.cv2.VideoCapture')
def test_capture_faces_early_exit(mock_video_capture, cleanup):
    mock_capture = MagicMock()
    mock_capture.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture

    capture_faces("test_output")

    mock_capture.release.assert_called_once()
