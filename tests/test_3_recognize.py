import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open
from module_recognize import FaceRecognizer

@patch('builtins.open', new_callable=mock_open)
@patch('module_recognize.pickle.load')
def test_recognize_success(mock_pickle, mock_open_file):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [np.array([0.1, 0.9])]
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Test Person"]
    mock_pickle.side_effect = [mock_model, mock_le]

    with patch('module_recognize.FaceNet') as mock_face_net, \
         patch('module_recognize.cv2.CascadeClassifier') as mock_detector:
        mock_face_net.return_value = MagicMock()
        mock_detector.return_value.detectMultiScale.return_value = [(50, 50, 100, 100)]

        recognizer = FaceRecognizer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        result = recognizer.recognize(frame)

        assert result.shape == frame.shape

@patch('builtins.open', new_callable=mock_open)
@patch('module_recognize.pickle.load')
def test_recognize_no_faces(mock_pickle, mock_open_file):
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [np.array([0.1, 0.9])]
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Test Person"]
    mock_pickle.side_effect = [mock_model, mock_le]

    with patch('module_recognize.FaceNet') as mock_face_net, \
         patch('module_recognize.cv2.CascadeClassifier') as mock_detector:
        mock_face_net.return_value = MagicMock()
        mock_detector.return_value.detectMultiScale.return_value = []

        recognizer = FaceRecognizer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        result = recognizer.recognize(frame)

        assert result.shape == frame.shape