import shutil
import pytest
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
from module_capture_faces import capture_faces

# Фикстура pytest для очистки папки test_output после тестов
@pytest.fixture
def cleanup():
    yield  # выполнение теста
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")  # удаление папки после выполнения


# Тест успешного захвата лиц
@patch('module_capture_faces.cv2.VideoCapture')  # Подменяем видеозахват OpenCV
@patch('module_capture_faces.cv2.imwrite')       # Подменяем сохранение изображений
def test_capture_faces_success(mock_imwrite, mock_video_capture, cleanup):
    mock_capture = MagicMock()
    # Эмуляция успешного получения кадра (чёрное изображение 480x640)
    mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_capture  # Используем подменённую камеру

    with patch('module_capture_faces.cv2.CascadeClassifier') as mock_detector:
        # Эмуляция обнаружения одного лица на каждом кадре
        mock_detector.return_value.detectMultiScale.return_value = [(50, 50, 100, 100)]
        capture_faces("test_output")  # Запуск функции

    assert mock_imwrite.call_count == 100  # Проверяем, что было сохранено 100 изображений
    mock_capture.release.assert_called_once()  # Убедимся, что камера была корректно "отпущена"


# Тест досрочного завершения при ошибке чтения с камеры
@patch('module_capture_faces.cv2.VideoCapture')
def test_capture_faces_early_exit(mock_video_capture, cleanup):
    mock_capture = MagicMock()
    # Эмуляция неудачного чтения кадра (например, камера не работает)
    mock_capture.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture

    capture_faces("test_output")  # Запускаем захват

    mock_capture.release.assert_called_once()  # Проверяем, что камера была "отпущена" даже при ошибке
