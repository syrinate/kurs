import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, mock_open
from module_recognize import FaceRecognizer

# Тест успешного распознавания лица
@patch('builtins.open', new_callable=mock_open)              # Подменяем функцию open
@patch('module_recognize.pickle.load')                       # Подменяем загрузку объектов из pickle
def test_recognize_success(mock_pickle, mock_open_file):
    # Создаём фиктивную модель, которая "распознаёт" лицо
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [np.array([0.1, 0.9])]  # Модель возвращает вероятности принадлежности к классам

    # Подделка label encoder'а (декодирует индекс в имя)
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Test Person"]

    # Возвращаем модель и encoder при последовательной загрузке через pickle.load
    mock_pickle.side_effect = [mock_model, mock_le]

    with patch('module_recognize.FaceNet') as mock_face_net, \
         patch('module_recognize.cv2.CascadeClassifier') as mock_detector:

        mock_face_net.return_value = MagicMock()  # Подменяем модель FaceNet
        # Эмуляция обнаружения одного лица на изображении
        mock_detector.return_value.detectMultiScale.return_value = [(50, 50, 100, 100)]

        recognizer = FaceRecognizer()  # Создаём экземпляр класса распознавания
        frame = np.zeros((300, 300, 3), dtype=np.uint8)  # Заглушка — пустое изображение
        result = recognizer.recognize(frame)  # Выполняем распознавание

        assert result.shape == frame.shape  # Проверяем, что результат — изображение того же размера


# Тест, когда лицо не обнаружено на кадре
@patch('builtins.open', new_callable=mock_open)
@patch('module_recognize.pickle.load')
def test_recognize_no_faces(mock_pickle, mock_open_file):
    # Подделка модели и label encoder'а, как и в предыдущем тесте
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [np.array([0.1, 0.9])]
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Test Person"]
    mock_pickle.side_effect = [mock_model, mock_le]

    with patch('module_recognize.FaceNet') as mock_face_net, \
         patch('module_recognize.cv2.CascadeClassifier') as mock_detector:

        mock_face_net.return_value = MagicMock()
        # Лицо не найдено (список пуст)
        mock_detector.return_value.detectMultiScale.return_value = []

        recognizer = FaceRecognizer()
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        result = recognizer.recognize(frame)  # Пытаемся распознать лицо, которого нет

        assert result.shape == frame.shape  # Убедимся, что функция возвращает изображение нужной формы
