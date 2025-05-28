import os
import shutil
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from module_train_face import train_face_recognition

# Подготовка тестовой среды — создаём структуру папок с фейковыми изображениями лиц
def setup_test_environment():
    os.makedirs("my_faces", exist_ok=True)
    os.makedirs(os.path.join("my_faces", "person1"), exist_ok=True)
    os.makedirs(os.path.join("my_faces", "person2"), exist_ok=True)

    # Создаём белое изображение 100x100 пикселей и сохраняем по 2 изображения для каждой "персоны"
    dummy_face = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    for i in range(2):
        cv2.imwrite(os.path.join("my_faces", "person1", f"face_{i}.jpg"), dummy_face)
        cv2.imwrite(os.path.join("my_faces", "person2", f"face_{i}.jpg"), dummy_face)

# Очистка тестовой среды после выполнения теста
def teardown_test_environment():
    shutil.rmtree("my_faces", ignore_errors=True)

# Основной тест функции обучения face recognition
@patch('module_train_face.FaceNet')  # Подменяем модель FaceNet
@patch('module_train_face.cv2.CascadeClassifier')  # Подменяем каскадный классификатор OpenCV
@patch('module_train_face.os.path.exists', return_value=True)  # Подменяем проверку наличия пути
def test_train_face_recognition_success(mock_exists, mock_detector, mock_face_net):
    setup_test_environment()  # Создаём тестовую структуру файлов

    # Эмуляция обнаружения одного лица на изображении
    mock_detector.return_value.detectMultiScale.return_value = [(0, 0, 100, 100)]

    # Подделка модели FaceNet с фиктивным методом embeddings
    mock_embedder = MagicMock()
    mock_embedder.embeddings.return_value = [np.random.rand(512)]  # Возвращаем случайный эмбеддинг
    mock_face_net.return_value = mock_embedder

    # Подмена функции os.listdir — возвращает файлы и папки в зависимости от запроса
    def fake_listdir(path):
        if "person1" in path or "person2" in path:
            return ["face_0.jpg", "face_1.jpg"]
        return ["person1", "person2"]

    # Патчим все зависимости, используемые в train_face_recognition
    with patch('module_train_face.os.listdir', side_effect=fake_listdir), \
         patch('module_train_face.os.path.isdir', return_value=True), \
         patch('sklearn.calibration.CalibratedClassifierCV.fit', return_value=None), \
         patch('sklearn.calibration.CalibratedClassifierCV.score', return_value=1.0):
        result = train_face_recognition()  # Запускаем функцию обучения

    teardown_test_environment()  # Удаляем созданные тестовые данные
    assert result == 0  # Проверяем, что функция завершилась успешно
