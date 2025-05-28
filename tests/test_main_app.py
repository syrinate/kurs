import tkinter as tk
from unittest.mock import patch, MagicMock, mock_open
from main_app import FaceRecognitionApp

# Тест успешной загрузки моделей
@patch('builtins.open', new_callable=mock_open)  # Подменяем встроенную функцию open
@patch('main_app.pickle.load')                   # Подменяем загрузку через pickle
@patch('main_app.cv2.CascadeClassifier')         # Подменяем загрузку каскадного классификатора OpenCV
@patch('main_app.FaceNet', new=MagicMock())      # Подменяем FaceNet заглушкой
@patch('main_app.os.path.exists')                # Подменяем проверку существования файлов
def test_load_models_success(mock_exists, mock_detector, mock_pickle, mock_open_file):
    app = FaceRecognitionApp(headless=True)  # Создаём экземпляр приложения в headless-режиме (без GUI)

    # Подменяем модель и label encoder
    app.model = MagicMock()
    app.le = MagicMock()
    app.status_var = MagicMock()

    # Устанавливаем возвращаемые значения для моков
    mock_exists.return_value = True  # Файлы "существуют"
    mock_pickle.side_effect = [MagicMock(), MagicMock()]  # Имитация загрузки объектов
    mock_detector.return_value = MagicMock()  # Имитация загрузки каскадного классификатора

    app.load_models()  # Вызываем метод загрузки моделей

    assert app.models_loaded is True  # Проверяем, что флаг успешной загрузки установлен


# Тест неудачной загрузки моделей (файлы не найдены)
@patch('main_app.os.path.exists', return_value=False)  # Всегда возвращает False – файлов нет
def test_load_models_failure(mock_exists):
    app = FaceRecognitionApp(headless=True)
    app.status_var = MagicMock()

    app.load_models()

    assert app.models_loaded is False  # Проверяем, что загрузка не удалась
    app.status_var.set.assert_called_with("Models not found")  # Проверяем, что выведено сообщение об ошибке


# Тест логики начала захвата нового лица (без GUI-элементов)
@patch('main_app.capture_faces')  # Подменяем функцию захвата лица
@patch('main_app.simpledialog.askstring', return_value="Test Person")  # Эмуляция ввода имени
@patch('main_app.os.makedirs')  # Подменяем создание папки
def test_start_new_person_capture_logic_only(mock_makedirs, mock_askstring, mock_capture_faces):
    mock_root = MagicMock()
    app = FaceRecognitionApp(mock_root)  # Создаём приложение с поддельным корнем

    app.add_face_btn = MagicMock()  # Подменяем кнопку добавления лица

    # Подменяем методы, не связанные с логикой
    with patch.object(app, 'start_camera'), \
         patch.object(app, 'display_image'), \
         patch('main_app.threading.Thread') as mock_thread:
        mock_thread.return_value.start = MagicMock()  # Подменяем запуск потока
        app.start_new_person_capture()  # Вызываем метод захвата нового лица

    # Проверяем, что переменные обновлены корректно
    assert app.new_person_mode is True
    assert app.new_person_name == "Test Person"
    app.add_face_btn.config.assert_called_once_with(state='normal')  # Кнопка активирована
