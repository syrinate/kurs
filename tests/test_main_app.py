import tkinter as tk
import pytest
import time
from unittest.mock import patch, MagicMock, mock_open
from main_app import FaceRecognitionApp

@pytest.fixture
def app():
    root = tk.Tk()
    root.withdraw()  # скрываем окно, чтобы избежать отображения в тестах
    return FaceRecognitionApp(root)

@patch('builtins.open', new_callable=mock_open)
@patch('main_app.pickle.load')
@patch('main_app.cv2.CascadeClassifier')
@patch('main_app.FaceNet')
@patch('main_app.os.path.exists', return_value=True)
def test_load_models_success(mock_exists, mock_facenet, mock_detector, mock_pickle, mock_open_file, app):
    mock_model = MagicMock()
    mock_pickle.side_effect = [mock_model, MagicMock()]
    mock_facenet.return_value = MagicMock()
    app.load_models()  # Прямой вызов вместо async
    app.root.update()
    assert app.models_loaded is True

@patch('main_app.os.path.exists', return_value=False)
def test_load_models_failure(mock_exists, app):
    app.load_models()  # Прямой вызов вместо async
    app.root.update()
    assert app.models_loaded is False
    assert "Модели не найдены" in app.status_var.get()

@patch('main_app.simpledialog.askstring', return_value="Test Person")
@patch('main_app.os.makedirs')
def test_start_new_person_capture_logic_only(mock_makedirs, mock_askstring):
    mock_root = MagicMock()
    app = FaceRecognitionApp(mock_root)

    # Заменяем кнопку на мок с методом config
    app.add_face_btn = MagicMock()

    with patch.object(app, 'start_camera'), patch.object(app, 'display_image'):
        app.start_new_person_capture()

    assert app.new_person_mode is True
    assert app.new_person_name == "Test Person"
    mock_makedirs.assert_any_call("my_faces/Test Person", exist_ok=True)
    app.add_face_btn.config.assert_called_once_with(state='normal')
