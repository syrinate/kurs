import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import pickle
import threading
import subprocess
import sys


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)  # Фиксируем минимальный размер окна

        # Состояние системы
        self.is_camera_active = False
        self.current_face = None
        self.models_loaded = False
        self.new_person_mode = False
        self.new_person_name = ""
        self.new_faces_count = 0

        # Инициализация интерфейса
        self.setup_ui()

        # Блокировка кнопок при запуске
        self.toggle_buttons_state(False)

        # Загрузка моделей в фоновом режиме
        self.load_models_async()

    def setup_ui(self):
        """Инициализация пользовательского интерфейса"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Панель управления
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Область отображения
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Кнопки управления
        self.capture_btn = ttk.Button(
            control_frame, text="1. Захватить нового человека",
            command=self.start_new_person_capture)
        self.capture_btn.pack(fill=tk.X, pady=5)

        self.process_btn = ttk.Button(
            control_frame, text="2. Обработать данные",
            command=self.process_data)
        self.process_btn.pack(fill=tk.X, pady=5)

        self.train_btn = ttk.Button(
            control_frame, text="3. Обучить модель",
            command=self.train_model)
        self.train_btn.pack(fill=tk.X, pady=5)

        self.recognition_btn = ttk.Button(
            control_frame, text="4. Запустить распознавание",
            command=self.toggle_recognition)
        self.recognition_btn.pack(fill=tk.X, pady=5)

        self.add_face_btn = ttk.Button(
            control_frame, text="Добавить текущее лицо",
            command=self.add_current_face, state=tk.DISABLED)
        self.add_face_btn.pack(fill=tk.X, pady=5)

        # Canvas для отображения видео
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Загрузка моделей...")
        ttk.Label(control_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN).pack(fill=tk.X, pady=5, side=tk.BOTTOM)

        # Прогресс бар
        self.progress = ttk.Progressbar(
            control_frame, orient=tk.HORIZONTAL,
            mode='indeterminate', length=100)
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()

    def toggle_buttons_state(self, enabled):
        """Блокировка/разблокировка кнопок"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.capture_btn.config(state=state)
        self.process_btn.config(state=state)
        self.train_btn.config(state=state)
        self.recognition_btn.config(state=state)

    def run_subprocess_with_callback(self, command, callback):
        """Запускает подпроцесс и вызывает callback по завершении"""
        try:
            process = subprocess.Popen(command)
            process.wait()  # Ожидаем завершения
            if process.returncode == 0:
                self.root.after(0, callback)
            else:
                self.root.after(0, lambda: self.show_error(
                    f"Процесс завершился с ошибкой (код {process.returncode})"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Ошибка выполнения: {str(e)}"))

    def load_models_async(self):
        """Фоновая загрузка моделей с обновлением статуса"""

        def load_models():
            try:
                # Проверяем существование необходимых файлов
                if all(os.path.exists(f) for f in ["svm_model.pkl", "label_encoder.pkl"]):
                    with open("svm_model.pkl", "rb") as f:
                        self.model = pickle.load(f)
                    with open("label_encoder.pkl", "rb") as f:
                        self.le = pickle.load(f)
                    from keras_facenet import FaceNet
                    self.embedder = FaceNet()
                    self.detector = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    self.models_loaded = True
                    self.root.after(0, lambda: self.status_var.set("Модели загружены"))
                else:
                    self.root.after(0, lambda: self.status_var.set("Модели не найдены"))

                self.root.after(0, self.hide_loading)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Ошибка загрузки моделей: {str(e)}"))

        threading.Thread(target=load_models, daemon=True).start()

    def hide_loading(self):
        """Скрытие индикатора загрузки"""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set("Готов к работе" if self.models_loaded else "Модели не загружены")
        self.toggle_buttons_state(True)

    def show_error(self, message):
        """Отображение ошибки"""
        messagebox.showerror("Ошибка", message)
        self.status_var.set("Ошибка")
        self.toggle_buttons_state(True)

    def start_new_person_capture(self):
        """Начало захвата нового человека"""
        self.new_person_name = simpledialog.askstring(
            "Новый человек", "Введите имя человека:", parent=self.root)

        if self.new_person_name:
            os.makedirs(f"my_faces/{self.new_person_name}", exist_ok=True)
            self.new_person_mode = True
            self.new_faces_count = 0
            self.status_var.set(f"Захват лиц для {self.new_person_name} - добавляйте лица")
            self.start_camera()
            self.add_face_btn.config(state=tk.NORMAL)

    def start_camera(self):
        """Запуск камеры для захвата"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        self.is_camera_active = True
        self.update_frame()

    def process_data(self):
        """Обработка данных для обучения"""
        if not os.path.exists("my_faces"):
            self.show_error("Сначала захватите лица!")
            return

        try:
            self.toggle_buttons_state(False)
            self.status_var.set("Обработка данных...")

            def completion_callback():
                self.status_var.set("Данные успешно обработаны!")
                self.toggle_buttons_state(True)

            # Запуск в отдельном потоке для отслеживания завершения
            threading.Thread(
                target=self.run_subprocess_with_callback,
                args=([sys.executable, "2_extract_embeddings.py"], completion_callback),
                daemon=True
            ).start()

        except Exception as e:
            self.show_error(f"Ошибка обработки данных: {str(e)}")

    def train_model(self):
        """Обучение модели классификации"""
        if not os.path.exists("embeddings.npy"):
            self.show_error("Сначала обработайте данные!")
            return

        try:
            self.toggle_buttons_state(False)
            self.status_var.set("Обучение модели...")

            def completion_callback():
                self.status_var.set("Модель успешно обучена!")
                self.load_models_async()  # Перезагружаем модели
                self.toggle_buttons_state(True)

            threading.Thread(
                target=self.run_subprocess_with_callback,
                args=([sys.executable, "3_train_model.py"], completion_callback),
                daemon=True
            ).start()

        except Exception as e:
            self.show_error(f"Ошибка обучения: {str(e)}")

    def toggle_recognition(self):
        """Переключение режима распознавания"""
        if not self.is_camera_active:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """Запуск распознавания"""
        if not self.models_loaded:
            self.show_error("Модели не загружены!")
            return

        self.new_person_mode = False
        self.start_camera()
        self.recognition_btn.config(text="Остановить распознавание")
        self.add_face_btn.config(state=tk.DISABLED)  # Кнопка добавления отключена при распознавании
        self.status_var.set("Режим распознавания")

    def stop_recognition(self):
        """Остановка распознавания"""
        self.is_camera_active = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.recognition_btn.config(text="Запустить распознавание")
        self.add_face_btn.config(state=tk.DISABLED)
        self.status_var.set("Режим остановлен")

    def update_frame(self):
        """Обновление кадров с камеры"""
        if self.is_camera_active and hasattr(self, 'cap') and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_face = frame.copy()

                if not self.new_person_mode and self.models_loaded:
                    frame = self.process_frame(frame)

                self.display_image(frame)

            self.root.after(10, self.update_frame)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face = cv2.resize(face_img, (160, 160))

            try:
                embedding = self.embedder.embeddings(np.expand_dims(face, axis=0))[0]
                proba = self.model.predict_proba([embedding])[0]
                max_prob = np.max(proba)
                pred_class = np.argmax(proba)
                name = self.le.inverse_transform([pred_class])[0]

                # Все классы кроме "Unknown" — зеленые
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name} {max_prob:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"Ошибка: {e}")

        return frame

    def display_image(self, frame):
        """Отображение кадра в интерфейсе"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def add_current_face(self):
        """Добавление текущего лица в датасет"""
        if self.current_face is None or not self.new_person_mode:
            return

        try:
            cv2.imwrite(f"my_faces/{self.new_person_name}/face_{self.new_faces_count}.jpg",
                        self.current_face)
            self.new_faces_count += 1
            self.status_var.set(
                f"Добавлено {self.new_faces_count} лиц. Продолжайте или перейдите к обработке")
        except Exception as e:
            self.show_error(f"Ошибка сохранения: {str(e)}")

    def on_closing(self):
        """Обработчик закрытия окна"""
        self.stop_recognition()
        self.root.destroy()


if __name__ == "__main__":
    # Проверка зависимостей
    try:
        import numpy as np
        import cv2
        from PIL import Image, ImageTk
        import pickle
        from keras_facenet import FaceNet
    except ImportError as e:
        print(f"Ошибка: Не установлены необходимые библиотеки: {e}")
        print("Пожалуйста, выполните: pip install numpy opencv-python pillow keras-facenet")
        sys.exit(1)

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()