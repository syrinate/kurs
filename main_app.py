import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import os
import threading
import subprocess
import sys
from unittest.mock import MagicMock

import cv2
import pickle
from keras_facenet import FaceNet

from module_recognize import FaceRecognizer
from module_capture_faces import capture_faces
from module_train_face import train_face_recognition

class FaceRecognitionApp:
    def __init__(self, root=None, headless=False):
        self.root = root or tk.Tk()
        self.headless = headless

        if not self.headless:
            self.setup_ui()
        else:
            self.status_var = MagicMock()
            self.add_face_btn = MagicMock()
            self.capture_btn = MagicMock()
            self.process_train_btn = MagicMock()
            self.recognition_btn = MagicMock()
        self.root.title("Face Recognition System")
        self.root.geometry("800x500")
        self.root.minsize(800, 500)

        # System state
        self.is_camera_active = False
        self.current_face = None
        self.models_loaded = False
        self.new_person_mode = False
        self.new_person_name = ""
        self.new_faces_count = 0
        self.recognition_active = False

        # Disable buttons at startup
        self.toggle_buttons_state(False)

        # Load models in background
        self.load_models_async()

    def setup_ui(self):
        """Initialize user interface"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel with fixed width
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)

        # Display area
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Button options
        button_options = {
            'width': 25,
            'padding': 5
        }

        # Capture button (will cancel recognition when clicked)
        self.capture_btn = ttk.Button(
            control_frame, text="1. Захватить нового человека",
            command=self.handle_capture_button, **button_options)
        self.capture_btn.pack(fill=tk.X, pady=5)

        # Combined process and train button
        self.process_train_btn = ttk.Button(
            control_frame, text="2. Обработать и обучить",
            command=self.process_and_train, **button_options)
        self.process_train_btn.pack(fill=tk.X, pady=5)

        # Recognition toggle button
        self.recognition_btn = ttk.Button(
            control_frame, text="3. Запустить распознавание",
            command=self.toggle_recognition, **button_options)
        self.recognition_btn.pack(fill=tk.X, pady=5)

        # Video display canvas
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Загрузка моделей...")
        ttk.Label(control_frame, textvariable=self.status_var,
                 relief=tk.SUNKEN).pack(fill=tk.X, pady=5, side=tk.BOTTOM)

        # Progress bar
        self.progress = ttk.Progressbar(
            control_frame, orient=tk.HORIZONTAL,
            mode='indeterminate', length=100)
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()

    def handle_capture_button(self):
        """Handle capture button click - cancels recognition if active"""
        if self.recognition_active:
            self.stop_recognition()
        self.start_new_person_capture()

    def process_and_train(self):
        if not os.path.exists("my_faces"):
            self.show_error("Сначала захватите лица!")
            return

        self.toggle_buttons_state(False)
        self.status_var.set("Обработка и обучение...")

        def run_training():
            code = train_face_recognition()
            self.root.after(0, lambda: self.training_finished(code))

        threading.Thread(target=run_training, daemon=True).start()

    def training_finished(self, code):
        if code == 0:
            self.status_var.set("Обучение завершено!")
            self.models_loaded = True
            self.recognizer = FaceRecognizer()
        else:
            self.show_error("Ошибка при обучении.")
        self.toggle_buttons_state(True)

    def toggle_recognition(self):
        """Toggle recognition mode"""
        if not self.recognition_active:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """Start recognition mode"""
        if not self.models_loaded:
            self.show_error("Модели не загружены!")
            return

        self.new_person_mode = False
        self.recognition_active = True
        self.start_camera()
        self.recognition_btn.config(text="Остановить распознавание")
        self.add_face_btn.config(state=tk.DISABLED)
        self.status_var.set("Режим распознавания")

    def stop_recognition(self):
        """Stop recognition mode"""
        self.recognition_active = False
        self.is_camera_active = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.recognition_btn.config(text="Запустить распознавание")
        self.add_face_btn.config(state=tk.DISABLED)
        self.status_var.set("Режим остановлен")

    def toggle_buttons_state(self, enabled):
        """Enable/disable buttons"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.capture_btn.config(state=state)
        self.process_train_btn.config(state=state)
        self.recognition_btn.config(state=state)

    def run_subprocess_with_callback(self, command, callback):
        """Run subprocess and call callback when finished"""
        try:
            process = subprocess.Popen(command)
            process.wait()
            if process.returncode == 0:
                self.root.after(0, callback)
            else:
                self.root.after(0, lambda: self.show_error(
                    f"Процесс завершился с ошибкой (код {process.returncode})"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Ошибка выполнения: {str(e)}"))

    def load_models(self):
        try:
            print("DEBUG: Загрузка моделей начата")

            if not os.path.exists("svm_model.pkl") or not os.path.exists("label_encoder.pkl"):
                print("DEBUG: Модели не найдены")
                self.status_var.set("Models not found")
                return

            with open("svm_model.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open("label_encoder.pkl", "rb") as f:
                self.le = pickle.load(f)

            print("DEBUG: pickle.load прошёл")

            self.embedder = FaceNet(model_path="models/20180402-114759")
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            print("DEBUG: FaceNet и CascadeClassifier инициализированы")

            self.models_loaded = True
            self.status_var.set("Models loaded")

            print("DEBUG: models_loaded = True")

        except Exception as e:
            print(f"ERROR in load_models: {str(e)}")
            self.status_var.set(f"Model loading error: {str(e)}")

    def load_models_async(self):
        """Load models in background thread"""
        def load_models():
            try:
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
                    self.recognizer = FaceRecognizer()
                    self.root.after(0, lambda: self.status_var.set("Модели загружены"))
                else:
                    self.root.after(0, lambda: self.status_var.set("Модели не найдены"))

                self.root.after(0, self.hide_loading)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(f"Ошибка загрузки моделей: {str(e)}"))

        threading.Thread(target=load_models, daemon=True).start()

    def hide_loading(self):
        """Hide loading indicator"""
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set("Готов к работе" if self.models_loaded else "Модели не загружены")
        self.toggle_buttons_state(True)

    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Ошибка", message)
        self.status_var.set("Ошибка")
        self.toggle_buttons_state(True)

    def start_new_person_capture(self):
        self.new_person_name = simpledialog.askstring("Новый человек", "Введите имя человека:", parent=self.root)
        if self.new_person_name:
            self.new_person_mode = True
            self.status_var.set(f"Захват лиц для {self.new_person_name} — подождите...")
            self.toggle_buttons_state(False)
            self.add_face_btn.config(state="normal")
            def run_capture():
                capture_faces(self.new_person_name)
                self.root.after(0, lambda: [
                    self.status_var.set(f"Захват завершён для {self.new_person_name}"),
                    self.toggle_buttons_state(True)
                ])

            threading.Thread(target=run_capture, daemon=True).start()

    def start_camera(self):
        """Start camera capture"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Не удалось открыть камеру!")
            return

        self.is_camera_active = True
        self.update_frame()

    def update_frame(self):
        """Update camera frames"""
        if self.is_camera_active and hasattr(self, 'cap') and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_face = frame.copy()

                if not self.new_person_mode and self.models_loaded:
                    frame = self.process_frame(frame)

                self.display_image(frame)

            self.root.after(10, self.update_frame)

    def process_frame(self, frame):
        return self.recognizer.recognize(frame)

    def display_image(self, frame):
        """Display image in GUI"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def on_closing(self):
        """Properly stop camera and close the app"""
        self.is_camera_active = False
        self.recognition_active = False

        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()

        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    # Check dependencies
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