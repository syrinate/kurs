import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
import os

class FaceRecognizer:
    def __init__(self):
        # Инициализация моделей
        self.embedder = FaceNet()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Загрузка обученных моделей
        with open("svm_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)

        # Пороговые значения
        self.confidence_threshold = 0.97  # Минимальная уверенность для распознавания
        self.face_size = (160, 160)  # Размер лица для обработки

    def recognize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face = cv2.resize(face_img, (160, 160))

            embedding = self.embedder.embeddings(np.expand_dims(face, axis=0))[0]
            proba = self.model.predict_proba([embedding])[0]
            pred_class = np.argmax(proba)
            name = self.le.inverse_transform([pred_class])[0]

            # Красный только для "Unknown", остальные — зеленые
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {np.max(proba):.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame


def main():
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognizer.recognize(frame)
        cv2.imshow("Face Recognition (Press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()