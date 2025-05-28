import cv2  # OpenCV — для работы с изображениями и камерой
import numpy as np  # NumPy — для работы с массивами
import pickle  # Для загрузки сохранённых моделей
from keras_facenet import FaceNet  # Для получения эмбеддингов лица
import os

# Класс, реализующий распознавание лиц
class FaceRecognizer:
    def __init__(self):
        # Инициализация модели FaceNet (эмбеддинги)
        self.embedder = FaceNet()

        # Инициализация детектора лиц (каскад Хаара)
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Загрузка обученного классификатора из файла
        with open("svm_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Загрузка кодировщика имён
        with open("label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)

        # Порог уверенности (чем выше, тем строже распознавание)
        self.confidence_threshold = 0.97

        # Размер изображения лица, с которым работает FaceNet
        self.face_size = (160, 160)

    # Метод распознавания лиц на одном кадре
    def recognize(self, frame):
        # Перевод изображения в ч/б для поиска лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц на изображении
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        # Обработка каждого найденного лица
        for (x, y, w, h) in faces:
            # Вырезаем изображение лица и приводим к нужному размеру
            face_img = frame[y:y + h, x:x + w]
            face = cv2.resize(face_img, self.face_size)

            # Получаем эмбеддинг — числовое представление лица
            embedding = self.embedder.embeddings(np.expand_dims(face, axis=0))[0]

            # Прогноз вероятностей по всем классам (пользователям)
            proba = self.model.predict_proba([embedding])[0]

            # Получаем индекс и имя класса с максимальной вероятностью
            pred_class = np.argmax(proba)
            name = self.le.inverse_transform([pred_class])[0]

            # Цвет рамки: зелёный для известных, красный для "Unknown"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Отрисовываем рамку и имя с уровнем уверенности
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {np.max(proba):.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame  # Возвращаем изображение с наложенными подписями

# Демонстрация работы в видеопотоке
def main():
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)  # Подключение к камере

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр: распознаём лицо
        frame = recognizer.recognize(frame)

        # Показываем результат
        cv2.imshow("Face Recognition (Press Q to quit)", frame)

        # Выход по клавише Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Освобождение камеры
    cv2.destroyAllWindows()  # Закрытие всех окон

# Точка входа
if __name__ == "__main__":
    main()
