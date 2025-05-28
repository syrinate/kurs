import numpy as np
import cv2
import os
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import traceback  # Добавляем для отладки


def train_face_recognition():
    try:
        # Step 1: Extract embeddings
        embedder = FaceNet()
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        embeddings = []
        labels = []

        # Проверяем, что папка с лицами существует
        if not os.path.exists("my_faces"):
            print("Ошибка: Папка 'test_faces' не найдена!")
            return 1

        # Собираем данные всех известных людей
        for person_name in os.listdir("my_faces"):
            person_dir = os.path.join("my_faces", person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

                if len(faces) == 0:
                    print(f"Не обнаружено лиц на изображении: {img_path}")
                    continue

                x, y, w, h = faces[0]
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (160, 160))

                try:
                    embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
                    embeddings.append(embedding)
                    labels.append(person_name)

                    if person_name == "Unknown":
                        embeddings.append(np.random.normal(size=(512,)))
                        labels.append("Unknown")
                except Exception as e:
                    print(f"Ошибка обработки изображения {img_path}: {str(e)}")
                    continue

        if len(embeddings) == 0:
            print("Ошибка: Не найдено ни одного лица для обучения!")
            return 1

        # Кодируем метки
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Сохраняем данные
        np.save("embeddings.npy", np.array(embeddings))
        np.save("labels.npy", labels_encoded)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        print(f"Обработано {len(embeddings)} лиц для {len(le.classes_)} человек")

        # Step 2: Train model
        X = np.array(embeddings)
        y = np.array(labels_encoded)

        # Проверяем, что есть достаточно данных для обучения
        if len(np.unique(y)) < 2:
            print("Ошибка: Недостаточно классов для обучения (нужно минимум 2 разных человека)!")
            return 1

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучаем модель
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
        calibrated_model.fit(X_train, y_train)

        # Сохраняем модель
        with open("svm_model.pkl", "wb") as f:
            pickle.dump(calibrated_model, f)

        # Оценка точности
        accuracy = calibrated_model.score(X_test, y_test)
        print(f"Точность на тестах: {accuracy:.2%}")

        return 0

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = train_face_recognition()
    exit(exit_code)