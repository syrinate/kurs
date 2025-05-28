import numpy as np  # Работа с массивами
import cv2  # OpenCV — для обработки изображений
import os  # Работа с файловой системой
import pickle  # Сохранение модели и кодировщика
from keras_facenet import FaceNet  # Модель для получения эмбеддингов лица
from sklearn.preprocessing import LabelEncoder  # Кодировка имён в числа
from sklearn.model_selection import train_test_split  # Деление на train/test
from sklearn.ensemble import GradientBoostingClassifier  # Бустинг-классификатор
from sklearn.calibration import CalibratedClassifierCV  # Калибровка вероятностей
import traceback  # Для вывода ошибок при отладке


def train_face_recognition():
    try:
        # Шаг 1. Инициализация моделей
        embedder = FaceNet()  # FaceNet — преобразует лицо в вектор (512 признаков)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        embeddings = []  # Список эмбеддингов
        labels = []      # Список меток (имён)

        # Проверка наличия папки с лицами
        if not os.path.exists("my_faces"):
            print("Ошибка: Папка 'my_faces' не найдена!")
            return 1

        # Обход всех пользователей
        for person_name in os.listdir("my_faces"):
            person_dir = os.path.join("my_faces", person_name)
            if not os.path.isdir(person_dir):
                continue  # Пропускаем, если не папка

            # Обход изображений внутри папки пользователя
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue

                # Преобразуем изображение в оттенки серого
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Находим лицо на изображении
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                if len(faces) == 0:
                    print(f"Не обнаружено лиц на изображении: {img_path}")
                    continue

                # Используем первое найденное лицо
                x, y, w, h = faces[0]
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (160, 160))

                # Получаем эмбеддинг (вектор признаков)
                try:
                    embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
                    embeddings.append(embedding)
                    labels.append(person_name)

                    # Если пользователь — "Unknown", добавим случайный шум
                    if person_name == "Unknown":
                        embeddings.append(np.random.normal(size=(512,)))
                        labels.append("Unknown")
                except Exception as e:
                    print(f"Ошибка обработки изображения {img_path}: {str(e)}")
                    continue

        if len(embeddings) == 0:
            print("Ошибка: Не найдено ни одного лица для обучения!")
            return 1

        # Шаг 2. Кодирование меток (имён) в числа
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Сохраняем эмбеддинги и метки
        np.save("embeddings.npy", np.array(embeddings))
        np.save("labels.npy", labels_encoded)
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        print(f"Обработано {len(embeddings)} лиц для {len(le.classes_)} человек")

        # Подготовка данных к обучению
        X = np.array(embeddings)
        y = np.array(labels_encoded)

        # Проверка: минимум 2 разных пользователя
        if len(np.unique(y)) < 2:
            print("Ошибка: Недостаточно классов для обучения (нужно минимум 2 разных человека)!")
            return 1

        # Делим выборку на обучающую и тестовую
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Обучение классификатора (градиентный бустинг)
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        # Калибровка — чтобы получить вероятности (а не просто метки)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
        calibrated_model.fit(X_train, y_train)

        # Сохраняем модель в файл
        with open("svm_model.pkl", "wb") as f:
            pickle.dump(calibrated_model, f)

        # Оцениваем точность на тесте
        accuracy = calibrated_model.score(X_test, y_test)
        print(f"Точность на тестах: {accuracy:.2%}")

        return 0  # Успешное завершение

    except Exception as e:
        # В случае ошибки — выводим её в консоль
        print(f"Критическая ошибка: {str(e)}")
        traceback.print_exc()
        return 1


# Запуск напрямую
if __name__ == "__main__":
    exit_code = train_face_recognition()
    exit(exit_code)
