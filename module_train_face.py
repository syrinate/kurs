import numpy as np
import cv2
import os
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import traceback


def _detect_and_extract_face(img, detector):
    """Detect a face in the image and return the cropped, resized face or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]
    return cv2.resize(face, (160, 160))


def _compute_embedding(face, embedder):
    """Compute and return embedding for a single face image."""
    return embedder.embeddings(np.expand_dims(face, axis=0))[0]


def _process_image(img_path, detector, embedder, person_name, embeddings, labels):
    """Process a single image: detect face, compute embedding, append results."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {img_path}")
        return

    face = _detect_and_extract_face(img, detector)
    if face is None:
        print(f"Не обнаружено лиц на изображении: {img_path}")
        return

    try:
        embedding = _compute_embedding(face, embedder)
    except Exception as e:
        print(f"Ошибка обработки изображения {img_path}: {str(e)}")
        return

    embeddings.append(embedding)
    labels.append(person_name)

    if person_name == "Unknown":
        embeddings.append(np.random.normal(size=(512,)))
        labels.append("Unknown")


def _collect_embeddings(detector, embedder):
    """Walk through 'my_faces' directory and collect embeddings + labels."""
    embeddings = []
    labels = []

    for person_name in os.listdir("my_faces"):
        person_dir = os.path.join("my_faces", person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            _process_image(img_path, detector, embedder, person_name, embeddings, labels)

    return embeddings, labels


def _save_embeddings_and_encoder(embeddings, labels):
    """Encode labels, save embeddings, labels, and the label encoder. Return (X, y, le)."""
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    np.save("embeddings.npy", np.array(embeddings))
    np.save("labels.npy", labels_encoded)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Обработано {len(embeddings)} лиц для {len(le.classes_)} человек")
    return np.array(embeddings), np.array(labels_encoded), le


def _train_and_save_model(X, y):
    """Split data, train a calibrated GradientBoosting model, save it, and print accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    with open("svm_model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)

    accuracy = calibrated_model.score(X_test, y_test)
    print(f"Точность на тестах: {accuracy:.2%}")


def train_face_recognition():
    try:
        embedder = FaceNet()
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not os.path.exists("my_faces"):
            print("Ошибка: Папка 'my_faces' не найдена!")
            return 1

        embeddings, labels = _collect_embeddings(detector, embedder)

        if len(embeddings) == 0:
            print("Ошибка: Не найдено ни одного лица для обучения!")
            return 1

        X, y, le = _save_embeddings_and_encoder(embeddings, labels)

        if len(np.unique(y)) < 2:
            print("Ошибка: Недостаточно классов для обучения (нужно минимум 2 разных человека)!")
            return 1

        _train_and_save_model(X, y)
        return 0

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = train_face_recognition()
    exit(exit_code)
