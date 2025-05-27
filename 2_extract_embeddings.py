import numpy as np
import cv2
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle


def extract_embeddings():
    embedder = FaceNet()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    embeddings = []
    labels = []

    # Собираем данные всех известных людей
    for person_name in os.listdir("my_faces"):
        person_dir = os.path.join("my_faces", person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (160, 160))
                embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
                embeddings.append(embedding)
                labels.append(person_name)
                if person_name == "Unknown":
                    embeddings.append(np.random.normal(size=(512,)))  # Искусственный шум
                    labels.append("Unknown")

    # Кодируем метки
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Сохраняем данные
    np.save("embeddings.npy", np.array(embeddings))
    np.save("labels.npy", labels_encoded)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Обработано {len(embeddings)} лиц для {len(le.classes_)} человек")


if __name__ == "__main__":
    extract_embeddings()