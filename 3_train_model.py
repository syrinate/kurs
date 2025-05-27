import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    X = np.load("embeddings.npy")
    y = np.load("labels.npy")

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем SVM с вероятностным выводом
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Сохраняем модель
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Оценка точности
    accuracy = model.score(X_test, y_test)
    print(f"Точность на тестах: {accuracy:.2%}")


if __name__ == "__main__":
    train_model()