import cv2          # Библиотека OpenCV — работа с изображениями и видео
import os           # Для работы с файловой системой
import sys          # Для получения аргументов командной строки

# Основная функция захвата лиц
def capture_faces(person_name):
    # Создаём папку для сохранения изображений (если ещё не существует)
    output_dir = f"my_faces/{person_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем классификатор для обнаружения лиц (каскад Хаара)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Подключаемся к камере (0 — первая камера)
    cap = cv2.VideoCapture(0)

    count = 0                  # Счётчик сохранённых лиц
    total_photos = 100         # Общее количество фото для захвата

    # Окно вывода изображения с камеры
    window_name = f"Capture - {person_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Цикл захвата кадров, пока не достигнуто нужное число фото
    while count < total_photos:
        ret, frame = cap.read()           # Считываем кадр с камеры
        if not ret:
            break                         # Если кадр не получен — выходим

        # Преобразуем в чёрно-белое изображение (для детектора)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ищем лица на изображении
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Отрисовываем прямоугольник на экране
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Вырезаем изображение лица
            face_img = frame[y:y + h, x:x + w]

            # Формируем путь и сохраняем изображение
            save_path = f"{output_dir}/face_{count:03d}.jpg"
            cv2.imwrite(save_path, face_img)
            count += 1

            if count >= total_photos:
                break  # Если достигли лимита — выходим из цикла

        # Отображаем на экране прогресс захвата
        progress_text = f"Capturing photo: {count}/{total_photos}"
        cv2.putText(frame, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Инструкция по выходу
        cv2.putText(frame, "Press 'Q' to exit early", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Показываем кадр в окне
        cv2.imshow(window_name, frame)

        # Умышленная задержка (300 мс) между снимками
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break  # Выход по нажатию Q

    # Освобождаем ресурсы камеры и закрываем окна
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Минимальная задержка для закрытия окна

    print(f"Captured {count} photos for {person_name}.")

# Запуск из командной строки
if __name__ == "__main__":
    if len(sys.argv) > 1:
        capture_faces(sys.argv[1])  # Передаём имя человека как аргумент
    else:
        print("Please provide a name: python module_capture_faces.py John")
