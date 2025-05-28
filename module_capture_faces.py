import cv2
import os
import sys


def capture_faces(person_name):
    output_dir = f"my_faces/{person_name}"
    os.makedirs(output_dir, exist_ok=True)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0

    # Create a fixed-size window
    cv2.namedWindow(f"Захват для {person_name} (20 фото)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Захват для {person_name} (20 фото)", 800, 600)

    while count < 20:  # Захватываем 20 фото
        ret, frame = cap.read()
        if not ret:
            break

        # Add a border frame to the video feed
        frame_with_border = cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        cv2.putText(frame_with_border, f"Capturing: {count}/20", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_with_border, (x + 50, y + 50), (x + w + 50, y + h + 50), (255, 0, 0), 2)
            face_img = frame[y:y + h, x:x + w]

            # Автоматическое сохранение
            cv2.imwrite(f"{output_dir}/face_{count}.jpg", face_img)
            count += 1

        cv2.imshow(f"Захват для {person_name} (20 фото)", frame_with_border)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        capture_faces(sys.argv[1])