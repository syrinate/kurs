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
    total_photos = 100

    window_name = f"Capture - {person_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while count < total_photos:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face_img = frame[y:y + h, x:x + w]

            save_path = f"{output_dir}/face_{count:03d}.jpg"
            cv2.imwrite(save_path, face_img)
            count += 1

            if count >= total_photos:
                break

        # Display progress text
        progress_text = f"Capturing photo: {count}/{total_photos}"
        cv2.putText(frame, progress_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Exit instruction
        cv2.putText(frame, "Press 'Q' to exit early", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        # Slower capture speed
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(f"Captured {count} photos for {person_name}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        capture_faces(sys.argv[1])
    else:
        print("Please provide a name: python module_capture_faces.py John")
