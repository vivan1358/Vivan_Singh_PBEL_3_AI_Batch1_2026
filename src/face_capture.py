"""
Captures face samples for a new student via webcam and registers them
in the database. Run through the GUI, or call capture_faces() directly.
"""
import os

import cv2

import config
from src import db


def capture_faces(student_id, name, roll_no, branch):
    if db.student_exists(student_id):
        return False, "That Student ID is already registered."

    student_dir = os.path.join(config.DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        return False, "Could not access the camera."

    count = 0
    target = config.FACE_SAMPLES_PER_STUDENT

    try:
        while count < target:
            ret, frame = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                count += 1
                face_img = cv2.resize(gray[y:y + h, x:x + w], config.FACE_IMG_SIZE)
                cv2.imwrite(os.path.join(student_dir, f"{count}.jpg"), face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {count}/{target}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break  # one face per frame is enough

            cv2.imshow("Registering face - press 'q' to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if count < target * 0.5:
        return False, f"Only captured {count} usable samples — registration cancelled. Try again with better lighting."

    db.add_student(student_id, name, roll_no, branch)
    return True, f"Captured {count} face samples for {name}."
