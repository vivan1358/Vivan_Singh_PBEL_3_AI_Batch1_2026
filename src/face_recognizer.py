"""
Opens the webcam, recognizes registered faces frame by frame using the
trained LBPH model, and marks attendance (once per student per day).
"""
import os
import pickle

import cv2

import config
from src import db


def run_attendance():
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.LABELS_PATH):
        return False, "Model not trained yet. Train the model first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(config.MODEL_PATH)

    with open(config.LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)  # numeric_label -> student_id

    students = {row[0]: row[1] for row in db.get_all_students()}  # student_id -> name

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        return False, "Could not access the camera."

    marked_this_session = set()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], config.FACE_IMG_SIZE)
                label, confidence = recognizer.predict(face_img)

                if confidence < config.RECOGNITION_CONFIDENCE_THRESHOLD and label in label_map:
                    student_id = label_map[label]
                    name = students.get(student_id, student_id)
                    display, color = f"{name} ({confidence:.0f})", (0, 255, 0)
                    if student_id not in marked_this_session and db.mark_attendance(student_id, name):
                        marked_this_session.add(name)
                else:
                    display, color = "Unknown", (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            status = ", ".join(sorted(marked_this_session)) if marked_this_session else "None yet"
            cv2.putText(frame, f"Marked: {status}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow("Attendance - press 'q' to stop", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return True, f"Session ended. Marked present: {len(marked_this_session)} student(s)."
