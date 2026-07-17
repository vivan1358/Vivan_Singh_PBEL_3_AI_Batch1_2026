"""
Evaluates recognition accuracy on a held-out test split of the captured
dataset. This is the piece that gives you real numbers to quote in a
project report: accuracy, a classification report, and a confusion
matrix, computed with scikit-learn on images the model never trained on.

Run after registering at least two students:
    python -m src.evaluate
"""
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config


def load_dataset():
    faces, labels, label_names = [], [], {}
    current_label = 0
    for student_id in sorted(os.listdir(config.DATASET_DIR)):
        student_path = os.path.join(config.DATASET_DIR, student_id)
        if not os.path.isdir(student_path):
            continue
        label_names[current_label] = student_id
        for img_name in sorted(os.listdir(student_path)):
            img = cv2.imread(os.path.join(student_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, config.FACE_IMG_SIZE))
            labels.append(current_label)
        current_label += 1
    return faces, np.array(labels), label_names


def evaluate_model(test_size=0.2, random_state=42):
    if not os.path.isdir(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        print("No dataset found. Register at least two students first.")
        return

    faces, labels, label_names = load_dataset()
    num_classes = len(set(labels.tolist()))
    if num_classes < 2:
        print("Need at least 2 registered students with samples to evaluate meaningfully.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, y_train)

    y_pred = [recognizer.predict(img)[0] for img in X_test]

    print(f"Train images: {len(X_train)}  |  Test images: {len(X_test)}  |  Students: {num_classes}")
    print(f"Held-out test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

    print("Label -> Student ID:")
    for label_id, student_id in sorted(label_names.items()):
        print(f"  {label_id}: {student_id}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
