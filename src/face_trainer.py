"""
Trains an LBPH (Local Binary Patterns Histogram) face recognizer on
every image under dataset/<student_id>/*.jpg, then saves the model and
a label map (numeric label -> student_id) for face_recognizer.py to use.
"""
import os
import pickle

import cv2
import numpy as np

import config


def train_model():
    if not os.path.isdir(config.DATASET_DIR) or not os.listdir(config.DATASET_DIR):
        return False, "No dataset found. Register at least one student first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}  # numeric_label -> student_id
    current_label = 0

    for student_id in sorted(os.listdir(config.DATASET_DIR)):
        student_path = os.path.join(config.DATASET_DIR, student_id)
        if not os.path.isdir(student_path):
            continue

        label_map[current_label] = student_id
        for img_name in sorted(os.listdir(student_path)):
            img = cv2.imread(os.path.join(student_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(cv2.resize(img, config.FACE_IMG_SIZE))
            labels.append(current_label)
        current_label += 1

    if not faces:
        return False, "No usable face images found in the dataset."

    recognizer.train(faces, np.array(labels))
    recognizer.save(config.MODEL_PATH)

    with open(config.LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    return True, f"Model trained on {len(faces)} images across {len(label_map)} student(s)."
