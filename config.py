import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")
DB_PATH = os.path.join(BASE_DIR, "attendance_system.db")
EXPORT_DIR = os.path.join(BASE_DIR, "attendance_records")

CAMERA_INDEX = 0
FACE_SAMPLES_PER_STUDENT = 60
FACE_IMG_SIZE = (200, 200)

# LBPH predict() returns a distance, not a probability: LOWER = better match.
# Tighten this if strangers get marked present; loosen it if real students
# keep showing up as "Unknown".
RECOGNITION_CONFIDENCE_THRESHOLD = 70


def ensure_dirs():
    """Create the runtime folders if they don't exist yet."""
    for d in (DATASET_DIR, MODEL_DIR, EXPORT_DIR):
        os.makedirs(d, exist_ok=True)
