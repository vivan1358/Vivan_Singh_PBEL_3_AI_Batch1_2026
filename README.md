# AI Based Smart Attendance Monitoring System

A desktop app that recognizes registered students through a webcam and
logs their attendance automatically — no roll call, no sign-in sheet.

## How it works (the ML part)

1. **Face detection** — OpenCV's Haar Cascade classifier locates a face
   in each webcam frame.
2. **Face recognition** — an **LBPH (Local Binary Patterns Histogram)**
   recognizer, trained on each student's captured face images, predicts
   *who* the detected face belongs to and how confident the match is.
3. **Decision** — if the confidence score is under a threshold
   (`config.RECOGNITION_CONFIDENCE_THRESHOLD`), that student is marked
   present for the day in a SQLite database (once per day, per student).

LBPH works by dividing a face into small regions, building a texture
histogram for each region, and comparing the combined histogram against
the ones seen during training. No GPU needed, trains in seconds — a
good fit for a classroom-sized dataset (tens of students).

`src/evaluate.py` is what gives you numbers for a project report: it
splits each student's images into train/test sets with scikit-learn,
retrains on the train split only, then prints accuracy, a
classification report, and a confusion matrix on the held-out test
images.

## Project structure

```
smart_attendance_system/
├── main.py                  # Entry point — run this
├── config.py                 # Paths, thresholds, camera index
├── requirements.txt
├── src/
│   ├── db.py                  # SQLite: students + attendance tables
│   ├── face_capture.py        # Registers a student (captures ~60 face samples)
│   ├── face_trainer.py        # Trains the LBPH recognizer on the dataset
│   ├── face_recognizer.py     # Live recognition + attendance marking
│   ├── evaluate.py            # Train/test accuracy report (run from terminal)
│   ├── reports.py             # CSV export + attendance % chart
│   └── gui.py                  # Tkinter desktop UI
├── dataset/                   # Captured face images (created on first run)
├── trained_model/             # Saved model + label map (created on first run)
└── attendance_records/        # Exported CSV files (created on first run)
```

## Setup

```bash
cd smart_attendance_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Linux:** if `import tkinter` fails, install it via your package manager
(it isn't always bundled):
```bash
sudo apt install python3-tk
```

**On WSL:** there's no direct webcam access by default. Either run this
with native Windows Python, or set up `usbipd-win` to pass the webcam
through. Native Windows or Linux with a built-in/USB webcam is the path
of least resistance.

**Install only one OpenCV package.** `opencv-contrib-python` includes
everything `opencv-python` does, plus the `cv2.face` module this project
needs. If `opencv-python` is already installed, remove it first:
```bash
pip uninstall opencv-python
```

## Running it

```bash
python main.py
```

Typical flow:
1. **Register New Student** — enter ID / name / roll no / branch, then
   look at the camera and slowly turn your head left/right until ~60
   samples are captured. Even, front-facing lighting helps a lot.
2. Repeat for every student.
3. **Train Recognition Model** — re-run any time you add or remove a
   student.
4. **Take Attendance** — opens the camera; recognized faces are marked
   present automatically. Press `q` to end the session.
5. **View Attendance Records** / **Attendance Report (Chart)** — review
   or export what's been logged.

For the accuracy numbers, run separately from a terminal:
```bash
python -m src.evaluate
```

## Tuning accuracy

- More samples per student (`config.FACE_SAMPLES_PER_STUDENT`) and
  varied head angles help more than almost anything else.
- Lower `RECOGNITION_CONFIDENCE_THRESHOLD` in `config.py` for stricter
  matching (fewer false positives, more false "Unknown"s); raise it if
  real students keep showing as "Unknown".
- Consistent lighting during both registration and attendance-taking
  makes a big practical difference.

## Extending this project

- Swap the Haar Cascade for OpenCV's DNN face detector (SSD) — more
  robust at odd angles and in poor lighting.
- Swap LBPH for deep face embeddings (`face_recognition`/dlib, or a
  FaceNet-style model) plus an SVM/KNN classifier — higher accuracy,
  heavier install.
- Add a liveness check (e.g. requiring a blink or head turn) so a
  printed photo can't be used to fake attendance.
- No built-in webcam? Point `config.CAMERA_INDEX`'s VideoCapture at a
  phone-as-IP-camera URL instead: `cv2.VideoCapture("http://<ip>:<port>/video")`.
- Move the GUI to a Flask/Streamlit web app for remote access, or email
  a daily attendance summary automatically.

## A note on consent

Only capture face data for people who've agreed to it, and let them
know how the images and attendance records will be stored and used —
worth doing for any face-recognition project, not just this one.
