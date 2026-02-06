# Active Speaker Detection (CNN + MediaPipe)

This folder contains a step-by-step pipeline for building a Visual Voice Activity Detection (VVAD) system using facial motion.

## Steps

- Step 1: CSV ↔ video timestamp alignment (`scripts/step1_check_alignment.py`)
- Step 2: AVA bbox → face crops (`scripts/step2_crop_face.py`)
- Step 3: face crops → MediaPipe FaceLandmarker landmarks (`scripts/step3_extract_landmarks.py`)
- Step 4: landmarks → fixed-length windows for CNN training (`scripts/step4_build_windows.py`)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## MediaPipe model (required for Step 3)

Step 3 uses MediaPipe Tasks `FaceLandmarker` and requires a `.task` model file.

- Put the model at `data/models/face_landmarker_v2.task`

## Notes

- `data/` contains large inputs (videos/labels) and generated artifacts (crops/landmarks/windows) and is ignored by git.
