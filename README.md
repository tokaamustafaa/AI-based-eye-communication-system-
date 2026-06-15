# AI-Based Eye Gaze Communication System for Paralysis Patients

Assistive gaze communication system for ALS/paralyzed users.
This graduation project uses eye-tracking, gaze mapping, and blink confirmation to let users select words from a 3×3 communication grid.

## What it does

- Uses `EfficientNet-B0` for pupil/gaze regression
- Maps gaze to screen cells with adaptive calibration
- Runs a pygame 3×3 communication grid
- Supports blink-to-confirm selection
- Includes a Streamlit launcher and a live operational system

## Key files

- `app.py` - Streamlit launcher for landing page, patient setup, and live session dashboard
- `main_system.py` - Main runtime entrypoint for calibration and grid interface
- `gaze_mapper.py` - Calibration and gaze mapping logic
- `eye_detector.py` - Blink and eye state detection
- `grid_ui.py` - Grid UI, gaze-to-cell voting, and selection logic
- `memory/project_gaze_communication.md` - Project notes and milestone details

## Run the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit UI

```bash
streamlit run app.py
```

This opens the landing page and can launch the gaze system from the browser.

### 2. Run the live gaze system directly

```bash
python main_system.py --flip
```

### Common runtime options

```bash
python main_system.py --recal --flip --debug
python main_system.py --flip --no-blink
python main_system.py --flip --shift-y 200
```

## Dependencies

The main dependencies are listed in `requirements.txt`.

Recommended libraries:

```bash
streamlit
torch
torchvision
opencv-python
numpy
pygame
matplotlib
mediapipe
requests
```




