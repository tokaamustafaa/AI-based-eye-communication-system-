# Eye Communication System

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

### 1. Launch the Streamlit UI

```bash
pip install streamlit
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

## Notes

- `.claude/` contains local editor settings and is ignored from the repository.
- `memory/project_gaze_communication.md` contains project research notes and is included intentionally.
- There is no `requirements.txt` in this repo yet; install dependencies manually as needed.

## Recommended GitHub setup

- Keep `checkpoints/` and `archive/` out of the repo if they contain large trained models or data files.
- Add additional ignored files as needed for your local environment.

## Info needed from you

Please confirm or provide:

- Project title and short description you want shown at the top
- Your name or organization for the author section
- Any required dependencies or a `requirements.txt` you want included
- Preferred license (MIT, Apache 2.0, GPL, etc.)
