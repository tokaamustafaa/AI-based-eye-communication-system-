---
name: project-gaze-communication
description: ALS graduation project — EfficientNet-B0 gaze tracking + pygame 3×3 communication grid with TTS. Best stable state documented here.
metadata:
  type: project
---

# Gaze Communication System — Graduation Project

**Why:** Graduation project for ALS/paralyzed patients — lets them communicate by looking at a word grid and blinking to confirm.

**How to apply:** Always preserve the full pipeline: EfficientNet-B0 pupil regression → GazeMapper polynomial calibration → PersonalizedGazeMapper (Ridge on 1280-dim backbone features) → pygame 3×3 grid with TTS.

## Best-Result Milestone (2026-06-09)

The following configuration gave the best observed results before further blink improvements:

### Files and key settings

- **`eye_detector.py`** — `BlinkDetector` uses rolling 90-frame EAR history; top-third = open-eye baseline; `_ear_close = baseline × 0.55`, `_ear_open = baseline × 0.72`; freeze disabled until 30 frames collected.
- **`grid_ui.py`** — `EMA_ALPHA = 0.20` (gaze dot smoothing), `CELL_VOTE_N = 8` (majority-vote cell stability), blink-freeze uses `self._ema_sx/ema_sy` instead of old `_sx_buf/_sy_buf`.
- **`gaze_mapper.py`** — calibration EAR filter uses same adaptive logic (`_cal_ear_history`, `_cal_ear_close()`), not hardcoded 0.18.

### Known remaining issue at this milestone

Blink-to-confirm success rate ~70%. The 30% failure is specifically at top-row cells (0, 1, 2): blinking causes iris to jump downward (MediaPipe artifact when looking up), which moves the gaze dot to the cell below. Two sub-causes:
1. During dwell phase — blink-induced iris jump resets dwell timer to wrong cell.
2. During confirm phase — iris drift briefly leaves pre-selected cell, risks grace-period cancellation.

## Architecture

- `main_system.py` → runs calibration → launches `GridUI`
- `run_calibration()` in `gaze_mapper.py` → 9-point grid, 30 samples/point, filters blink frames via adaptive EAR
- Returns `(GazeMapper, PersonalizedGazeMapper)` tuple
- `GridUI._get_gaze_cell()` → EfficientNet forward pass → EMA-smoothed screen coords → cell vote → dwell state machine
- Two-stage selection: 800ms dwell → orange pre-select → deliberate blink (150–700ms) → TTS

## Run commands

```
python main_system.py --recal --flip --debug   # recalibrate + debug overlay
python main_system.py --flip                    # normal run after calibration
python main_system.py --flip --no-blink         # dwell-only mode (no blink needed)
python main_system.py --flip --shift-y 200      # if gaze reads one row low
```
