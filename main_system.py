"""
main_system.py — Entry point for the full gaze communication system.

Usage:
    python main_system.py           → calibrate if needed, then launch grid
    python main_system.py --recal   → force a new calibration session
    python main_system.py --tts     → test TTS audio only
"""

import sys
import argparse
import json
import datetime
import torch
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "session_status.json"

def _write_status(state: str, patient_name: str = "", started_at: str = "") -> None:
    STATUS_FILE.write_text(json.dumps({
        "state":       state,
        "patient_name": patient_name,
        "started_at":  started_at,
        "updated_at":  datetime.datetime.now().strftime("%H:%M:%S"),
    }))

from model        import GazeEstimationModel
from utils        import load_checkpoint
from dataset      import _eval_transform
from eye_detector import EyeDetector
from gaze_mapper  import GazeMapper, PersonalizedGazeMapper, run_calibration
from grid_ui      import GridUI
from tts_engine   import speak

CHECKPOINT = "checkpoints/best_model_v2.pth"


def load_model(device: torch.device) -> tuple:
    model = GazeEstimationModel(pretrained=False)
    load_checkpoint(CHECKPOINT, model, device=device)
    model.eval()
    transform = _eval_transform()
    return model, transform


def main():
    parser = argparse.ArgumentParser(description="Gaze Communication System")
    parser.add_argument("--recal", action="store_true",
                        help="Force a new calibration session")
    parser.add_argument("--tts",   action="store_true",
                        help="Test TTS audio and exit")
    parser.add_argument("--dwell", type=int, default=1500,
                        help="Dwell time in ms (default 1500)")
    parser.add_argument("--debug", action="store_true",
                        help="Show live pupil/screen coord overlay on grid")
    parser.add_argument("--flip",     action="store_true",
                        help="Flip camera feed horizontally (fix iVCam mirror)")
    parser.add_argument("--no-blink", action="store_true",
                        help="Pure dwell-only mode — no blink confirmation required")
    parser.add_argument("--shift-y",  type=int, default=0,
                        help="Shift gaze UP by N pixels (fix row-below confusion, e.g. --shift-y 200)")
    parser.add_argument("--patient-name", type=str, default="",
                        help="Patient name (passed by the Streamlit launcher)")
    args = parser.parse_args()
    patient_name = args.patient_name.strip()

    # ── TTS self-test ──────────────────────────────────────────────────────────
    if args.tts:
        print("TTS test — speaking all grid items …")
        from gaze_mapper import GRID_ITEMS
        import time
        for item in GRID_ITEMS:
            print(f"  {item}")
            speak(item, block=True)
            time.sleep(0.3)
        print("TTS test complete.")
        return

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
    print("[System] Loading gaze model …")
    model, transform = load_model(device)
    print("[System] Model loaded.")

    detector = EyeDetector(padding=0.30)
    mapper   = GazeMapper()

    # ── Calibration ────────────────────────────────────────────────────────────
    personalized_mapper = PersonalizedGazeMapper()
    if args.recal or not mapper.load():
        _write_status("calibrating", patient_name)
        print("[System] Starting calibration …")
        print("[System]   • A fullscreen window will open.")
        print("[System]   • Look at each red dot when it appears.")
        print("[System]   • Keep your head still during collection.")
        print("[System]   • Press ESC to cancel.")
        try:
            mapper, personalized_mapper = run_calibration(
                detector, model, transform, device, flip=args.flip
            )
        except KeyboardInterrupt:
            print("[System] Calibration cancelled. Exiting.")
            detector.release()
            return
    else:
        # Load personalized mapper if it exists alongside the polynomial one
        if not personalized_mapper.load():
            personalized_mapper = None

    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Main loop — recalibrate and relaunch whenever caregiver presses R ─────
    while True:
        speak("System ready.", block=True)
        _write_status("running", patient_name, started_at)

        print(f"[System] Launching communication grid  (dwell={args.dwell} ms) …")
        print("[System]   • Look at a word and hold your gaze for the dwell time.")
        print("[System]   • The system will speak the selected word.")
        print("[System]   • Hold eyes closed 2 s to pause/resume board.")
        print("[System]   • Press R to recalibrate  |  ESC to exit.")

        ui = GridUI(mapper, detector, model, transform, device,
                    dwell_ms=args.dwell, debug=args.debug, flip=args.flip,
                    personalized_mapper=personalized_mapper,
                    blink_mode=not args.no_blink,
                    shift_y=args.shift_y)
        ui.patient_name = patient_name

        recal_requested = ui.run()

        if not recal_requested:
            break   # ESC pressed — exit normally

        # R was pressed — run a fresh calibration then relaunch the grid
        _write_status("calibrating", patient_name)
        print("[System] Starting recalibration …")
        try:
            mapper, personalized_mapper = run_calibration(
                detector, model, transform, device, flip=args.flip
            )
        except KeyboardInterrupt:
            print("[System] Recalibration cancelled. Exiting.")
            break

    _write_status("done", patient_name, started_at)
    import pygame
    pygame.quit()
    detector.release()
    print("[System] Goodbye.")


if __name__ == "__main__":
    main()
