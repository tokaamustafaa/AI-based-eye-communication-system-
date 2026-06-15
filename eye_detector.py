"""
eye_detector.py — Real-time eye region extraction using MediaPipe Face Landmarker.

Works with MediaPipe 0.10+ (Tasks API).
Model file: checkpoints/face_landmarker.task  (auto-downloaded on first run)

Responsibilities:
    1. Receive a raw BGR camera frame from OpenCV
    2. Detect the face using MediaPipe FaceLandmarker (478 landmarks)
    3. Crop both eye regions with padding
    4. Return cropped eyes as PIL Images (ready for _eval_transform → model)

Landmark naming convention (MediaPipe uses the PERSON's perspective):
    "right eye" = person's right eye = camera's LEFT side
    "left eye"  = person's left eye  = camera's RIGHT side
"""

import os
import urllib.request

import cv2
from camera_setup import open_camera
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python         import vision
from mediapipe.tasks.python.vision  import FaceLandmarker, FaceLandmarkerOptions, RunningMode


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "face_landmarker.task"
)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print(f"[EyeDetector] Downloading face_landmarker.task …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[EyeDetector] Saved → {_MODEL_PATH}")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# Eye contour landmark indices (same numbering as old FaceMesh 478-point model)
# ---------------------------------------------------------------------------

# Person's RIGHT eye (appears on left side of the camera frame)
_RIGHT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

# Person's LEFT eye (appears on right side of the camera frame)
_LEFT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

# Fixed eye-corner landmark indices for iris gaze normalization.
# These are bone/tendon anchor points that do NOT move with the iris.
# Right eye (camera LEFT):  outer=33 (temporal), inner=133 (nasal)
# Left eye  (camera RIGHT): inner=362 (nasal),   outer=263 (temporal)
_R_CORNER_L, _R_CORNER_R = 33,  133   # left/right in image for right eye
_R_LID_TOP,  _R_LID_BOT  = 159, 145   # upper / lower eyelid centre
_R_IRIS                   = 468

_L_CORNER_L, _L_CORNER_R = 362, 263   # left/right in image for left eye
_L_LID_TOP,  _L_LID_BOT  = 386, 374
_L_IRIS                   = 473


# ---------------------------------------------------------------------------
# EyeDetector
# ---------------------------------------------------------------------------

class EyeDetector:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API) to extract cropped eye images.

    Args:
        padding                  : fractional padding around the eye bounding box
        min_detection_confidence : face detection confidence threshold
        min_tracking_confidence  : face tracking confidence threshold
    """

    def __init__(self,
                 padding: float = 0.30,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float  = 0.5):
        self.padding = padding

        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one BGR camera frame.

        Returns a dict:
            face_detected    : bool
            right_eye        : PIL Image (person's right) or None
            left_eye         : PIL Image (person's left)  or None
            right_eye_box    : (x1,y1,x2,y2) pixel rect or None
            left_eye_box     : (x1,y1,x2,y2) pixel rect or None
            iris_gaze        : (gaze_x, gaze_y) normalised [0,1] within eye box, or None
            landmarks        : raw landmark list or None
            annotated_frame  : BGR frame with eye boxes drawn
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = self._landmarker.detect(mp_image)

        annotated: np.ndarray = frame_bgr.copy()
        out = {
            "face_detected"  : False,
            "right_eye"      : None,
            "left_eye"       : None,
            "right_eye_box"  : None,
            "left_eye_box"   : None,
            "iris_gaze"      : None,
            "landmarks"      : None,
            "annotated_frame": annotated,
        }

        if not result.face_landmarks:
            cv2.putText(annotated, "No face detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return out

        landmarks = result.face_landmarks[0]   # list of NormalizedLandmark
        out["face_detected"] = True
        out["landmarks"]     = landmarks

        h, w = frame_bgr.shape[:2]

        r_crop, r_box = self._crop_eye(frame_rgb, landmarks, _RIGHT_EYE_LANDMARKS, h, w)
        l_crop, l_box = self._crop_eye(frame_rgb, landmarks, _LEFT_EYE_LANDMARKS,  h, w)

        out["right_eye"]     = r_crop
        out["left_eye"]      = l_crop
        out["right_eye_box"] = r_box
        out["left_eye_box"]  = l_box

        # Iris gaze: normalize iris position within its OWN eye corners.
        # Using per-eye corners gives a much larger range than biocular width.
        if len(landmarks) > 473 and (r_box or l_box):
            use_right = r_box is not None

            if use_right:
                iris_i    = _R_IRIS
                corner_l  = 33    # outer corner (temple side)
                corner_r  = 133   # inner corner (nose side)
                lid_top   = 159   # upper lid center
                lid_bot   = 145   # lower lid center
            else:
                iris_i    = _L_IRIS
                corner_l  = 362   # inner corner (nose side)
                corner_r  = 263   # outer corner (temple side)
                lid_top   = 386
                lid_bot   = 374

            iris_nx = landmarks[iris_i].x
            iris_ny = landmarks[iris_i].y

            # Horizontal: between this eye's own corners
            cx1 = landmarks[corner_l].x
            cx2 = landmarks[corner_r].x
            eye_w = abs(cx2 - cx1)

            # Vertical: between this eye's own lids
            top_y = landmarks[lid_top].y
            bot_y = landmarks[lid_bot].y
            eye_h = abs(bot_y - top_y)

            if eye_w > 0.01 and eye_h > 0.001:
                # X: normalize within eye corners
                gaze_x = (iris_nx - min(cx1, cx2)) / eye_w

                # Y: use nose bridge as fixed reference, flip so up=small, down=large
                nose_y = landmarks[6].y
                gaze_y = 0.5 - (iris_ny - nose_y) / eye_w  # flipped

                out["iris_gaze"] = (
                    float(max(0.0, min(1.0, gaze_x))),
                    float(max(0.0, min(1.0, gaze_y))),
                )

        self._draw_overlay(annotated, r_box, l_box)
        return out

    def release(self) -> None:
        """Free MediaPipe resources."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_eye(self,
                  frame_rgb: np.ndarray,
                  landmarks: list,
                  indices: list,
                  h: int,
                  w: int) -> tuple:
        """
        Compute bounding box from landmark indices, add padding, crop.
        Returns (PIL Image, (x1,y1,x2,y2)) or (None, None).
        """
        xs = [int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        eye_w = x_max - x_min
        eye_h = y_max - y_min

        if eye_w < 4 or eye_h < 2:
            return None, None

        pad_x = int(eye_w * self.padding)
        pad_y = int(eye_h * self.padding)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w, x_max + pad_x)
        y2 = min(h, y_max + pad_y)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        return Image.fromarray(crop), (x1, y1, x2, y2)

    def _draw_overlay(self, frame: np.ndarray, r_box, l_box) -> None:
        """Draw eye bounding boxes on the annotated frame."""
        if r_box:
            cv2.rectangle(frame, (r_box[0], r_box[1]),
                          (r_box[2], r_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, "R", (r_box[0], r_box[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if l_box:
            cv2.rectangle(frame, (l_box[0], l_box[1]),
                          (l_box[2], l_box[3]), (255, 100, 0), 2)
            cv2.putText(frame, "L", (l_box[0], l_box[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.putText(frame, "Face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# BlinkDetector
# ---------------------------------------------------------------------------

class BlinkDetector:
    """
    Detects deliberate slow blinks using the Eye Aspect Ratio (EAR).

    EAR = vertical_eye_opening / horizontal_eye_width

    Thresholds adapt automatically by watching the user's own EAR range over
    the first 30 frames — no hardcoded scale. Works for any eye size, camera
    distance, or phone model.

    Natural involuntary blinks (<150 ms) are silently ignored.
    A deliberate "confirm" blink (150–700 ms) fires the "confirm_blink" event.
    """

    MIN_MS       = 150   # shortest confirm blink
    MAX_MS       = 700   # longest confirm blink
    LONG_BLINK_MS = 3000 # hold eyes closed this long → toggle board on/off

    def __init__(self):
        self._closed_since:     float | None = None
        self._is_closed:        bool         = False
        self._long_blink_fired: bool         = False   # prevent double-fire
        self._ear_history:      list         = []
        self._ear_close:        float        = 0.0
        self._ear_open:         float        = 0.0
        self._baseline_ready:   bool         = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_eye_closing(self, landmarks) -> bool:
        """
        Returns True only when the eye is physically closing AND we have a
        reliable per-user baseline. Before baseline is ready always returns
        False so gaze is never accidentally frozen.
        """
        if landmarks is None or not self._baseline_ready:
            return False
        return self._ear(landmarks) < self._ear_close

    def update(self, landmarks, h: int, w: int) -> str:
        """
        Call once per frame with the current MediaPipe landmarks.

        Returns:
            "open"          – eyes open
            "closing"       – eye currently closed, timer running
            "confirm_blink" – deliberate blink (150–700 ms)   → confirm selection
            "long_blink"    – held closed ≥ 3 s               → toggle board on/off
            "ignore_blink"  – too short                        → discard
            "no_face"       – no landmarks
        """
        import time
        if landmarks is None:
            self._closed_since     = None
            self._is_closed        = False
            self._long_blink_fired = False
            return "no_face"

        ear = self._ear(landmarks)
        self._adapt(ear)
        now = time.time()

        if not self._baseline_ready:
            return "open"

        if not self._is_closed and ear < self._ear_close:
            self._is_closed        = True
            self._closed_since     = now
            self._long_blink_fired = False
            return "closing"

        if self._is_closed:
            duration_ms = (now - self._closed_since) * 1000

            # Fire long_blink exactly once while the eye is still held closed
            if duration_ms >= self.LONG_BLINK_MS and not self._long_blink_fired:
                self._long_blink_fired = True
                return "long_blink"

            if ear < self._ear_open:
                return "closing"

            # Eye just opened — classify the completed blink
            fired              = self._long_blink_fired
            self._is_closed        = False
            self._closed_since     = None
            self._long_blink_fired = False

            if fired:
                return "open"   # long_blink already handled above
            if self.MIN_MS <= duration_ms <= self.MAX_MS:
                return "confirm_blink"
            return "ignore_blink"

        return "open"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapt(self, ear: float) -> None:
        """
        Collect all EAR values and derive thresholds from the distribution.

        After 30 frames the top-third of values represents eyes-open.
        Close threshold = 55 % of that baseline.
        Open threshold  = 72 % of that baseline.

        This works regardless of camera distance or eye size because it uses
        the user's own maximum EAR as the reference, not a hardcoded number.
        """
        self._ear_history.append(ear)
        if len(self._ear_history) > 90:
            self._ear_history = self._ear_history[-90:]

        if len(self._ear_history) >= 30:
            s = sorted(self._ear_history, reverse=True)
            n_open = max(1, len(s) // 3)       # top third = eyes open
            baseline = sum(s[:n_open]) / n_open
            self._ear_close       = baseline * 0.65   # freeze early — catches iris drift on upward gaze
            self._ear_open        = baseline * 0.82
            self._baseline_ready  = True

    def _ear(self, landmarks) -> float:
        """Simplified EAR from right-eye top/bottom lids and inner/outer corners."""
        try:
            top_y   = landmarks[_R_LID_TOP].y
            bot_y   = landmarks[_R_LID_BOT].y
            left_x  = landmarks[_R_CORNER_L].x
            right_x = landmarks[_R_CORNER_R].x
            horiz   = abs(right_x - left_x)
            if horiz < 1e-5:
                return 0.3
            return abs(top_y - bot_y) / horiz
        except (IndexError, AttributeError):
            return 0.3   # assume open on error


# ---------------------------------------------------------------------------
# Live demo  (python eye_detector.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    import torch
    from model   import GazeEstimationModel
    from utils   import load_checkpoint
    from dataset import _eval_transform

    CHECKPOINT = "checkpoints/best_model_v2.pth"
    device     = torch.device("cpu")

    gaze_model = GazeEstimationModel(pretrained=False)
    load_checkpoint(CHECKPOINT, gaze_model, device=device)
    gaze_model.eval()
    transform = _eval_transform()
    print("Model loaded. Press Q to quit.")

    detector = EyeDetector(padding=0.30)
    cap      = open_camera()

    if not cap.isOpened():
        print("ERROR: cannot open camera.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result  = detector.detect(frame)
        display = result["annotated_frame"]
        eye_img = result["right_eye"] or result["left_eye"]
        label   = "right" if result["right_eye"] else "left"

        if eye_img is not None:
            with torch.no_grad():
                inp  = transform(eye_img).unsqueeze(0)
                pred = gaze_model(inp).squeeze().tolist()

            cv2.putText(display,
                        f"Pupil ({label}):  x={pred[0]:.3f}  y={pred[1]:.3f}",
                        (20, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

            box = result["right_eye_box"] if result["right_eye"] else result["left_eye_box"]
            if box:
                bx1, by1, bx2, by2 = box
                bw, bh = bx2 - bx1, by2 - by1
                cv2.circle(display,
                           (int(bx1 + pred[0] * bw), int(by1 + pred[1] * bh)),
                           5, (0, 0, 255), -1)

        cv2.imshow("Eye Detector — press Q to quit", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.release()
    cv2.destroyAllWindows()