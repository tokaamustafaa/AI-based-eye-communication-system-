"""
gaze_mapper.py — Calibration and gaze-to-screen coordinate mapping.

The trained model outputs pupil position INSIDE the cropped eye image [0,1].
This module learns the non-linear mapping:

    (pupil_x, pupil_y) → (screen_x_pixels, screen_y_pixels)

using a one-time calibration session where the patient looks at 9 known
screen positions (the center of each grid cell).

Typical workflow:
    mapper = run_calibration(detector, model, transform, device)
    # — or, for subsequent sessions —
    mapper = GazeMapper(); mapper.load()

    screen_x, screen_y = mapper.predict(pupil_x, pupil_y)
"""

import os
from typing import Any
import json
import numpy as np
import joblib

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model  import Ridge
from sklearn.pipeline      import make_pipeline


CALIBRATION_FILE     = "checkpoints/gaze_calibration.pkl"
PERSONALIZED_CAL_FILE = "checkpoints/gaze_personalized.pkl"

_GRID_DEFAULTS: list[str] = [
    "YES", "NO", "FOOD",
    "PAIN", "TOILET", "SLEEP",
    "CALL", "WATER", "LIGHT",
]


def _load_grid_items() -> list[str]:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "grid_config.json"
    )
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                items = json.load(f).get("grid_items", [])
            if len(items) == 9:
                return [str(x) for x in items]
        except Exception:
            pass
    return _GRID_DEFAULTS[:]


# 3×3 grid labels in row-major order (top-left → bottom-right).
# Edit grid_config.json to change vocabulary without touching source code.
GRID_ITEMS: list[str] = _load_grid_items()


# ---------------------------------------------------------------------------
# GazeMapper
# ---------------------------------------------------------------------------

class GazeMapper:
    """
    Polynomial regression mapper: (pupil_x, pupil_y) → (screen_x, screen_y).

    Uses degree-2 polynomial features so it can handle the non-linear
    relationship between eye rotation and screen position.
    With 9 calibration points and 6 polynomial features the system is
    slightly over-determined — Ridge regularisation keeps it stable.
    """

    def __init__(self):
        self._pipeline_x: Any = None   # predicts screen_x
        self._pipeline_y: Any = None   # predicts screen_y
        self._calibrated: bool   = False
        self._pupil_pts:  list   = []     # raw calibration data (kept for diagnostics)
        self._screen_pts: list   = []

        # Normalization range — set from calibration data in fit().
        # Stretches the observed pupil range to fill [0,1] so the polynomial
        # mapping works correctly even when the camera produces a compressed
        # gaze range (e.g. iVCam phone cameras, unusual viewing distances).
        self._px_min: float = 0.0
        self._px_max: float = 1.0
        self._py_min: float = 0.0
        self._py_max: float = 1.0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _normalize(self, pupil_x: float, pupil_y: float) -> tuple[float, float]:
        """Stretch observed pupil coords into [0,1] using calibration range."""
        rx = self._px_max - self._px_min
        ry = self._py_max - self._py_min
        nx = (pupil_x - self._px_min) / rx if rx > 1e-6 else 0.5
        ny = (pupil_y - self._py_min) / ry if ry > 1e-6 else 0.5
        return nx, ny

    def predict(self, pupil_x: float, pupil_y: float) -> tuple[float, float]:
        """
        Map normalised pupil coords → screen pixel position.

        Returns:
            (screen_x, screen_y) as floats
        Raises:
            RuntimeError if calibration has not been done yet.
        """
        if not self._calibrated:
            raise RuntimeError(
                "GazeMapper not calibrated. Call fit() or load() first."
            )
        # Normalize to the range observed during calibration so the polynomial
        # generalises correctly across different cameras and viewing distances.
        pupil_x, pupil_y = self._normalize(pupil_x, pupil_y)
        feat = np.array([[pupil_x, pupil_y]])
        sx   = float(self._pipeline_x.predict(feat)[0])
        sy   = float(self._pipeline_y.predict(feat)[0])
        return sx, sy

    def is_calibrated(self) -> bool:
        return self._calibrated

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, pupil_points: list, screen_points: list):
        """
        Learn the mapping from calibration pairs.

        Args:
            pupil_points  : [(px, py), ...]  pupil coords in [0, 1]
            screen_points : [(sx, sy), ...]  matching screen pixel positions
        """
        assert len(pupil_points) == len(screen_points), \
            "pupil_points and screen_points must have the same length."
        assert len(pupil_points) >= 4, \
            "Need at least 4 calibration points for a reliable fit."

        self._pupil_pts  = list(pupil_points)
        self._screen_pts = list(screen_points)

        # Compute the observed pupil range from calibration data and store it
        # so predict() can normalise live predictions to the same [0,1] space.
        xs = [p[0] for p in pupil_points]
        ys = [p[1] for p in pupil_points]
        self._px_min = float(np.min(xs))
        self._px_max = float(np.max(xs))
        self._py_min = float(np.min(ys))
        self._py_max = float(np.max(ys))
        print(f"[GazeMapper] Pupil range  "
              f"x=[{self._px_min:.3f}, {self._px_max:.3f}]  "
              f"y=[{self._py_min:.3f}, {self._py_max:.3f}]")

        # Fit the polynomial on normalised coords so the mapping is stable
        # regardless of the absolute pupil range.
        normed = [self._normalize(p[0], p[1]) for p in pupil_points]
        X  = np.array(normed)                                   # (N, 2)  [0,1]
        Yx = np.array([p[0] for p in screen_points], float)    # (N,) screen_x
        Yy = np.array([p[1] for p in screen_points], float)    # (N,) screen_y

        def _make_pipe():
            return make_pipeline(
                PolynomialFeatures(degree=2, include_bias=True),
                Ridge(alpha=0.1),
            )

        self._pipeline_x = _make_pipe()
        self._pipeline_y = _make_pipe()
        self._pipeline_x.fit(X, Yx)
        self._pipeline_y.fit(X, Yy)
        self._calibrated = True

        # Report fit quality
        pred_sx = self._pipeline_x.predict(X)
        pred_sy = self._pipeline_y.predict(X)
        err_px  = np.sqrt(((pred_sx - Yx)**2 + (pred_sy - Yy)**2)).mean()
        print(f"[GazeMapper] Fitted on {len(pupil_points)} points. "
              f"Train residual: {err_px:.1f} px")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str = CALIBRATION_FILE):
        """Persist the fitted pipelines and raw calibration data to disk."""
        if not self._calibrated:
            raise RuntimeError("Cannot save: not calibrated yet.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "pipeline_x": self._pipeline_x,
            "pipeline_y": self._pipeline_y,
            "pupil_pts":  self._pupil_pts,
            "screen_pts": self._screen_pts,
            "px_min":     self._px_min,
            "px_max":     self._px_max,
            "py_min":     self._py_min,
            "py_max":     self._py_max,
        }, path)
        print(f"[GazeMapper] Saved → {path}")

    def load(self, path: str = CALIBRATION_FILE) -> bool:
        """
        Load a previously saved calibration.

        Returns True on success, False if file does not exist.
        """
        if not os.path.exists(path):
            print(f"[GazeMapper] No calibration file at {path}")
            return False
        data = joblib.load(path)
        self._pipeline_x = data["pipeline_x"]
        self._pipeline_y = data["pipeline_y"]
        self._pupil_pts  = data.get("pupil_pts", [])
        self._screen_pts = data.get("screen_pts", [])
        self._px_min     = data.get("px_min", 0.0)
        self._px_max     = data.get("px_max", 1.0)
        self._py_min     = data.get("py_min", 0.0)
        self._py_max     = data.get("py_max", 1.0)
        self._calibrated = True
        print(f"[GazeMapper] Loaded ← {path}  "
              f"({len(self._pupil_pts)} calibration points)  "
              f"pupil range x=[{self._px_min:.3f},{self._px_max:.3f}] "
              f"y=[{self._py_min:.3f},{self._py_max:.3f}]")
        return True

    # ------------------------------------------------------------------
    # Grid hit test
    # ------------------------------------------------------------------

    def screen_to_grid_cell(self,
                             screen_x: float,
                             screen_y: float,
                             screen_w: int,
                             screen_h: int,
                             margin_x: int = 100,
                             margin_y: int = 80) -> int | None:
        """
        Map a screen position to one of the 9 grid cells (0-indexed, row-major).

        Returns the cell index [0..8], or None if outside all cells.
        """
        grid_w = (screen_w - 2 * margin_x) / 3
        grid_h = (screen_h - 2 * margin_y) / 3

        col = int((screen_x - margin_x) / grid_w)
        row = int((screen_y - margin_y) / grid_h)

        if 0 <= col < 3 and 0 <= row < 3:
            return row * 3 + col
        return None


# ---------------------------------------------------------------------------
# PersonalizedGazeMapper — feature-space adaptation
# ---------------------------------------------------------------------------

class PersonalizedGazeMapper:
    """
    User-adaptive mapper: EfficientNet backbone features (1280-dim) → screen coords.

    Fitted during calibration with Ridge regression on all collected eye images.
    No gradient updates — just a linear model fit in milliseconds.

    Why this beats the polynomial mapper for new users:
      - The 1280-dim feature vector carries the full eye appearance (texture, shape,
        context) — not just the 2-dim pupil summary that the polynomial uses.
      - Ridge regression finds the optimal linear combination of those features
        for *this person* and *this camera*, adapting to different optics and
        eye appearances automatically.
      - Generalises well from ~360 samples because the backbone was pretrained on
        gaze data and produces structured, low-noise features.
    """

    def __init__(self):
        self._pipe_x:    Any  = None
        self._pipe_y:    Any  = None
        self._calibrated: bool = False

    def fit(self, feat_vecs: list, screen_points: list) -> None:
        """
        Args:
            feat_vecs     : [(1280-dim np.ndarray), ...]  one per calibration sample
            screen_points : [(sx, sy), ...]               matching screen pixel positions
        """
        assert len(feat_vecs) == len(screen_points), \
            "feat_vecs and screen_points must have the same length."
        assert len(feat_vecs) >= 9, \
            "Need at least 9 feature vectors."

        X  = np.array(feat_vecs)                                 # (N, 1280)
        Yx = np.array([p[0] for p in screen_points], float)     # (N,)  screen_x
        Yy = np.array([p[1] for p in screen_points], float)     # (N,)  screen_y

        def _pipe():
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

        self._pipe_x = _pipe()
        self._pipe_y = _pipe()
        self._pipe_x.fit(X, Yx)
        self._pipe_y.fit(X, Yy)
        self._calibrated = True

        pred_x = self._pipe_x.predict(X)
        pred_y = self._pipe_y.predict(X)
        err = float(np.sqrt(((pred_x - Yx) ** 2 + (pred_y - Yy) ** 2)).mean())
        print(f"[PersonalizedMapper] Fitted {len(feat_vecs)} samples "
              f"({X.shape[1]}-dim).  Train residual: {err:.1f} px")

    def predict(self, feat_vec) -> tuple[float, float]:
        if not self._calibrated:
            raise RuntimeError("PersonalizedGazeMapper not calibrated.")
        X  = np.array(feat_vec).reshape(1, -1)
        sx = float(self._pipe_x.predict(X)[0])
        sy = float(self._pipe_y.predict(X)[0])
        return sx, sy

    def is_calibrated(self) -> bool:
        return self._calibrated

    def save(self, path: str = PERSONALIZED_CAL_FILE) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"pipe_x": self._pipe_x, "pipe_y": self._pipe_y}, path)
        print(f"[PersonalizedMapper] Saved → {path}")

    def load(self, path: str = PERSONALIZED_CAL_FILE) -> bool:
        if not os.path.exists(path):
            print(f"[PersonalizedMapper] No personalized calibration at {path}")
            return False
        data = joblib.load(path)
        self._pipe_x = data["pipe_x"]
        self._pipe_y = data["pipe_y"]
        self._calibrated = True
        print(f"[PersonalizedMapper] Loaded ← {path}")
        return True


# ---------------------------------------------------------------------------
# Screen layout helper (shared with grid_ui.py)
# ---------------------------------------------------------------------------

def build_calibration_points(screen_w: int,
                              screen_h: int,
                              margin_x: int = 100,
                              margin_y: int = 80) -> list:
    """
    Return 9 (sx, sy) positions that exactly match the 3×3 grid cell centers.
    These are used both as calibration dots and as the mapping target.
    """
    grid_w = (screen_w - 2 * margin_x) / 3
    grid_h = (screen_h - 2 * margin_y) / 3

    pts = []
    for row in range(3):
        for col in range(3):
            cx = int(margin_x + col * grid_w + grid_w / 2)
            cy = int(margin_y + row * grid_h + grid_h / 2)
            pts.append((cx, cy))
    return pts


# ---------------------------------------------------------------------------
# Interactive calibration (pygame fullscreen)
# Called from main_system.py — NOT from the notebook
# ---------------------------------------------------------------------------

def run_calibration(detector,
                    gaze_model=None,
                    transform=None,
                    device=None,
                    n_samples_per_point: int = 40,
                    dwell_ms: int = 1500,
                    save_path: str = CALIBRATION_FILE,
                    flip: bool = False) -> "GazeMapper":
    """
    Fullscreen 3x3 grid calibration.

    The actual grid is shown at all times:
      - Pending cells are dimmed.
      - The active cell has a pulsing red dot — patient looks at it.
      - Collecting: a cyan arc fills as samples are gathered.
      - Done cells turn green.

    Caregiver presses SPACE to start collecting each point.
    Press ESC at any time to cancel.

    Returns a fitted and saved GazeMapper.
    """
    import math
    import time
    import pygame
    import cv2
    from camera_setup import open_camera
    import os

    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    os.environ.setdefault("SDL_VIDEO_FULLSCREEN_DISPLAY", "0")

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SW, SH = screen.get_size()
    clock  = pygame.time.Clock()
    pygame.display.set_caption("Gaze Calibration")

    font_label  = pygame.font.SysFont("Arial", 54, bold=True)
    font_sub    = pygame.font.SysFont("Arial", 26)
    font_status = pygame.font.SysFont("Arial", 21)

    # ── Grid layout (identical to GridUI) ─────────────────────────────────
    MARGIN_X, MARGIN_Y = 100, 80
    gw = (SW - 2 * MARGIN_X) / 3
    gh = (SH - 2 * MARGIN_Y) / 3
    cells = []
    for row in range(3):
        for col in range(3):
            cells.append(pygame.Rect(
                int(MARGIN_X + col * gw),
                int(MARGIN_Y + row * gh),
                int(gw), int(gh),
            ))

    cal_points = build_calibration_points(SW, SH)

    # ── Colour palette ─────────────────────────────────────────────────────
    BG          = (12,  14,  22)
    CELL_IDLE   = (22,  26,  42)
    CELL_ACTIVE = (38,  50,  85)
    CELL_DONE   = (15,  90,  50)
    CELL_WARN   = (75,  50,   5)
    BORDER_IDLE = (38,  44,  65)
    BORDER_ACT  = (210, 215, 240)
    BORDER_DONE = (0,   210,  90)
    BORDER_WARN = (255, 180,   0)
    TEXT_BRIGHT = (215, 220, 238)
    TEXT_DIM    = (65,  70,  95)
    RED_DOT     = (255,  45,  45)
    CYAN_ARC    = (0,   200, 255)
    GREEN_TAG   = (0,   230, 110)

    cap       = open_camera()
    pupil_pts : list = []
    screen_pts: list = []
    done_set  : set  = set()

    # Feature vectors for PersonalizedGazeMapper (one per accepted sample)
    feat_vecs_all  : list = []   # (N_total, 1280) accumulated across all points
    screen_pts_all : list = []   # matching screen positions (one per sample)

    # Shared debug state — updated each sample, displayed by draw()
    _dbg = {"px": None, "py": None}

    # Adaptive EAR threshold for calibration (same logic as BlinkDetector).
    # Collects all EAR values; after 30 frames uses top-third as open-eye baseline.
    _cal_ear_history: list = []

    def _cal_ear_close() -> float:
        """Return the current blink threshold. Returns 0.0 (never discard) until baseline ready."""
        if len(_cal_ear_history) < 30:
            return 0.0
        s = sorted(_cal_ear_history, reverse=True)
        n_open = max(1, len(s) // 3)
        baseline = sum(s[:n_open]) / n_open
        return baseline * 0.55

    # ── Draw the full grid in a given state ────────────────────────────────
    def draw(current_idx: int,
             state: str,
             progress: float = 0.0,
             quality_msg: str = "") -> None:
        now = time.time()
        screen.fill(BG)

        for i, (rect, label) in enumerate(zip(cells, GRID_ITEMS)):
            is_done    = i in done_set
            is_current = i == current_idx

            if is_done:
                bg, border = CELL_DONE, BORDER_DONE
            elif is_current and state == "warn":
                bg, border = CELL_WARN, BORDER_WARN
            elif is_current:
                bg, border = CELL_ACTIVE, BORDER_ACT
            else:
                bg, border = CELL_IDLE, BORDER_IDLE

            pygame.draw.rect(screen, bg,     rect, border_radius=18)
            pygame.draw.rect(screen, border, rect, 2, border_radius=18)

            # Word label
            text_col = TEXT_BRIGHT if (is_done or is_current) else TEXT_DIM
            surf = font_label.render(label, True, text_col)
            screen.blit(surf, (rect.centerx - surf.get_width()  // 2,
                                rect.centery - surf.get_height() // 2 - 8))

            # Completed cell: small "Done" tag at bottom
            if is_done:
                tag = font_status.render("Done", True, GREEN_TAG)
                screen.blit(tag, (rect.centerx - tag.get_width() // 2,
                                  rect.bottom - 26))

            # Active cell waiting: pulsing red dot
            if is_current and state == "waiting":
                pulse = 0.72 + 0.28 * math.sin(now * 5)
                r_outer = int(26 * pulse)
                r_inner = int(14 * pulse)
                pygame.draw.circle(screen, (180, 30, 30), rect.center, r_outer + 5)
                pygame.draw.circle(screen, RED_DOT, rect.center, r_outer)
                pygame.draw.circle(screen, (255, 200, 200), rect.center, r_inner)

            # Active cell collecting: small red dot + cyan arc + %
            if is_current and state == "collecting":
                pygame.draw.circle(screen, RED_DOT, rect.center, 10)
                if progress > 0:
                    arc_rect = pygame.Rect(rect.centerx - 50,
                                          rect.centery - 50, 100, 100)
                    end_angle = -math.pi / 2 + progress * 2 * math.pi
                    pygame.draw.arc(screen, CYAN_ARC, arc_rect,
                                    -math.pi / 2, end_angle, 7)
                pct = font_status.render(f"{int(progress * 100)}%", True, CYAN_ARC)
                screen.blit(pct, (rect.centerx - pct.get_width() // 2,
                                  rect.bottom - 26))

            # Warn state: yellow dot
            if is_current and state == "warn":
                pygame.draw.circle(screen, (255, 180, 0), rect.center, 14)

        # ── Header row ────────────────────────────────────────────────────
        hdr = font_sub.render(
            f"Calibration  —  Point {current_idx + 1} of 9",
            True, (150, 155, 185))
        screen.blit(hdr, (SW // 2 - hdr.get_width() // 2, 18))

        # Progress dots (red=current, green=done, dim=pending)
        for i in range(9):
            cx = SW // 2 - 4 * 18 + i * 18
            if i in done_set:
                col = (0, 200, 80)
            elif i == current_idx:
                col = RED_DOT
            else:
                col = (45, 50, 70)
            pygame.draw.circle(screen, col, (cx, 52), 5)

        # ── Bottom instruction bar ─────────────────────────────────────────
        if state == "waiting":
            line1 = font_sub.render(
                "Patient: keep head still and look at the red dot",
                True, (0, 215, 125))
            line2 = font_sub.render(
                "Caregiver: press  SPACE  when gaze is steady",
                True, (110, 170, 255))
        elif state == "collecting":
            line1 = font_sub.render(
                "Collecting...  keep looking at the red dot  —  do not move",
                True, CYAN_ARC)
            line2 = font_status.render(
                "Almost there — hold still!", True, (140, 145, 165))
        elif state == "warn":
            line1 = font_sub.render(
                quality_msg, True, (255, 175, 0))
            line2 = font_sub.render(
                "Press  R  to retry this point  |  SPACE  to accept anyway",
                True, (150, 150, 160))
        else:
            line1 = font_sub.render("", True, TEXT_BRIGHT)
            line2 = font_sub.render("", True, TEXT_BRIGHT)

        screen.blit(line1, (SW // 2 - line1.get_width() // 2, SH - 58))
        screen.blit(line2, (SW // 2 - line2.get_width() // 2, SH - 30))

        esc = font_status.render("ESC to cancel", True, (45, 50, 68))
        screen.blit(esc, (SW - esc.get_width() - 16, SH - 22))

        # Live pupil coordinate debug (top-left corner)
        if _dbg["px"] is not None:
            dbg_txt = (f"pupil_x: {_dbg['px']:.3f}   "
                       f"pupil_y: {_dbg['py']:.3f}")
            dbg_s = font_status.render(dbg_txt, True, (0, 255, 180))
            overlay = pygame.Surface((dbg_s.get_width() + 16, dbg_s.get_height() + 10),
                                     pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (10, 10))
            screen.blit(dbg_s, (18, 15))

        pygame.display.flip()

    # ── Main calibration loop ──────────────────────────────────────────────
    try:
        for idx, (sx, sy) in enumerate(cal_points):

            while True:   # retry loop for this point

                # ── Phase 1: show grid + red dot, wait for SPACE ──────────
                waiting = True
                while waiting:
                    for ev in pygame.event.get():
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_ESCAPE:
                                raise KeyboardInterrupt("Calibration cancelled")
                            if ev.key == pygame.K_SPACE:
                                waiting = False

                    # Keep debug overlay live while waiting
                    ret, frame = cap.read()
                    if ret:
                        if flip:
                            frame = cv2.flip(frame, 1)
                        w_result = detector.detect(frame)
                        if gaze_model is not None and transform is not None:
                            import torch
                            eye_img = w_result["right_eye"] or w_result["left_eye"]
                            if eye_img is not None:
                                with torch.no_grad():
                                    tensor = transform(eye_img).unsqueeze(0).to(device)
                                    pred, _ = gaze_model.forward_with_features(tensor)
                                    pred = pred.squeeze()
                                _dbg["px"] = float(pred[0].cpu())
                                _dbg["py"] = float(pred[1].cpu())
                        else:
                            iris_gaze = w_result["iris_gaze"]
                            if iris_gaze is not None:
                                _dbg["px"] = iris_gaze[0]
                                _dbg["py"] = iris_gaze[1]

                    draw(idx, "waiting")
                    clock.tick(30)

                # ── Phase 2: collect samples ──────────────────────────────
                samples: list = []
                feat_vecs_point: list = []   # backbone features for this point (reset on retry)
                while len(samples) < n_samples_per_point:
                    for ev in pygame.event.get():
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt("Calibration cancelled")

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    if flip:
                        frame = cv2.flip(frame, 1)

                    result = detector.detect(frame)

                    # Skip frames where the eye is closing/blinking — corrupted
                    # samples bias the calibration mean (especially at top cells).
                    # Landmark indices: 33=outer corner, 133=inner corner,
                    #                   159=upper lid centre, 145=lower lid centre
                    lm = result.get("landmarks")
                    if lm and len(lm) > 159:
                        try:
                            horiz = abs(lm[133].x - lm[33].x)
                            ear   = abs(lm[159].y - lm[145].y) / max(horiz, 1e-5)
                            # Feed into rolling history for adaptive threshold
                            _cal_ear_history.append(ear)
                            if len(_cal_ear_history) > 90:
                                _cal_ear_history[:] = _cal_ear_history[-90:]
                            threshold = _cal_ear_close()
                            if threshold > 0 and ear < threshold:
                                draw(idx, "collecting", len(samples) / n_samples_per_point)
                                clock.tick(30)
                                continue
                        except (IndexError, AttributeError):
                            pass

                    if gaze_model is not None and transform is not None:
                        import torch
                        eye_img = result["right_eye"] or result["left_eye"]
                        if eye_img is None:
                            continue
                        with torch.no_grad():
                            tensor = transform(eye_img).unsqueeze(0).to(device)
                            # Single pass — get coords AND backbone features
                            pred, feat = gaze_model.forward_with_features(tensor)
                            pred = pred.squeeze()
                        pupil = (float(pred[0].cpu()), float(pred[1].cpu()))
                        feat_vecs_point.append(feat.squeeze().cpu().numpy())
                    else:
                        iris_gaze = result["iris_gaze"]
                        if iris_gaze is None:
                            continue
                        pupil = tuple(iris_gaze)

                    # Update debug display
                    _dbg["px"] = pupil[0]
                    _dbg["py"] = pupil[1]

                    samples.append(list(pupil))
                    draw(idx, "collecting", len(samples) / n_samples_per_point)
                    clock.tick(30)

                # ── Phase 3: quality check ────────────────────────────────
                arr          = np.array(samples)
                std          = np.std(arr, axis=0)
                std_x, std_y = float(std[0]), float(std[1])

                if std_x < 0.05 and std_y < 0.05:
                    break   # quality OK

                # Unstable gaze — let caregiver decide
                q_msg         = (f"Gaze was unstable  "
                                 f"(variance x={std_x:.3f}, y={std_y:.3f})")
                action_chosen = False
                redo          = False
                while not action_chosen:
                    for ev in pygame.event.get():
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_ESCAPE:
                                raise KeyboardInterrupt("Calibration cancelled")
                            if ev.key == pygame.K_r:
                                redo = True
                                action_chosen = True
                            elif ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                                action_chosen = True
                    draw(idx, "warn", quality_msg=q_msg)
                    clock.tick(30)

                if not redo:
                    break   # accepted despite poor quality

            # ── Accept point ──────────────────────────────────────────────
            mean_pupil = np.mean(samples, axis=0)
            pupil_pts.append(tuple(mean_pupil))
            screen_pts.append((sx, sy))
            done_set.add(idx)

            # Accumulate all per-sample feature vectors (not just the mean)
            if feat_vecs_point:
                feat_vecs_all.extend(feat_vecs_point)
                screen_pts_all.extend([(sx, sy)] * len(feat_vecs_point))

            std_final = np.std(np.array(samples), axis=0)
            print(f"  [{idx + 1}/9] screen=({sx},{sy})  "
                  f"pupil=({mean_pupil[0]:.4f},{mean_pupil[1]:.4f})  "
                  f"std=({std_final[0]:.4f},{std_final[1]:.4f})")

            # Brief pause so the green flash is visible before advancing
            draw(idx, "waiting")
            pygame.time.wait(400)

    except KeyboardInterrupt:
        print("[GazeMapper] Calibration cancelled.")
        raise

    finally:
        cap.release()
        pygame.quit()

    mapper = GazeMapper()
    mapper.fit(pupil_pts, screen_pts)
    mapper.save(save_path)

    # Build the personalized mapper if backbone features were collected
    personalized: PersonalizedGazeMapper | None = None
    if feat_vecs_all:
        personalized = PersonalizedGazeMapper()
        personalized.fit(feat_vecs_all, screen_pts_all)
        personalized.save()
        print("[PersonalizedMapper] Personalized calibration complete and saved.")

    # ── Y-offset diagnostic ────────────────────────────────────────────────
    # Predict the screen Y for each calibration point and compare.
    # If the system consistently predicts too low, the patient will need to
    # look above the target. The tip below tells them exactly how to fix it.
    top_errs, bot_errs = [], []
    for i, ((px, py), (tx, ty)) in enumerate(zip(pupil_pts, screen_pts)):
        pred_x, pred_y = mapper.predict(px, py)
        err_y = ty - pred_y   # positive = predicted too low (need shift up)
        if i in (0, 1, 2):      # top row
            top_errs.append(err_y)
        elif i in (6, 7, 8):    # bottom row
            bot_errs.append(err_y)

    if top_errs:
        mean_top_err = sum(top_errs) / len(top_errs)
        if abs(mean_top_err) > 40:
            direction = "UP" if mean_top_err > 0 else "DOWN"
            shift_val = int(abs(mean_top_err))
            flag = f"--shift-y {shift_val}" if mean_top_err > 0 else f"--shift-y -{shift_val}"
            print(f"\n[GazeMapper] Y diagnostic: top-row predictions are off by "
                  f"{mean_top_err:+.0f} px ({direction} shift needed).")
            print(f"[GazeMapper] Recommended:  python main_system.py {flag}\n")
        else:
            print("[GazeMapper] Y diagnostic: top-row alignment looks good.")

    print("[GazeMapper] Calibration complete and saved.")
    return mapper, personalized