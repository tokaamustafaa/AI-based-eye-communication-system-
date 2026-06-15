"""
grid_ui.py — Fullscreen 3×3 gaze-controlled communication grid.

Full pipeline each frame:
    camera → EyeDetector → GazeModel → GazeMapper → cell index → dwell timer → TTS

Dwell-time selection:
    The patient holds their gaze on a cell for DWELL_MS milliseconds.
    A cyan arc around the cell fills up as dwell progresses.
    On completion the cell flashes green and the item is spoken aloud.
    A short cooldown prevents immediate re-triggering.

Controls:
    ESC — exit the grid
"""

import time
import datetime
import json
import os
from collections import deque, Counter
import cv2
from camera_setup import open_camera
import torch
import numpy as np
import pygame

from gaze_mapper import GRID_ITEMS, GazeMapper, PersonalizedGazeMapper
from eye_detector import EyeDetector, BlinkDetector
from dataset      import _eval_transform
from model        import GazeEstimationModel
from utils        import load_checkpoint
from tts_engine        import speak
from pushover_notify   import send_notification


# ---------------------------------------------------------------------------
# Layout & timing constants
# ---------------------------------------------------------------------------

MARGIN_X    = 100      # px from screen edge to grid edge (horizontal)
MARGIN_Y    = 80       # px from screen edge to grid edge (vertical)
DWELL_MS    = 1500     # ms dwell for pure dwell-only mode (--no-blink)
DWELL_PRE_MS       = 800   # stage 1 dwell (blink mode): gaze → pre-select
CONFIRM_TIMEOUT_MS = 2000  # stage 2 (blink mode): ms to blink after pre-select
GRACE_MS           = 400   # grace: resume dwell timer if gaze briefly drifts
COOLDOWN_MS = 1000     # ms lockout after selection (prevents re-trigger)
FPS         = 30       # camera + render loop rate

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

BG_COLOR        = (12,  14,  22)
CELL_IDLE       = (28,  33,  52)
CELL_HOVER      = (45,  58,  100)
CELL_SELECTED   = (18,  155, 75)
BORDER_IDLE     = (55,  65,  95)
BORDER_HOVER    = (90,  110, 180)
TEXT_COLOR      = (215, 220, 238)
DWELL_COLOR      = (0,   200, 255)
PRESELECT_COLOR  = (255, 155,  0)   # orange — cell pre-selected, awaiting blink
CELL_PRESELECT   = (75,  45,   5)   # dark orange background
BORDER_PRESELECT = (255, 155,  0)
GAZE_DOT_COLOR   = (255, 70,  70)
STATUS_COLOR     = (110, 115, 135)

EMA_ALPHA   = 0.20   # gaze EMA: lower = more stable dot, slightly more lag (0.15–0.25)
CELL_VOTE_N = 8      # frames to vote on which cell gaze is on — prevents dwell reset on jitter
LOG_FILE     = "medical_log.json"
URGENT_ITEMS = frozenset({"PAIN", "CALL", "TOILET"})


# ---------------------------------------------------------------------------
# Medical log
# ---------------------------------------------------------------------------

def save_log(item: str) -> None:
    entry = {
        "time":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "item":   item,
        "urgent": item in URGENT_ITEMS,
    }
    logs: list = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"[Log] Saved → {item} at {entry['time']}")


# ---------------------------------------------------------------------------
# GridUI
# ---------------------------------------------------------------------------

class GridUI:
    """
    Fullscreen pygame gaze-communication board.
f
    Args:
        mapper    : fitted GazeMapper
        detector  : EyeDetector instance
        model     : loaded GazeEstimationModel (eval mode)
        transform : _eval_transform() from dataset.py
        device    : torch.device
        dwell_ms  : override dwell threshold in milliseconds
    """

    def __init__(self,
                 mapper:               GazeMapper,
                 detector:             EyeDetector,
                 model:                GazeEstimationModel,
                 transform,
                 device:               torch.device,
                 dwell_ms:             int  = DWELL_MS,
                 debug:                bool = False,
                 flip:                 bool = False,
                 personalized_mapper:  "PersonalizedGazeMapper | None" = None,
                 blink_mode:           bool = True,
                 shift_y:              int  = 0):

        self.mapper              = mapper
        self.personalized_mapper = personalized_mapper
        self.detector            = detector
        self.model               = model
        self.transform           = transform
        self.device              = device
        self.dwell_ms            = dwell_ms
        self.debug               = debug
        self.flip                = flip
        self.blink_mode          = blink_mode
        self.shift_y             = shift_y   # pixels to shift gaze UP (fix row-below confusion)

        if personalized_mapper is not None and personalized_mapper.is_calibrated():
            print("[GridUI] Using personalized gaze mapper (feature-space adaptation).")
        else:
            print("[GridUI] Using polynomial gaze mapper (no personalized calibration found).")
        print(f"[GridUI] Selection mode: {'blink-confirm' if blink_mode else 'dwell-only'}")

        # Tell Windows we handle DPI ourselves so pygame gets true pixel counts
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

        import os
        os.environ.setdefault("SDL_VIDEO_FULLSCREEN_DISPLAY", "0")  # primary monitor

        pygame.init()
        # (0, 0) lets pygame pick the native resolution — avoids DPI-scaling offset
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.SW, self.SH = self.screen.get_size()
        pygame.display.set_caption("Gaze Communication Board")
        self.clock = pygame.time.Clock()

        self.font_item   = pygame.font.SysFont("Arial", 68, bold=True)
        self.font_status = pygame.font.SysFont("Arial", 22)
        self.font_dwell  = pygame.font.SysFont("Arial", 18)

        # Pre-compute cell rectangles once
        self._cells: list[pygame.Rect] = self._build_cells()

        # Runtime state — dwell tracking
        self._hovered:         int | None   = None   # cell currently being tracked
        self._dwell_start:     float | None = None   # when tracking of _hovered began
        self._last_on_hovered: float        = 0.0    # last frame gaze was ON _hovered
        self._last_selected:   int | None   = None
        self._flash_frames:    int          = 0
        self._cooldown_until:  float        = 0.0

        # Blink-confirm stage 2
        self._preselect_cell:  int | None   = None   # cell pending blink confirmation
        self._preselect_start: float | None = None

        # Overall selection state for _draw()
        self._sel_state: str = "idle"   # idle / dwelling / preselected / cooldown

        # Blink detector
        self._blink_detector: BlinkDetector | None = (
            BlinkDetector() if blink_mode else None
        )
        self._last_landmarks  = None
        self._last_frame_hw   = (480, 640)
        self._last_blink_event: str = "open"

        # Board sleep/wake — long blink (2 s) toggles between active and paused
        self._board_active: bool = True
        self.patient_name:  str  = ""   # set by main_system.py after init

        # Gaze smoothing: EMA for screen position, vote buffer for cell stability
        self._ema_sx: float | None = None
        self._ema_sy: float | None = None
        self._cell_vote_buf: deque = deque(maxlen=CELL_VOTE_N)

        # Last raw pupil prediction — shown in debug overlay
        self._dbg_pupil_x: float | None = None
        self._dbg_pupil_y: float | None = None

        try:
            self._cap = open_camera()
        except RuntimeError as exc:
            self._show_error(str(exc))
            pygame.quit()
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cells(self) -> list[pygame.Rect]:
        gw = (self.SW - 2 * MARGIN_X) / 3
        gh = (self.SH - 2 * MARGIN_Y) / 3
        rects = []
        for row in range(3):
            for col in range(3):
                x = int(MARGIN_X + col * gw)
                y = int(MARGIN_Y + row * gh)
                rects.append(pygame.Rect(x, y, int(gw), int(gh)))
        return rects

    def _show_error(self, message: str, duration_ms: int = 3000) -> None:
        self.screen.fill((25, 10, 10))
        font_err  = pygame.font.SysFont("Arial", 32, bold=True)
        font_hint = pygame.font.SysFont("Arial", 22)
        lines = [
            font_err.render("System Error",  True, (255, 80,  80)),
            font_err.render(message,         True, (210, 180, 180)),
            font_hint.render("Please check your camera and restart.", True, (150, 150, 150)),
        ]
        y = self.SH // 2 - sum(s.get_height() + 10 for s in lines) // 2
        for surf in lines:
            self.screen.blit(surf, (self.SW // 2 - surf.get_width() // 2, y))
            y += surf.get_height() + 10
        pygame.display.flip()
        pygame.time.wait(duration_ms)

    def _get_gaze_cell(self, frame: np.ndarray) -> tuple[int | None, float | None, float | None]:
        """
        Run full pipeline on one BGR frame.
        Returns (cell_index, screen_x, screen_y) — all None on failure.
        Uses the EfficientNet model for pupil prediction; falls back to iris_gaze
        if no eye crop is available.
        """
        if self.flip:
            frame = cv2.flip(frame, 1)

        result  = self.detector.detect(frame)
        self._last_landmarks = result.get("landmarks")
        self._last_frame_hw  = (frame.shape[0], frame.shape[1])

        # Preselect anchor: when a cell is orange ("BLINK NOW"), the user's only job
        # is to blink — any iris drift (especially for top-row cells when looking up)
        # must not move the dot or change which cell gets confirmed.
        # We still run detect() above so landmarks are fresh for blink detection.
        if self._preselect_cell is not None:
            locked = self._cells[self._preselect_cell]
            return self._preselect_cell, float(locked.centerx), float(locked.centery)

        # Freeze gaze position while the eye is physically closing.
        # Without this, MediaPipe reports a wrong iris position during a blink
        # which jumps the gaze dot to a different cell and breaks dwell tracking.
        if (self._blink_detector is not None
                and self._blink_detector.is_eye_closing(self._last_landmarks)
                and self._ema_sx is not None):
            sx_s = self._ema_sx
            sy_s = self._ema_sy - self.shift_y
            cell = self.mapper.screen_to_grid_cell(
                sx_s, sy_s, self.SW, self.SH, MARGIN_X, MARGIN_Y)
            return cell, sx_s, sy_s   # frozen — do not update EMA

        eye_img = result["right_eye"] or result["left_eye"]

        if eye_img is None:
            self._ema_sx = None
            self._ema_sy = None
            self._dbg_pupil_x = None
            self._dbg_pupil_y = None
            return None, None, None

        with torch.no_grad():
            tensor = self.transform(eye_img).unsqueeze(0).to(self.device)
            # Single forward pass — get both coords and backbone features
            pred, feat = self.model.forward_with_features(tensor)
            pred   = pred.squeeze()
            pupil_x = float(pred[0].cpu())
            pupil_y = float(pred[1].cpu())

        self._dbg_pupil_x = pupil_x
        self._dbg_pupil_y = pupil_y

        # Use personalized mapper (feature-space) if available — more accurate
        if (self.personalized_mapper is not None
                and self.personalized_mapper.is_calibrated()):
            feat_vec = feat.squeeze().cpu().numpy()
            sx, sy   = self.personalized_mapper.predict(feat_vec)
        elif self.mapper.is_calibrated():
            sx, sy = self.mapper.predict(pupil_x, pupil_y)
        else:
            return None, None, None

        # EMA: low-pass filter on screen coordinates — suppresses model noise
        # while still tracking real eye movements in ~5 frames (167 ms at 30 fps)
        if self._ema_sx is None:
            self._ema_sx = sx
            self._ema_sy = sy
        else:
            self._ema_sx = EMA_ALPHA * sx + (1.0 - EMA_ALPHA) * self._ema_sx
            self._ema_sy = EMA_ALPHA * sy + (1.0 - EMA_ALPHA) * self._ema_sy

        sx_s = self._ema_sx
        sy_s = self._ema_sy - self.shift_y

        cell = self.mapper.screen_to_grid_cell(sx_s, sy_s, self.SW, self.SH,
                                                MARGIN_X, MARGIN_Y)
        return cell, sx_s, sy_s

    def _draw(self,
              cell_idx: int | None,
              sx: float | None,
              sy: float | None) -> None:
        now = time.time()
        self.screen.fill(BG_COLOR)

        for i, (rect, label) in enumerate(zip(self._cells, GRID_ITEMS)):

            # ── cell background & border ──────────────────────────────────
            if i == self._last_selected and self._flash_frames > 0:
                bg, border = CELL_SELECTED,   BORDER_HOVER
            elif i == self._preselect_cell and self._sel_state == "preselected":
                bg, border = CELL_PRESELECT,  BORDER_PRESELECT
            elif i == self._hovered and self._sel_state == "dwelling":
                bg, border = CELL_HOVER,      BORDER_HOVER
            else:
                bg, border = CELL_IDLE,       BORDER_IDLE

            pygame.draw.rect(self.screen, bg,     rect, border_radius=20)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=20)

            # ── label ────────────────────────────────────────────────────
            surf = self.font_item.render(label, True, TEXT_COLOR)
            self.screen.blit(
                surf,
                (rect.centerx - surf.get_width()  // 2,
                 rect.centery - surf.get_height() // 2),
            )

            arc_r = pygame.Rect(rect.centerx - 52, rect.centery - 52, 104, 104)

            # ── Stage 1 arc: cyan filling up (dwell progress) ─────────────
            if i == self._hovered and self._sel_state == "dwelling" \
                    and self._dwell_start is not None:
                stage_ms = DWELL_PRE_MS if self.blink_mode else self.dwell_ms
                progress = min((now - self._dwell_start) * 1000 / stage_ms, 1.0)
                pygame.draw.arc(
                    self.screen, DWELL_COLOR, arc_r,
                    -1.5708, -1.5708 + progress * 6.2832, 7,
                )
                pct = self.font_dwell.render(f"{int(progress * 100)}%", True, DWELL_COLOR)
                self.screen.blit(pct,
                    (rect.centerx - pct.get_width() // 2,
                     rect.bottom  - pct.get_height() - 8))

            # ── Stage 2 arc: orange draining (confirm countdown) ──────────
            elif i == self._preselect_cell and self._sel_state == "preselected" \
                    and self._preselect_start is not None:
                elapsed   = (now - self._preselect_start) * 1000
                remaining = max(0.0, 1.0 - elapsed / CONFIRM_TIMEOUT_MS)
                if remaining > 0:
                    pygame.draw.arc(
                        self.screen, PRESELECT_COLOR, arc_r,
                        -1.5708, -1.5708 + remaining * 6.2832, 7,
                    )
                # Pulsing "BLINK" label
                pulse     = 0.6 + 0.4 * abs((now * 3) % 2 - 1)
                blink_col = (int(255 * pulse), int(155 * pulse), 0)
                blink_s   = self.font_dwell.render("BLINK NOW!", True, blink_col)
                self.screen.blit(blink_s,
                    (rect.centerx - blink_s.get_width() // 2,
                     rect.bottom  - blink_s.get_height() - 8))

        # ── gaze dot ─────────────────────────────────────────────────────
        if sx is not None and sy is not None:
            gx = int(max(0, min(sx, self.SW - 1)))
            gy = int(max(0, min(sy, self.SH - 1)))
            pygame.draw.circle(self.screen, GAZE_DOT_COLOR, (gx, gy), 9)
            pygame.draw.circle(self.screen, (255, 255, 255), (gx, gy), 9, 2)

        # ── status bar ───────────────────────────────────────────────────
        if self._sel_state == "preselected" and self._preselect_cell is not None:
            secs_left  = max(0.0, CONFIRM_TIMEOUT_MS / 1000
                             - (now - self._preselect_start))
            status     = (f"BLINK to select  {GRID_ITEMS[self._preselect_cell]}"
                          f"  ({secs_left:.1f}s)")
            status_col = PRESELECT_COLOR
        elif self._sel_state == "dwelling" and self._hovered is not None:
            stage_ms   = DWELL_PRE_MS if self.blink_mode else self.dwell_ms
            pct        = int(min((now - self._dwell_start) * 1000 / stage_ms, 1.0) * 100)
            status     = f"Looking at: {GRID_ITEMS[self._hovered]}  ({pct}%)"
            status_col = STATUS_COLOR
        elif cell_idx is not None:
            status     = f"Gaze on: {GRID_ITEMS[cell_idx]}"
            status_col = STATUS_COLOR
        else:
            status     = "No eye detected — look directly at camera"
            status_col = STATUS_COLOR

        s = self.font_status.render(status, True, status_col)
        self.screen.blit(s, (20, self.SH - 34))

        # Blink-mode indicator top-right (turns red when eye is closing)
        if self.blink_mode:
            eye_col  = (255, 80, 80) if self._last_blink_event == "closing" \
                       else (0, 200, 120)
            mode_txt = self.font_status.render("BLINK MODE", True, eye_col)
        else:
            mode_txt = self.font_status.render("DWELL MODE", True, (80, 90, 120))
        self.screen.blit(mode_txt, (self.SW - mode_txt.get_width() - 20, 14))

        esc = self.font_status.render("ESC to exit", True, (60, 65, 85))
        self.screen.blit(esc, (self.SW - esc.get_width() - 20, self.SH - 34))

        # ── debug overlay (--debug flag) ──────────────────────────────────
        if self.debug:
            px_str = f"{self._dbg_pupil_x:.3f}" if self._dbg_pupil_x is not None else "---"
            py_str = f"{self._dbg_pupil_y:.3f}" if self._dbg_pupil_y is not None else "---"
            sx_str = f"{sx:.0f}" if sx is not None else "---"
            sy_str = f"{sy:.0f}" if sy is not None else "---"
            debug_lines = [
                f"pupil_x: {px_str}   pupil_y: {py_str}",
                f"screen_x: {sx_str}   screen_y: {sy_str}",
                f"cell: {cell_idx}   state: {self._sel_state}",
                f"blink: {self._last_blink_event}",
            ]
            box_x, box_y = 14, 14
            pad = 8
            line_h = self.font_status.get_height() + 4
            box_h = pad * 2 + line_h * len(debug_lines)
            box_w = 340
            overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (box_x, box_y))
            for i, line in enumerate(debug_lines):
                surf = self.font_status.render(line, True, (0, 255, 180))
                self.screen.blit(surf, (box_x + pad, box_y + pad + i * line_h))

        # ── flash countdown ───────────────────────────────────────────────
        if self._flash_frames > 0:
            self._flash_frames -= 1

        pygame.display.flip()

    def _draw_sleep(self) -> None:
        """Shown when the board is paused via long blink."""
        self.screen.fill((6, 8, 14))
        now = time.time()
        pulse = 0.55 + 0.45 * abs((now * 0.6) % 2 - 1)
        eye_col = (int(0 * pulse), int(180 * pulse), int(220 * pulse))

        lines = [
            self.font_item.render("👁", True, eye_col),
            self.font_status.render(
                f"Session paused — {self.patient_name}" if self.patient_name
                else "Session paused",
                True, (80, 90, 110)),
            self.font_status.render(
                "Hold eyes closed 2 seconds to resume",
                True, (55, 65, 85)),
        ]
        total_h = sum(s.get_height() + 14 for s in lines)
        y = (self.SH - total_h) // 2
        for surf in lines:
            self.screen.blit(surf, (self.SW // 2 - surf.get_width() // 2, y))
            y += surf.get_height() + 14
        pygame.display.flip()

    def _on_select(self, cell_idx: int) -> None:
        """Trigger TTS, log the event, notify caregiver, and reset all selection state."""
        item = GRID_ITEMS[cell_idx]
        print(f"[GridUI] Selected → {item}")
        speak(item)
        save_log(item)
        send_notification(item)

        self._last_selected   = cell_idx
        self._flash_frames    = int(FPS * 0.6)
        self._cooldown_until  = time.time() + COOLDOWN_MS / 1000
        self._hovered         = None
        self._dwell_start     = None
        self._last_on_hovered = 0.0
        self._preselect_cell  = None
        self._preselect_start = None
        self._sel_state       = "cooldown"

    def _update_selection(self, cell_idx: int | None, now: float, blink_event: str) -> None:
        """
        State machine — call once per frame.

        States:
            idle        – no gaze tracked
            dwelling    – gaze on a cell, stage-1 timer running
                          (grace period: timer pauses but does not reset if gaze
                           briefly leaves and returns within GRACE_MS)
            preselected – stage-1 complete; waiting for a deliberate blink
            cooldown    – post-selection lockout
        """
        if now < self._cooldown_until:
            self._sel_state = "cooldown"
            return

        # ── Stage 2: pre-selected, waiting for blink ─────────────────────
        if self._preselect_cell is not None:
            elapsed_ms = (now - self._preselect_start) * 1000

            if blink_event == "confirm_blink":
                self._on_select(self._preselect_cell)
                return

            # Still on the pre-selected cell (or within grace)?
            on_cell    = (cell_idx == self._preselect_cell)
            grace_ok   = (now - self._last_on_hovered) < GRACE_MS / 1000
            timed_out  = elapsed_ms > CONFIRM_TIMEOUT_MS

            if on_cell:
                self._last_on_hovered = now

            if timed_out or (not on_cell and not grace_ok):
                # Cancel — patient looked away or ran out of time
                self._preselect_cell  = None
                self._preselect_start = None
                self._hovered         = None
                self._dwell_start     = None
                self._last_on_hovered = 0.0
                self._sel_state       = "idle"
                return

            self._sel_state = "preselected"
            return

        # ── Stage 1: dwell tracking ───────────────────────────────────────
        if cell_idx is not None and cell_idx == self._hovered:
            # Gaze is on the tracked cell
            self._last_on_hovered = now
            stage_ms   = DWELL_PRE_MS if self.blink_mode else self.dwell_ms
            elapsed_ms = (now - self._dwell_start) * 1000

            if elapsed_ms >= stage_ms:
                if self.blink_mode:
                    # Promote to pre-selected
                    self._preselect_cell  = cell_idx
                    self._preselect_start = now
                    self._sel_state       = "preselected"
                else:
                    self._on_select(cell_idx)
            else:
                self._sel_state = "dwelling"

        elif (self._hovered is not None
              and (now - self._last_on_hovered) < GRACE_MS / 1000):
            # Gaze briefly off tracked cell — grace period, keep timer running
            self._sel_state = "dwelling"

        else:
            # Start tracking a new cell (or go idle)
            if cell_idx is not None:
                self._hovered         = cell_idx
                self._dwell_start     = now
                self._last_on_hovered = now
                self._sel_state       = "dwelling"
            else:
                self._hovered         = None
                self._dwell_start     = None
                self._last_on_hovered = 0.0
                self._sel_state       = "idle"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Start the main loop. Blocks until ESC or R is pressed.
        Returns True if caregiver pressed R to request recalibration.
        """
        try:
            while True:
                # ── event handling ────────────────────────────────────
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False
                        if event.key == pygame.K_r:
                            print("[GridUI] R pressed — restarting calibration.")
                            return True

                # ── grab frame & run pipeline ─────────────────────────
                ret, frame = self._cap.read()
                if not ret:
                    continue

                now = time.time()

                # Always run face detection so blink detector has fresh landmarks
                # (needed in both active and sleep modes to detect the wake gesture)
                if not self._board_active:
                    result = self.detector.detect(
                        cv2.flip(frame, 1) if self.flip else frame)
                    self._last_landmarks = result.get("landmarks")
                    self._last_frame_hw  = (frame.shape[0], frame.shape[1])
                    if self._blink_detector is not None:
                        self._last_blink_event = self._blink_detector.update(
                            self._last_landmarks,
                            self._last_frame_hw[0], self._last_frame_hw[1])
                        if self._last_blink_event == "long_blink":
                            self._board_active = True
                            speak("Board on.")
                    self._draw_sleep()
                    self.clock.tick(FPS)
                    continue

                cell_idx, sx, sy = self._get_gaze_cell(frame)

                # Blink detection (uses landmarks saved inside _get_gaze_cell)
                if self._blink_detector is not None:
                    self._last_blink_event = self._blink_detector.update(
                        self._last_landmarks,
                        self._last_frame_hw[0],
                        self._last_frame_hw[1],
                    )
                else:
                    self._last_blink_event = "open"

                # Long blink → sleep
                if self._last_blink_event == "long_blink":
                    self._board_active    = False
                    self._hovered         = None
                    self._dwell_start     = None
                    self._preselect_cell  = None
                    self._preselect_start = None
                    self._sel_state       = "idle"
                    self._cell_vote_buf.clear()
                    speak("Board off.")
                    self.clock.tick(FPS)
                    continue

                # Cell vote: require a majority over the last CELL_VOTE_N frames before
                # switching the active cell. A single jittered frame cannot reset the
                # dwell timer or flip the pre-selected cell.
                # During a blink, the iris position is unreliable — record None so
                # blink frames don't contaminate the vote and flip the tracked cell.
                eye_is_closing = (
                    self._blink_detector is not None
                    and self._blink_detector.is_eye_closing(self._last_landmarks)
                )
                self._cell_vote_buf.append(None if eye_is_closing else cell_idx)
                counts = Counter(c for c in self._cell_vote_buf if c is not None)
                if counts:
                    top_cell, top_count = counts.most_common(1)[0]
                    stable_cell = top_cell if top_count > len(self._cell_vote_buf) // 2 else None
                else:
                    stable_cell = None

                self._update_selection(stable_cell, now, self._last_blink_event)
                self._draw(stable_cell, sx, sy)
                self.clock.tick(FPS)

        finally:
            self._cap.release()


# ---------------------------------------------------------------------------
# Convenience launcher (used by main_system.py)
# ---------------------------------------------------------------------------

def launch_grid(checkpoint_path: str = "checkpoints/best_model_v2.pth",
                dwell_ms: int = DWELL_MS) -> None:
    """
    Load everything from disk and launch the grid.
    The GazeMapper must already be calibrated (saved to checkpoints/).
    """
    device    = torch.device("cpu")
    model     = GazeEstimationModel(pretrained=False)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    transform = _eval_transform()
    detector  = EyeDetector(padding=0.30)
    mapper    = GazeMapper()

    if not mapper.load():
        print("[GridUI] No calibration found. Run calibration first.")
        return

    ui = GridUI(mapper, detector, model, transform, device, dwell_ms=dwell_ms)
    ui.run()
    detector.release()
