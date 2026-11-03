"""
Finger Controlled Mouse (Air Mouse)
===================================
A high-performance, touchless computer mouse control system powered by OpenCV,
MediaPipe, and PyAutoGUI. Controls cursor movement, clicks, dragging, and scrolling
using real-time hand gestures captured via your webcam.

Features:
- Full-screen applicable & resizable OpenCV display with responsive HUD
- Configurable gesture thresholds and Exponential Moving Average (EMA) cursor smoothing
- Visual interaction zone mapping camera coordinates to full screen bounds
- Real-time gesture classification (Move, Left Click, Right Click, Drag, Scroll, Stop)
- Modern Head-Up Display (HUD) with translucent status badges and metrics
- Interactive hotkeys for Fullscreen toggle, HUD toggle, and live sensitivity tuning
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Safety: Disable PyAutoGUI failsafe to prevent edge-of-screen crashes while gesturing
pyautogui.FAILSAFE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AirMouse")


class GestureType(Enum):
    """Enumeration of recognized hand gestures."""
    NONE = "NO HAND DETECTED"
    READY = "READY"
    MOVE = "MOVE"
    LEFT_CLICK = "LEFT CLICK"
    RIGHT_CLICK = "RIGHT CLICK"
    DRAG = "DRAG & DROP"
    SCROLL = "SCROLL MODE"
    STOP = "STOP"


@dataclass
class MouseConfig:
    """Runtime configuration parameters for Finger Controlled Mouse."""
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    display_width: int = 1280
    display_height: int = 720
    fullscreen: bool = False
    smoothening: float = 5.0
    click_threshold: float = 55.0
    click_cooldown: float = 0.18
    scroll_threshold: float = 16.0
    margin_x: int = 90
    margin_y: int = 70
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.60
    show_hud: bool = True
    show_mapping_zone: bool = True


class HandLandmarkIndex:
    """MediaPipe hand landmark landmark indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class HandDetector:
    """Encapsulates MediaPipe hand detection and landmark extraction."""

    def __init__(self, config: MouseConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame: np.ndarray):
        """Processes RGB frame and returns MediaPipe detection results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    @staticmethod
    def calculate_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """Euclidean distance between two 2D points."""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def get_pixel_positions(self, landmarks, frame_w: int, frame_h: int) -> Dict[str, Tuple[int, int]]:
        """Maps normalized landmarks to pixel coordinates for fingertips."""
        return {
            "thumb": (
                int(landmarks[HandLandmarkIndex.THUMB_TIP].x * frame_w),
                int(landmarks[HandLandmarkIndex.THUMB_TIP].y * frame_h),
            ),
            "index": (
                int(landmarks[HandLandmarkIndex.INDEX_TIP].x * frame_w),
                int(landmarks[HandLandmarkIndex.INDEX_TIP].y * frame_h),
            ),
            "middle": (
                int(landmarks[HandLandmarkIndex.MIDDLE_TIP].x * frame_w),
                int(landmarks[HandLandmarkIndex.MIDDLE_TIP].y * frame_h),
            ),
            "ring": (
                int(landmarks[HandLandmarkIndex.RING_TIP].x * frame_w),
                int(landmarks[HandLandmarkIndex.RING_TIP].y * frame_h),
            ),
            "pinky": (
                int(landmarks[HandLandmarkIndex.PINKY_TIP].x * frame_w),
                int(landmarks[HandLandmarkIndex.PINKY_TIP].y * frame_h),
            ),
            "wrist": (
                int(landmarks[HandLandmarkIndex.WRIST].x * frame_w),
                int(landmarks[HandLandmarkIndex.WRIST].y * frame_h),
            ),
        }

    def evaluate_fingers_up(self, landmarks, handedness_label: str = "Right") -> List[int]:
        """
        Determines which fingers are extended upward (1) or curled down (0).
        Fingers list order: [Thumb, Index, Middle, Ring, Pinky]
        """
        fingers = []

        # Thumb: In mirrored view, compare x coordinates based on handedness
        # For mirrored right hand: Thumb tip x > Thumb IP x when open
        thumb_tip = landmarks[HandLandmarkIndex.THUMB_TIP]
        thumb_ip = landmarks[HandLandmarkIndex.THUMB_IP]
        if handedness_label == "Right":
            fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)
        else:
            fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)

        # Other 4 fingers: Compare Y coordinate of tip vs PIP joint
        finger_pairs = [
            (HandLandmarkIndex.INDEX_TIP, HandLandmarkIndex.INDEX_PIP),
            (HandLandmarkIndex.MIDDLE_TIP, HandLandmarkIndex.MIDDLE_PIP),
            (HandLandmarkIndex.RING_TIP, HandLandmarkIndex.RING_PIP),
            (HandLandmarkIndex.PINKY_TIP, HandLandmarkIndex.PINKY_PIP),
        ]

        for tip_idx, pip_idx in finger_pairs:
            fingers.append(1 if landmarks[tip_idx].y < landmarks[pip_idx].y else 0)

        return fingers

    def draw_hand_skeleton(self, frame: np.ndarray, hand_landmarks) -> None:
        """Renders styled MediaPipe hand connections and joints."""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
            self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


class MouseController:
    """Translates hand gestures and fingertip coordinates into OS mouse actions."""

    def __init__(self, config: MouseConfig):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_x, self.prev_y = self.screen_width // 2, self.screen_height // 2
        self.curr_x, self.curr_y = self.prev_x, self.prev_y

        # State tracking
        self.left_click_state = False
        self.right_click_state = False
        self.dragging = False
        self.scroll_mode = False
        self.scroll_start_y = 0.0

        # Timing
        self.last_left_click_time = 0.0
        self.last_right_click_time = 0.0

    def move_cursor(self, raw_x: int, raw_y: int, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """
        Maps fingertip coordinates from camera bounding box to screen coordinates
        using Exponential Moving Average (EMA) smoothing for jitter reduction.
        """
        # Interpolate within active margin zone to reach screen edges comfortably
        margin_x = self.config.margin_x
        margin_y = self.config.margin_y

        target_x = np.interp(raw_x, [margin_x, frame_w - margin_x], [0, self.screen_width])
        target_y = np.interp(raw_y, [margin_y, frame_h - margin_y], [0, self.screen_height])

        # Clamp to valid screen boundaries
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        # Apply smoothing factor
        smooth = max(1.0, float(self.config.smoothening))
        self.curr_x = self.prev_x + (target_x - self.prev_x) / smooth
        self.curr_y = self.prev_y + (target_y - self.prev_y) / smooth

        pyautogui.moveTo(int(self.curr_x), int(self.curr_y))
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        return int(self.curr_x), int(self.curr_y)

    def trigger_left_click(self) -> bool:
        """Executes a left mouse click if cooldown has elapsed."""
        current_time = time.time()
        if not self.left_click_state and (current_time - self.last_left_click_time) > self.config.click_cooldown:
            pyautogui.click()
            self.left_click_state = True
            self.last_left_click_time = current_time
            logger.info("Gesture Action: Left Click")
            return True
        return False

    def trigger_right_click(self) -> bool:
        """Executes a right mouse click if cooldown has elapsed."""
        current_time = time.time()
        if not self.right_click_state and (current_time - self.last_right_click_time) > self.config.click_cooldown:
            pyautogui.rightClick()
            self.right_click_state = True
            self.last_right_click_time = current_time
            logger.info("Gesture Action: Right Click")
            return True
        return False

    def set_drag_state(self, active: bool) -> None:
        """Begins or ends a drag (mouse down / mouse up) operation."""
        if active and not self.dragging:
            pyautogui.mouseDown()
            self.dragging = True
            logger.info("Gesture Action: Drag Started")
        elif not active and self.dragging:
            pyautogui.mouseUp()
            self.dragging = False
            logger.info("Gesture Action: Drag Released")

    def handle_scroll(self, current_y: float) -> Optional[str]:
        """Scrolls up or down based on vertical displacement of the hand."""
        if not self.scroll_mode:
            self.scroll_start_y = current_y
            self.scroll_mode = True
            return None

        diff = self.scroll_start_y - current_y
        if abs(diff) > self.config.scroll_threshold:
            if diff > 0:
                pyautogui.scroll(4)
                self.scroll_start_y = current_y
                return "UP"
            else:
                pyautogui.scroll(-4)
                self.scroll_start_y = current_y
                return "DOWN"
        return None

    def reset_states(self) -> None:
        """Safely resets all interaction states and releases held buttons."""
        if self.dragging:
            pyautogui.mouseUp()
        self.left_click_state = False
        self.right_click_state = False
        self.dragging = False
        self.scroll_mode = False


class HUDOverlay:
    """Renders modern, semi-transparent Head-Up Display elements on the video feed."""

    # Colors (BGR)
    COLOR_BG_CARD = (25, 25, 30)
    COLOR_ACCENT = (255, 140, 0)       # Cyan / Blue
    COLOR_MOVE = (80, 220, 100)        # Vibrant Green
    COLOR_LEFT_CLICK = (60, 60, 255)    # Vibrant Red
    COLOR_RIGHT_CLICK = (255, 120, 40)  # Blue
    COLOR_DRAG = (0, 215, 255)         # Yellow/Gold
    COLOR_SCROLL = (255, 80, 200)      # Purple/Magenta
    COLOR_STOP = (140, 140, 140)       # Gray
    COLOR_ZONE = (100, 200, 255)       # Soft Gold/Cyan

    @classmethod
    def get_gesture_color(cls, gesture: GestureType) -> Tuple[int, int, int]:
        mapping = {
            GestureType.MOVE: cls.COLOR_MOVE,
            GestureType.LEFT_CLICK: cls.COLOR_LEFT_CLICK,
            GestureType.RIGHT_CLICK: cls.COLOR_RIGHT_CLICK,
            GestureType.DRAG: cls.COLOR_DRAG,
            GestureType.SCROLL: cls.COLOR_SCROLL,
            GestureType.STOP: cls.COLOR_STOP,
            GestureType.READY: (200, 200, 200),
            GestureType.NONE: (50, 50, 180),
        }
        return mapping.get(gesture, (220, 220, 220))

    @staticmethod
    def draw_glass_pill(
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.65,
    ) -> None:
        """Draws a semi-transparent colored badge/card overlay."""
        x1, y1 = pt1
        x2, y2 = pt2
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    @classmethod
    def render(
        cls,
        frame: np.ndarray,
        gesture: GestureType,
        fingers: List[int],
        positions: Dict[str, Tuple[int, int]],
        thumb_index_dist: float,
        index_middle_dist: float,
        thumb_middle_dist: float,
        config: MouseConfig,
        fps: float,
        is_fullscreen: bool,
    ) -> None:
        """Renders the comprehensive HUD onto the output frame."""
        h, w, _ = frame.shape

        # 1. Interaction Zone (Active Cursor Movement Area)
        if config.show_mapping_zone:
            mx, my = config.margin_x, config.margin_y
            cv2.rectangle(frame, (mx, my), (w - mx, h - my), cls.COLOR_ZONE, 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "Cursor Active Zone",
                (mx + 6, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                cls.COLOR_ZONE,
                1,
                cv2.LINE_AA,
            )

        # 2. Highlight fingertip proximity lines if hand detected
        if positions and "thumb" in positions and "index" in positions:
            t_pt = positions["thumb"]
            i_pt = positions["index"]
            ti_color = cls.COLOR_LEFT_CLICK if thumb_index_dist < config.click_threshold else (180, 180, 180)
            cv2.line(frame, t_pt, i_pt, ti_color, 2, cv2.LINE_AA)

            if "middle" in positions:
                m_pt = positions["middle"]
                im_color = cls.COLOR_DRAG if index_middle_dist < config.click_threshold else (180, 180, 180)
                cv2.line(frame, i_pt, m_pt, im_color, 2, cv2.LINE_AA)

            # Draw halo on index fingertip
            cv2.circle(frame, i_pt, 8, cls.COLOR_MOVE, -1, cv2.LINE_AA)
            cv2.circle(frame, i_pt, 14, cls.COLOR_MOVE, 1, cv2.LINE_AA)

        if not config.show_hud:
            return

        # 3. Top Banner: Active Gesture Card
        badge_w, badge_h = 320, 60
        cls.draw_glass_pill(frame, (20, 20), (20 + badge_w, 20 + badge_h), cls.COLOR_BG_CARD, alpha=0.75)
        accent_color = cls.get_gesture_color(gesture)
        cv2.circle(frame, (45, 50), 12, accent_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (45, 50), 16, accent_color, 1, cv2.LINE_AA)

        cv2.putText(
            frame,
            gesture.value,
            (70, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 4. Metrics & Diagnostics Card (Top Right)
        diag_w, diag_h = 310, 110
        diag_x = w - diag_w - 20
        cls.draw_glass_pill(frame, (diag_x, 20), (w - 20, 20 + diag_h), cls.COLOR_BG_CARD, alpha=0.70)

        fingers_str = "".join(str(f) for f in fingers) if fingers else "-----"
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Fingers: [{fingers_str}]",
            (diag_x + 12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Thumb-Index Dist: {int(thumb_index_dist):3d}px (Thr: {int(config.click_threshold)})",
            (diag_x + 12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Smooth: {config.smoothening:.1f} | Cooldown: {config.click_cooldown:.2f}s",
            (diag_x + 12, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Screen Size: {config.display_width}x{config.display_height} | Mode: {'FULL' if is_fullscreen else 'WINDOW'}",
            (diag_x + 12, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        # 5. Bottom Navigation / Hotkey Bar
        bar_h = 36
        cls.draw_glass_pill(frame, (20, h - bar_h - 15), (w - 20, h - 15), cls.COLOR_BG_CARD, alpha=0.75)
        hotkeys_msg = (
            "[F] Toggle Fullscreen  |  [H] Toggle HUD  |  [+/-] Sensitivity  |  [Q/ESC] Quit"
        )
        cv2.putText(
            frame,
            hotkeys_msg,
            (35, h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


class FingerMouseApp:
    """Master application coordinator managing camera capture, processing, and display."""

    WINDOW_NAME = "Finger Controlled Mouse"

    def __init__(self, config: MouseConfig):
        self.config = config
        self.detector = HandDetector(config)
        self.controller = MouseController(config)
        self.is_fullscreen = config.fullscreen

        # Camera setup
        self.camera = cv2.VideoCapture(config.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera device with index {config.camera_id}")

        # Attempt to set camera resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_cam_width = actual_w if actual_w > 0 else config.camera_width
        self.actual_cam_height = actual_h if actual_h > 0 else config.camera_height
        logger.info("Camera initialized at resolution: %dx%d", self.actual_cam_width, self.actual_cam_height)

        # Configure OpenCV display window
        self._setup_window()

    def _setup_window(self) -> None:
        """Configures a resizable OpenCV window with full-screen support."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.is_fullscreen:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.config.display_width, self.config.display_height)

    def toggle_fullscreen(self) -> None:
        """Toggles between fullscreen and normal resizable windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            logger.info("Display switched to Fullscreen Mode")
        else:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.config.display_width, self.config.display_height)
            logger.info("Display switched to Windowed Mode (%dx%d)", self.config.display_width, self.config.display_height)

    def classify_gesture(
        self,
        fingers: List[int],
        thumb_index_close: bool,
        thumb_middle_close: bool,
        index_middle_close: bool,
    ) -> GestureType:
        """Evaluates finger postures and distances into a designated GestureType."""
        total_fingers = sum(fingers)

        # 1. Point / Move Cursor (Index finger extended alone)
        if total_fingers == 1 and fingers[1] == 1:
            return GestureType.MOVE

        # 2. Left Click (Thumb + Index extended & pinched)
        if fingers[0] == 1 and fingers[1] == 1 and total_fingers == 2 and thumb_index_close:
            return GestureType.LEFT_CLICK

        # 3. Right Click (Thumb + Index + Middle extended & pinched)
        if (
            fingers[0] == 1
            and fingers[1] == 1
            and fingers[2] == 1
            and total_fingers == 3
            and thumb_index_close
            and thumb_middle_close
        ):
            return GestureType.RIGHT_CLICK

        # 4. Drag & Drop (Index + Middle extended & held together)
        if fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2 and index_middle_close:
            return GestureType.DRAG

        # 5. Scroll Mode (Index + Middle + Ring extended upward)
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[0] == 0 and fingers[4] == 0:
            return GestureType.SCROLL

        # 6. Stop / Emergency Brake (Fist - no fingers extended)
        if total_fingers == 0:
            return GestureType.STOP

        return GestureType.READY

    def run(self) -> None:
        """Main processing loop for camera capture, gesture detection, and UI rendering."""
        logger.info("Starting Finger Controlled Mouse application...")
        logger.info("Controls: Point=Move, Thumb+Index=LeftClick, Thumb+Index+Middle=RightClick, Index+Middle=Drag")
        logger.info("Press 'f' for Fullscreen, 'h' to toggle HUD, 'q' or ESC to quit.")

        prev_frame_time = time.time()
        fps = 30.0

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    logger.error("Failed to read frame from camera.")
                    time.sleep(0.05)
                    continue

                # Mirror frame for natural intuitive interaction
                frame = cv2.flip(frame, 1)

                # Upscale frame for high-definition rendering on large/fullscreen displays
                display_w = self.config.display_width
                display_h = self.config.display_height
                if (frame.shape[1], frame.shape[0]) != (display_w, display_h):
                    frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

                current_time = time.time()
                time_delta = current_time - prev_frame_time
                if time_delta > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / time_delta)
                prev_frame_time = current_time

                # Detect hands
                results = self.detector.process_frame(frame)

                gesture = GestureType.NONE
                fingers: List[int] = []
                positions: Dict[str, Tuple[int, int]] = {}
                thumb_index_dist = 999.0
                index_middle_dist = 999.0
                thumb_middle_dist = 999.0

                if results.multi_hand_landmarks:
                    # Identify primary hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = "Right"
                    if results.multi_handedness:
                        handedness = results.multi_handedness[0].classification[0].label

                    # Draw skeleton
                    self.detector.draw_hand_skeleton(frame, hand_landmarks)

                    # Extract positions and finger states
                    positions = self.detector.get_pixel_positions(
                        hand_landmarks.landmark, display_w, display_h
                    )
                    fingers = self.detector.evaluate_fingers_up(hand_landmarks.landmark, handedness)

                    # Calculate distances
                    thumb_index_dist = self.detector.calculate_distance(positions["thumb"], positions["index"])
                    index_middle_dist = self.detector.calculate_distance(positions["index"], positions["middle"])
                    thumb_middle_dist = self.detector.calculate_distance(positions["thumb"], positions["middle"])

                    thr = self.config.click_threshold
                    thumb_index_close = thumb_index_dist < thr
                    index_middle_close = index_middle_dist < thr
                    thumb_middle_close = thumb_middle_dist < thr

                    # Classify gesture
                    gesture = self.classify_gesture(
                        fingers, thumb_index_close, thumb_middle_close, index_middle_close
                    )

                    # Execute corresponding action
                    if gesture == GestureType.MOVE:
                        self.controller.move_cursor(positions["index"][0], positions["index"][1], display_w, display_h)
                        self.controller.reset_states()

                    elif gesture == GestureType.LEFT_CLICK:
                        self.controller.move_cursor(positions["index"][0], positions["index"][1], display_w, display_h)
                        self.controller.trigger_left_click()
                        if self.controller.dragging:
                            self.controller.set_drag_state(False)

                    elif gesture == GestureType.RIGHT_CLICK:
                        self.controller.move_cursor(positions["index"][0], positions["index"][1], display_w, display_h)
                        self.controller.trigger_right_click()
                        if self.controller.dragging:
                            self.controller.set_drag_state(False)

                    elif gesture == GestureType.DRAG:
                        self.controller.move_cursor(positions["index"][0], positions["index"][1], display_w, display_h)
                        self.controller.set_drag_state(True)
                        self.controller.left_click_state = False
                        self.controller.right_click_state = False

                    elif gesture == GestureType.SCROLL:
                        self.controller.handle_scroll(positions["index"][1])

                    elif gesture == GestureType.STOP:
                        self.controller.reset_states()

                    else:  # READY / transition state
                        if not (fingers[0] == 1 and fingers[1] == 1 and thumb_index_close):
                            self.controller.left_click_state = False
                        if not (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and thumb_index_close and thumb_middle_close):
                            self.controller.right_click_state = False
                        if self.controller.dragging and not (fingers[1] == 1 and fingers[2] == 1 and index_middle_close):
                            self.controller.set_drag_state(False)
                        if self.controller.scroll_mode and not (fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1):
                            self.controller.scroll_mode = False

                else:
                    self.controller.reset_states()

                # Render HUD
                HUDOverlay.render(
                    frame=frame,
                    gesture=gesture,
                    fingers=fingers,
                    positions=positions,
                    thumb_index_dist=thumb_index_dist,
                    index_middle_dist=index_middle_dist,
                    thumb_middle_dist=thumb_middle_dist,
                    config=self.config,
                    fps=fps,
                    is_fullscreen=self.is_fullscreen,
                )

                # Show frame
                cv2.imshow(self.WINDOW_NAME, frame)

                # Handle hotkeys
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    logger.info("Exit requested by user.")
                    break
                elif key in (ord("f"), ord("F")):
                    self.toggle_fullscreen()
                elif key in (ord("h"), ord("H")):
                    self.config.show_hud = not self.config.show_hud
                    logger.info("HUD visibility toggled to: %s", self.config.show_hud)
                elif key in (ord("+"), ord("=")):
                    self.config.click_threshold = min(120.0, self.config.click_threshold + 5.0)
                    logger.info("Click threshold increased to: %.1f", self.config.click_threshold)
                elif key in (ord("-"), ord("_")):
                    self.config.click_threshold = max(20.0, self.config.click_threshold - 5.0)
                    logger.info("Click threshold decreased to: %.1f", self.config.click_threshold)

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Releases camera resources and cleans up OS windows."""
        logger.info("Cleaning up resources...")
        self.controller.reset_states()
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def parse_arguments() -> MouseConfig:
    """Parses command-line arguments and returns a configured MouseConfig object."""
    parser = argparse.ArgumentParser(
        description="Finger Controlled Mouse (Air Mouse) - Touchless vision-based mouse control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in Fullscreen mode directly.")
    parser.add_argument("--width", type=int, default=1280, help="Display window width.")
    parser.add_argument("--height", type=int, default=720, help="Display window height.")
    parser.add_argument("--smooth", type=float, default=5.0, help="Cursor smoothing factor (higher = smoother).")
    parser.add_argument("--threshold", type=float, default=55.0, help="Pinch distance threshold in pixels.")
    parser.add_argument("--cooldown", type=float, default=0.18, help="Click cooldown time in seconds.")
    parser.add_argument("--no-hud", action="store_true", help="Disable the Head-Up Display overlay.")

    args = parser.parse_args()

    return MouseConfig(
        camera_id=args.camera,
        fullscreen=args.fullscreen,
        display_width=args.width,
        display_height=args.height,
        smoothening=args.smooth,
        click_threshold=args.threshold,
        click_cooldown=args.cooldown,
        show_hud=not args.no_hud,
    )


def main() -> None:
    """Application entrypoint."""
    config = parse_arguments()
    try:
        app = FingerMouseApp(config)
        app.run()
    except KeyboardInterrupt:
        logger.info("Program terminated by user keyboard interrupt.")
    except Exception as exc:
        logger.exception("Unexpected error during execution: %s", exc)
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    main()