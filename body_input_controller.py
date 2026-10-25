#!/usr/bin/env python3
"""
BodyPlay - turns upper-body motion into ordinary keyboard presses.

Design goals



mm
m
ml
m



------------
1. UPPER BODY ONLY. Only the nose, eyes, ears and shoulders are used, so a
   webcam that crops your legs (or even your shoulders) still works.
2. SCALE INVARIANT. Every measurement is divided by the width of your own
   head/shoulders, so standing closer or further from the camera does not
   change the sensitivity.
3. RELATIVE, NOT ABSOLUTE. Everything is measured as a change from a
   calibrated "neutral" pose, so your camera angle does not matter.
4. LOW LATENCY. The camera runs on its own thread with a 1-frame buffer and
   MediaPipe runs in LIVE_STREAM (async) mode, so we always work on the
   newest frame instead of a backlog of old ones.

Run:
    python body_input_controller.py --install      (first time only)
    python body_input_controller.py --profile subway_surfer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Bootstrap: install dependencies before we try to import them.
# --------------------------------------------------------------------------
DEPENDENCIES = ["opencv-python", "mediapipe", "pynput", "numpy"]

if "--install" in sys.argv:
    print("Installing BodyPlay dependencies (one time only)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *DEPENDENCIES])
    print("\nAll set. Now run:  python body_input_controller.py")
    sys.exit(0)

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from pynput.keyboard import Controller as KeyboardController, Key
except ImportError as exc:  # pragma: no cover
    print(f"A library is missing ({exc}).")
    print("Fix it with one command:  python body_input_controller.py --install")
    sys.exit(1)


# --------------------------------------------------------------------------
# Files and model
# --------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
CONFIG_FILE = os.path.join(APP_DIR, "body_input_settings.json")
MODEL_FILE = os.path.join(APP_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# MediaPipe Pose landmark numbers we care about (upper body only).
NOSE = 0
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 3, 6
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
UPPER_BODY = (NOSE, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_EAR, RIGHT_EAR,
              LEFT_SHOULDER, RIGHT_SHOULDER)

ACTIONS = ("left", "right", "up", "down")

DEFAULT_PROFILES: Dict[str, Dict[str, str]] = {
    "arrow_runner":         {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "temple_run_1":         {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "temple_run_2":         {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "subway_surfer":        {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "geometry_dash":        {"left": "left", "right": "right", "up": "space", "down": "down"},
    "blades_of_brim":       {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "talking_tom_gold_run": {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "sonic_dash":           {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "minion_rush":          {"left": "left", "right": "right", "up": "up",    "down": "down"},
    "wasd_runner":          {"left": "a",    "right": "d",     "up": "w",     "down": "s"},
    "touch_swipe":          {"left": "left", "right": "right", "up": "up",    "down": "down"},
}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    # --- camera ---
    camera_index: int = 0
    capture_width: int = 640      # 640x480 (4:3) gives the TALLEST field of view
    capture_height: int = 480     # on most webcams. 1280x720 crops the top/bottom.
    capture_fps: int = 60
    mirror: bool = True
    show_overlay: bool = True

    # --- how movement becomes a key ---
    hold_keys: bool = True             # steering keys stay held while you lean
    pulse_actions: List[str] = field(default_factory=lambda: ["up", "down"])
    cooldown_sec: float = 0.10         # minimum gap between two taps of one key
    start_armed: bool = False          # False = press P to start sending keys

    # --- sensitivity (SMALLER NUMBER = EASIER TO TRIGGER) ---
    lean_sensitivity: float = 0.13     # sideways: lean, head-turn or head-tilt
    jump_sensitivity: float = 0.10     # how high the head must rise
    jump_velocity: float = 0.80        # how FAST the head must rise (head-widths/sec)
    duck_sensitivity: float = 0.10     # how far the head must drop
    release_ratio: float = 0.55        # key releases at 55% of the trigger level

    # --- which body cues feed the sideways signal (0 turns one off) ---
    lean_gain: float = 1.0             # whole head moves sideways over shoulders
    yaw_gain: float = 1.0              # head TURNS left/right
    tilt_gain: float = 0.8             # head TILTS ear-toward-shoulder

    # --- smoothing / stability ---
    smoothing_alpha: float = 0.6       # 1.0 = no smoothing (fastest, jitterier)
    scale_smoothing: float = 0.06      # how fast the body-size reference updates
    gesture_frames_on: int = 2         # frames a gesture must hold before firing
    gesture_frames_off: int = 2        # frames it must stop before releasing
    neutral_adaptation: float = 0.02   # slow auto re-centering while you stand still
    calibrate_seconds: float = 1.5
    auto_calibrate_frames: int = 20

    # --- detection ---
    min_visibility: float = 0.05       # low on purpose: partial framing still works
    detection_confidence: float = 0.30
    tracking_confidence: float = 0.30

    # --- keys ---
    left_key: str = "left"
    right_key: str = "right"
    up_key: str = "up"
    down_key: str = "down"
    game_profile: str = "subway_surfer"
    profiles: Dict[str, Dict[str, str]] = field(default_factory=lambda: dict(DEFAULT_PROFILES))


def load_settings() -> Settings:
    settings = Settings()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            print("body_input_settings.json could not be read; using built-in defaults.")
            return settings
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    merged = dict(DEFAULT_PROFILES)
    merged.update(settings.profiles or {})
    settings.profiles = merged
    return settings


def save_settings(settings: Settings) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2)


def apply_profile(settings: Settings, name: Optional[str] = None) -> None:
    name = name or settings.game_profile
    profile = settings.profiles.get(name)
    if profile is None:
        print(f"Profile '{name}' not found. Using arrow_runner.")
        name, profile = "arrow_runner", DEFAULT_PROFILES["arrow_runner"]
    settings.game_profile = name
    for action in ACTIONS:
        if profile.get(action):
            setattr(settings, action + "_key", profile[action])


def ensure_model() -> None:
    """Find the pose model. Packaged builds carry it inside the app."""
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 100_000:
        return
    bundled = os.path.join(getattr(sys, "_MEIPASS", APP_DIR), "pose_landmarker_lite.task")
    if os.path.exists(bundled) and os.path.getsize(bundled) > 100_000:
        import shutil
        shutil.copyfile(bundled, MODEL_FILE)
        return
    print("Getting the pose model (about 6 MB, one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Pose model ready.")


# --------------------------------------------------------------------------
# Keyboard output
# --------------------------------------------------------------------------
SPECIAL_KEYS = {
    "left": Key.left, "right": Key.right, "up": Key.up, "down": Key.down,
    "space": Key.space, "enter": Key.enter, "return": Key.enter,
    "shift": Key.shift, "ctrl": Key.ctrl, "alt": Key.alt, "tab": Key.tab,
    "esc": Key.esc, "escape": Key.esc, "backspace": Key.backspace,
}


def parse_key(name: str):
    name = str(name).strip().lower()
    if name in SPECIAL_KEYS:
        return SPECIAL_KEYS[name]
    if len(name) == 1:
        return name
    raise ValueError(f"Unsupported key name: {name}")


class KeyEmitter:
    """Turns a dict like {'left': True, 'up': False} into real key events."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.keyboard = KeyboardController()
        self.held: set = set()
        self.last_press: Dict[str, float] = {}
        self.latched: set = set()   # pulse actions waiting to be re-armed
        self.armed = settings.start_armed

    def _key(self, action: str):
        return parse_key(getattr(self.settings, action + "_key"))

    def sync(self, active: Dict[str, bool]) -> List[str]:
        fired: List[str] = []
        if not self.armed:
            self.release_all()
            return fired
        now = time.perf_counter()
        pulse = set(self.settings.pulse_actions or [])
        for action in ACTIONS:
            wants = active.get(action, False)
            is_pulse = action in pulse or not self.settings.hold_keys
            if is_pulse:
                if wants and action not in self.latched:
                    if now - self.last_press.get(action, 0.0) >= self.settings.cooldown_sec:
                        key = self._key(action)
                        self.keyboard.press(key)
                        self.keyboard.release(key)
                        self.last_press[action] = now
                        self.latched.add(action)
                        fired.append(action)
                elif not wants:
                    self.latched.discard(action)
            else:
                if wants and action not in self.held:
                    self.keyboard.press(self._key(action))
                    self.held.add(action)
                    fired.append(action)
                elif not wants and action in self.held:
                    self.keyboard.release(self._key(action))
                    self.held.discard(action)
        return fired

    def release_all(self) -> None:
        for action in list(self.held):
            try:
                self.keyboard.release(self._key(action))
            except Exception:
                pass
        self.held.clear()
        self.latched.clear()

    def toggle_arm(self) -> bool:
        self.armed = not self.armed
        if not self.armed:
            self.release_all()
        return self.armed


# --------------------------------------------------------------------------
# Low-latency camera
# --------------------------------------------------------------------------
class CameraStream:
    """Reads frames on a background thread and always hands back the newest."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.capture = self._open(settings.camera_index)
        self.frame = None
        self.stamp = 0.0
        self.count = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self, index: int):
        system = platform.system()
        backends = [cv2.CAP_ANY]
        if system == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif system == "Darwin":
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif system == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.capture_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.capture_height)
                capture.set(cv2.CAP_PROP_FPS, self.settings.capture_fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture
            capture.release()
        raise RuntimeError(
            f"Camera {index} did not open. Run with --list-cameras to see which "
            f"numbers work, then set camera_index in body_input_settings.json."
        )

    def _loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.stamp = time.perf_counter()
                self.count += 1

    def read(self) -> Tuple[Optional["np.ndarray"], int, float]:
        with self.lock:
            if self.frame is None:
                return None, self.count, 0.0
            return self.frame.copy(), self.count, self.stamp

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


# --------------------------------------------------------------------------
# Pose tracking (async so the main loop never waits)
# --------------------------------------------------------------------------
class PoseTracker:
    def __init__(self, settings: Settings):
        self.lock = threading.Lock()
        self.result = None
        self.result_stamp = 0.0
        self.last_ts = 0
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILE),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=settings.detection_confidence,
            min_pose_presence_confidence=settings.detection_confidence,
            min_tracking_confidence=settings.tracking_confidence,
            output_segmentation_masks=False,
            result_callback=self._on_result,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        with self.lock:
            self.result = result
            self.result_stamp = time.perf_counter()

    def submit(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        stamp = int(time.perf_counter() * 1000)
        if stamp <= self.last_ts:
            stamp = self.last_ts + 1
        self.last_ts = stamp
        self.landmarker.detect_async(image, stamp)

    def latest(self):
        with self.lock:
            return self.result, self.result_stamp

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Turning landmarks into three numbers: steer, rise, drop
# --------------------------------------------------------------------------
def visibility_of(landmark) -> float:
    vis = getattr(landmark, "visibility", None)
    pres = getattr(landmark, "presence", None)
    values = [v for v in (vis, pres) if v is not None]
    return max(values) if values else 1.0


class SignalReader:
    """Produces scale-invariant numbers from the upper-body landmarks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.scale = None          # heavily smoothed body-size reference
        self.smooth: Dict[str, float] = {}
        self.history = deque(maxlen=8)   # (time, head_y) for velocity

    def reset(self):
        self.scale = None
        self.smooth = {}
        self.history.clear()

    def read(self, landmarks, aspect: float) -> Optional[Dict[str, float]]:
        """aspect = frame_height / frame_width, used to make x and y comparable."""
        nose = landmarks[NOSE]
        l_ear, r_ear = landmarks[LEFT_EAR], landmarks[RIGHT_EAR]
        l_eye, r_eye = landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER]
        l_sh, r_sh = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]

        if visibility_of(nose) < self.settings.min_visibility:
            return None

        def point(lm):
            return lm.x, lm.y * aspect

        nose_x, nose_y = point(nose)
        lex, ley = point(l_ear)
        rex, rey = point(r_ear)
        leyex, _ = point(l_eye)
        reyex, _ = point(r_eye)
        lsx, lsy = point(l_sh)
        rsx, rsy = point(r_sh)

        ear_span = abs(lex - rex)
        eye_span = abs(leyex - reyex)
        head_span = max(ear_span, eye_span * 1.45, 0.02)

        shoulders_visible = (
            visibility_of(l_sh) >= self.settings.min_visibility
            and visibility_of(r_sh) >= self.settings.min_visibility
        )
        shoulder_span = abs(lsx - rsx) if shoulders_visible else 0.0

        # One stable size reference. Shoulders are the best ruler; the head is
        # the backup when shoulders are out of frame.
        raw_scale = shoulder_span if shoulder_span > 0.05 else head_span * 2.4
        if self.scale is None:
            self.scale = raw_scale
        else:
            beta = max(0.005, min(0.5, self.settings.scale_smoothing))
            self.scale = beta * raw_scale + (1 - beta) * self.scale
        scale = max(self.scale, 1e-4)

        ear_mid_x = (lex + rex) / 2.0
        ear_mid_y = (ley + rey) / 2.0
        head_x = 0.5 * nose_x + 0.5 * ear_mid_x
        head_y = 0.55 * nose_y + 0.45 * ear_mid_y

        # 1) whole head shifted sideways relative to the shoulders
        lean = (head_x - (lsx + rsx) / 2.0) / scale if shoulders_visible else 0.0
        # 2) head TURNED: the nose slides away from the middle of the two ears
        yaw = (nose_x - ear_mid_x) / (scale * 0.45)
        # 3) head TILTED: one ear drops lower than the other
        tilt = (rey - ley) / (scale * 0.45)
        # 4) vertical position of the head
        clearance = ((lsy + rsy) / 2.0 - head_y) / scale if shoulders_visible else 0.0

        raw = {
            "lean": lean, "yaw": yaw, "tilt": tilt,
            "head_y": head_y / scale, "clearance": clearance,
            "shoulders": 1.0 if shoulders_visible else 0.0,
            "scale": scale,
        }

        alpha = max(0.05, min(1.0, self.settings.smoothing_alpha))
        if not self.smooth:
            self.smooth = dict(raw)
        else:
            for key in ("lean", "yaw", "tilt", "head_y", "clearance"):
                self.smooth[key] = alpha * raw[key] + (1 - alpha) * self.smooth[key]
            self.smooth["shoulders"] = raw["shoulders"]
            self.smooth["scale"] = raw["scale"]

        now = time.perf_counter()
        self.history.append((now, self.smooth["head_y"]))
        rise_speed = 0.0
        if len(self.history) >= 2:
            t0, y0 = self.history[0]
            dt = now - t0
            if dt > 1e-3:
                rise_speed = (y0 - self.smooth["head_y"]) / dt  # +ve = moving up
        self.smooth["rise_speed"] = rise_speed
        return dict(self.smooth)


class GestureDetector:
    """Compares live signals against your calibrated neutral pose."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.neutral: Optional[Dict[str, float]] = None
        self.state = {a: False for a in ACTIONS}
        self.on_count = {a: 0 for a in ACTIONS}
        self.off_count = {a: 0 for a in ACTIONS}
        self.calibrating = False
        self.calib_samples: List[Dict[str, float]] = []
        self.calib_until = 0.0
        self.auto_frames = 0
        self.debug = {"steer": 0.0, "rise": 0.0, "drop": 0.0, "speed": 0.0}

    # ---- calibration -----------------------------------------------------
    def start_calibration(self):
        self.calibrating = True
        self.calib_samples = []
        self.calib_until = time.perf_counter() + self.settings.calibrate_seconds
        self.state = {a: False for a in ACTIONS}

    def _finish_calibration(self):
        if self.calib_samples:
            keys = ("lean", "yaw", "tilt", "head_y", "clearance")
            self.neutral = {
                k: sum(s[k] for s in self.calib_samples) / len(self.calib_samples)
                for k in keys
            }
        self.calibrating = False
        self.calib_samples = []

    def reset(self):
        self.neutral = None
        self.auto_frames = 0
        self.state = {a: False for a in ACTIONS}
        self.on_count = {a: 0 for a in ACTIONS}
        self.off_count = {a: 0 for a in ACTIONS}

    # ---- the actual decision --------------------------------------------
    def update(self, signals: Optional[Dict[str, float]]) -> Dict[str, bool]:
        if signals is None:
            for action in ACTIONS:
                self.off_count[action] += 1
                if self.off_count[action] >= self.settings.gesture_frames_off:
                    self.state[action] = False
            return dict(self.state)

        if self.calibrating:
            self.calib_samples.append(signals)
            if time.perf_counter() >= self.calib_until:
                self._finish_calibration()
            return {a: False for a in ACTIONS}

        if self.neutral is None:
            self.calib_samples.append(signals)
            self.auto_frames += 1
            if self.auto_frames >= self.settings.auto_calibrate_frames:
                self._finish_calibration()
            return {a: False for a in ACTIONS}

        s, n, cfg = signals, self.neutral, self.settings

        # --- sideways: whichever cue is strongest wins -------------------
        candidates = [
            (s["lean"] - n["lean"]) * cfg.lean_gain if s["shoulders"] else 0.0,
            (s["yaw"] - n["yaw"]) * cfg.yaw_gain,
            (s["tilt"] - n["tilt"]) * cfg.tilt_gain,
        ]
        steer = max(candidates, key=abs)

        # --- vertical ----------------------------------------------------
        rise = n["head_y"] - s["head_y"]          # + = head above neutral
        if s["shoulders"]:
            rise = max(rise, s["clearance"] - n["clearance"])
        drop = -rise
        speed = s.get("rise_speed", 0.0)

        self.debug = {"steer": steer, "rise": rise, "drop": drop, "speed": speed}

        on_t = {
            "left":  steer <= -cfg.lean_sensitivity,
            "right": steer >= cfg.lean_sensitivity,
            "up":    rise >= cfg.jump_sensitivity or speed >= cfg.jump_velocity,
            "down":  drop >= cfg.duck_sensitivity,
        }
        release = max(0.1, min(0.95, cfg.release_ratio))
        off_t = {
            "left":  steer <= -cfg.lean_sensitivity * release,
            "right": steer >= cfg.lean_sensitivity * release,
            "up":    rise >= cfg.jump_sensitivity * release or speed >= cfg.jump_velocity * release,
            "down":  drop >= cfg.duck_sensitivity * release,
        }

        for action in ACTIONS:
            wants = on_t[action] if not self.state[action] else off_t[action]
            if wants:
                self.on_count[action] += 1
                self.off_count[action] = 0
                if self.on_count[action] >= cfg.gesture_frames_on:
                    self.state[action] = True
            else:
                self.off_count[action] += 1
                self.on_count[action] = 0
                if self.off_count[action] >= cfg.gesture_frames_off:
                    self.state[action] = False

        if self.state["left"] and self.state["right"]:
            loser = "right" if steer < 0 else "left"
            self.state[loser] = False
        if self.state["up"] and self.state["down"]:
            self.state["down" if rise > 0 else "up"] = False

        # Gently re-center while you stand still, so slow drift never builds up.
        if not any(self.state.values()):
            rate = max(0.0, min(0.2, cfg.neutral_adaptation))
            for key in ("lean", "yaw", "tilt", "head_y", "clearance"):
                self.neutral[key] = rate * s[key] + (1 - rate) * self.neutral[key]

        return dict(self.state)


# --------------------------------------------------------------------------
# Overlay drawing
# --------------------------------------------------------------------------
GREEN, YELLOW, CYAN, GREY, RED = (0, 220, 0), (0, 220, 255), (255, 220, 0), (150, 150, 150), (60, 60, 255)


def draw_text(frame, text, x, y, color=GREEN, scale=0.55, thick=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_meter(frame, x, y, width, value, threshold, label, bipolar=False):
    height = 12
    cv2.rectangle(frame, (x, y), (x + width, y + height), (45, 45, 45), -1)
    if bipolar:
        centre = x + width // 2
        span = width // 2
        filled = int(max(-1.0, min(1.0, value / max(threshold * 2, 1e-6))) * span)
        colour = GREEN if abs(value) >= threshold else CYAN
        cv2.rectangle(frame, (centre, y), (centre + filled, y + height), colour, -1)
        for sign in (-1, 1):
            mark = centre + int(sign * (threshold / max(threshold * 2, 1e-6)) * span)
            cv2.line(frame, (mark, y), (mark, y + height), YELLOW, 1)
        cv2.line(frame, (centre, y - 2), (centre, y + height + 2), GREY, 1)
    else:
        ratio = max(0.0, min(1.0, value / max(threshold * 2, 1e-6)))
        colour = GREEN if value >= threshold else CYAN
        cv2.rectangle(frame, (x, y), (x + int(ratio * width), y + height), colour, -1)
        mark = x + width // 2
        cv2.line(frame, (mark, y), (mark, y + height), YELLOW, 1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), GREY, 1)
    draw_text(frame, f"{label} {value:+.3f}", x + width + 8, y + height - 1, GREY, 0.45)


def draw_landmarks(frame, landmarks, aspect_unused=None):
    h, w = frame.shape[:2]
    pts = {}
    for index in UPPER_BODY:
        lm = landmarks[index]
        pts[index] = (int(lm.x * w), int(lm.y * h))
    for a, b in ((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_EAR, RIGHT_EAR),
                 (LEFT_EAR, LEFT_SHOULDER), (RIGHT_EAR, RIGHT_SHOULDER)):
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for index, point in pts.items():
        cv2.circle(frame, point, 4, YELLOW if index == NOSE else CYAN, -1)


def draw_framing_guide(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * 0.18), int(h * 0.05)), (int(w * 0.82), int(h * 0.80)), (70, 70, 70), 1)
    draw_text(frame, "Fit your head + shoulders in this box", 12, h - 14, GREY, 0.5)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def list_cameras(maximum: int = 6) -> None:
    print("Scanning camera indexes...")
    for index in range(maximum):
        capture = cv2.VideoCapture(index)
        ok = capture.isOpened()
        if ok:
            got, frame = capture.read()
            size = f"{frame.shape[1]}x{frame.shape[0]}" if got else "no frame"
            print(f"  camera_index {index}: WORKS ({size})")
        capture.release()
    print("Put the number you want into body_input_settings.json -> camera_index")


def nudge(settings: Settings, attribute: str, delta: float, low: float, high: float) -> str:
    value = round(min(high, max(low, getattr(settings, attribute) + delta)), 3)
    setattr(settings, attribute, value)
    return f"{attribute} = {value}"


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="BodyPlay upper-body motion controller")
    parser.add_argument("--profile", help="game profile name from body_input_settings.json")
    parser.add_argument("--camera", type=int, help="camera index override")
    parser.add_argument("--list-cameras", action="store_true", help="show working camera numbers")
    parser.add_argument("--install", action="store_true", help="install dependencies and exit")
    parser.add_argument("--armed", action="store_true", help="start sending keys immediately")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    settings = load_settings()
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.armed:
        settings.start_armed = True
    apply_profile(settings, args.profile)
    ensure_model()

    reader = SignalReader(settings)
    detector = GestureDetector(settings)
    emitter = KeyEmitter(settings)
    tracker = PoseTracker(settings)
    camera = CameraStream(settings)

    print(f"Profile: {settings.game_profile}  ->  "
          f"L={settings.left_key} R={settings.right_key} U={settings.up_key} D={settings.down_key}")
    print("P = arm/disarm keys | C = calibrate | R = reset | Q = quit")

    last_count = -1
    last_frame_time = time.perf_counter()
    fps = 0.0
    flash_text, flash_until = "", 0.0
    status, status_until = "", 0.0
    window = "BodyPlay - Motion Controller"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame, count, stamp = camera.read()
            if frame is None:
                time.sleep(0.005)
                continue
            if count != last_count:
                last_count = count
                if settings.mirror:
                    frame = cv2.flip(frame, 1)
                tracker.submit(frame)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_frame_time, 1e-6))
                last_frame_time = now
            elif settings.mirror:
                frame = cv2.flip(frame, 1)

            result, result_stamp = tracker.latest()
            landmarks = None
            if result is not None and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

            aspect = frame.shape[0] / float(frame.shape[1])
            signals = reader.read(landmarks, aspect) if landmarks else None
            active = detector.update(signals)
            fired = emitter.sync(active)
            if fired:
                flash_text = "  ".join(a.upper() for a in fired)
                flash_until = time.perf_counter() + 0.30

            if settings.show_overlay:
                draw_framing_guide(frame)
                if landmarks:
                    draw_landmarks(frame, landmarks)
                h, w = frame.shape[:2]
                panel = frame[0:150, 0:w].copy()
                frame[0:150, 0:w] = cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0)

                armed_text = "ARMED - keys are going to the game" if emitter.armed else "PAUSED - press P to arm"
                draw_text(frame, armed_text, 12, 22, GREEN if emitter.armed else RED, 0.6, 2)
                tracking = "tracking you" if signals else "looking for you"
                if detector.calibrating:
                    tracking = "CALIBRATING - hold still, look straight ahead"
                elif detector.neutral is None:
                    tracking = "learning your neutral pose - stand normally"
                draw_text(frame, f"{tracking}   |   {fps:4.1f} fps   |   profile: {settings.game_profile}",
                          12, 44, YELLOW, 0.5)
                pressed = " ".join(a.upper() for a in ACTIONS if active.get(a)) or "-"
                draw_text(frame, f"keys now: {pressed}", 12, 64, CYAN, 0.5)

                d = detector.debug
                draw_meter(frame, 12, 74, 180, d["steer"], settings.lean_sensitivity, "steer", bipolar=True)
                draw_meter(frame, 12, 94, 180, d["rise"], settings.jump_sensitivity, "rise")
                draw_meter(frame, 12, 114, 180, d["drop"], settings.duck_sensitivity, "duck")
                draw_text(frame, "P arm  C calibrate  R reset  [ ] lean  - = jump  ; ' duck  , . speed  "
                                 "H hold  M mirror  O overlay  TAB profile  S save  Q quit",
                          12, 146, GREY, 0.42)

                if time.perf_counter() < flash_until:
                    draw_text(frame, flash_text, w - 220, 40, (0, 200, 255), 1.0, 2)
                if time.perf_counter() < status_until:
                    draw_text(frame, status, 12, frame.shape[0] - 40, (0, 200, 255), 0.6, 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            message = ""
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                message = "keys ARMED" if emitter.toggle_arm() else "keys PAUSED"
            elif key == ord("c"):
                detector.start_calibration()
                message = "calibrating - stand normally and hold still"
            elif key == ord("r"):
                reader.reset()
                detector.reset()
                emitter.release_all()
                message = "reset - learning neutral again"
            elif key == ord("o"):
                settings.show_overlay = not settings.show_overlay
            elif key == ord("m"):
                settings.mirror = not settings.mirror
                message = f"mirror {'on' if settings.mirror else 'off'}"
            elif key == ord("h"):
                settings.hold_keys = not settings.hold_keys
                emitter.release_all()
                message = f"hold_keys {'on' if settings.hold_keys else 'off'}"
            elif key == ord("["):
                message = nudge(settings, "lean_sensitivity", -0.01, 0.02, 1.0) + "  (easier to steer)"
            elif key == ord("]"):
                message = nudge(settings, "lean_sensitivity", 0.01, 0.02, 1.0) + "  (harder to steer)"
            elif key == ord("-"):
                message = nudge(settings, "jump_sensitivity", -0.01, 0.01, 1.0) + "  (easier to jump)"
            elif key == ord("="):
                message = nudge(settings, "jump_sensitivity", 0.01, 0.01, 1.0) + "  (harder to jump)"
            elif key == ord(";"):
                message = nudge(settings, "duck_sensitivity", -0.01, 0.01, 1.0) + "  (easier to duck)"
            elif key == ord("'"):
                message = nudge(settings, "duck_sensitivity", 0.01, 0.01, 1.0) + "  (harder to duck)"
            elif key == ord(","):
                message = nudge(settings, "jump_velocity", -0.05, 0.05, 5.0) + "  (easier fast-jump)"
            elif key == ord("."):
                message = nudge(settings, "jump_velocity", 0.05, 0.05, 5.0) + "  (harder fast-jump)"
            elif key == 9:  # TAB
                names = sorted(settings.profiles)
                index = (names.index(settings.game_profile) + 1) % len(names) \
                    if settings.game_profile in names else 0
                emitter.release_all()
                apply_profile(settings, names[index])
                message = f"profile: {settings.game_profile}"
            elif key == ord("s"):
                save_settings(settings)
                message = "settings saved to body_input_settings.json"

            if message:
                status, status_until = message, time.perf_counter() + 2.0
                print(message)
    finally:
        emitter.release_all()
        save_settings(settings)
        tracker.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
