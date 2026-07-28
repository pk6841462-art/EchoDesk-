"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 2 - Camera Capture & High-Tech HUD Overlay
File:    Python/camera.py

Description:
This module manages OpenCV webcam capture and real-time Heads-Up Display (HUD)
rendering for live desktop visualization.

Features:
1. Video Capture: Connects to webcam index 0 (or custom index).
2. Demo Mode Fallback: Generates synthetic UI frames if no physical webcam is attached.
3. Modern HUD Overlay: Renders system stats, user identity, posture badge, focus/fatigue scores,
   and live recommendations directly on the video window.
================================================================================
"""

import time
from typing import Dict, Any, Optional
import cv2
import numpy as np


class Camera:
    """Webcam video capture wrapper with real-time HUD rendering."""

    def __init__(self, camera_index: int = 0, demo: bool = False):
        """
        Initialize the laptop webcam stream or fall back to a demo generator.
        
        :param camera_index: Hardware webcam index (Default: 0).
        :param demo: Force demo mode if True.
        """
        self.demo = demo
        self.capture: Optional[cv2.VideoCapture] = None
        self.capture_backend: Optional[str] = None
        self.frame_width = 640
        self.frame_height = 480

        if not self.demo:
            self.capture, self.capture_backend = self._open_camera(camera_index)
            if self.capture is None:
                print(f"[Camera] Laptop webcam index {camera_index} unavailable. Switching to Demo Mode.")
                self.demo = True
            else:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                print(f"[Camera] Using laptop camera via {self.capture_backend} (index {camera_index}).")

        # Baseline synthetic canvas for demo mode
        self.demo_canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    def _open_camera(self, camera_index: int) -> tuple[Optional[cv2.VideoCapture], Optional[str]]:
        """Try common Windows camera backends so the laptop webcam opens reliably."""
        candidates = []
        if hasattr(cv2, "CAP_DSHOW"):
            candidates.append(("DirectShow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("Media Foundation", cv2.CAP_MSMF))
        candidates.append(("Auto", cv2.CAP_ANY))

        for backend_name, backend_id in candidates:
            try:
                capture = cv2.VideoCapture(camera_index, backend_id)
            except Exception:
                capture = None

            if capture is not None and capture.isOpened():
                return capture, backend_name

            if capture is not None:
                capture.release()

        return None, None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return the next video frame.
        
        :return: OpenCV BGR image numpy array or synthetic demo frame.
        """
        if self.demo or self.capture is None:
            return self._generate_demo_canvas()

        success, frame = self.capture.read()
        if not success or frame is None:
            return self._generate_demo_canvas()

        return frame

    def show_frame(
        self,
        frame: np.ndarray,
        context: Dict[str, Any],
        recommendation: str,
        focus_score: float,
        fatigue_score: float,
        window_name: str = "EchoDesk AI Monitor"
    ):
        """
        Render real-time HUD overlay stats onto the frame and display in an OpenCV window.
        
        :param frame: Base BGR video frame.
        :param context: System context dictionary (identity, posture, user_present, temp, light, etc.)
        :param recommendation: Actionable recommendation string.
        :param focus_score: Focus score (0-100).
        :param fatigue_score: Fatigue score (0-100).
        :param window_name: Window title string.
        """
        if frame is None:
            return

        h, w = frame.shape[:2]
        hud = frame.copy()

        # Semi-transparent dark overlay panel at top
        overlay = hud.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 140), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        cv2.rectangle(hud, (10, 10), (w - 10, 140), (0, 220, 255), 1)

        # Header Title
        cv2.putText(hud, "ECHODESK - AI PRODUCTIVITY MONITOR", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Identity & Presence
        user_id = context.get("identity", "Guest")
        present = "Present" if context.get("user_present", True) else "Absent"
        id_color = (0, 255, 0) if user_id == "Owner" else (255, 200, 0)
        cv2.putText(hud, f"User: {user_id} ({present})", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 1)

        # Posture Badge
        posture = context.get("posture", "Good")
        posture_color = (0, 255, 0) if posture == "Good" else ((0, 200, 255) if posture == "Slight Lean" else (0, 0, 255))
        cv2.putText(hud, f"Posture: {posture}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 1)

        # Focus & Fatigue Scores
        cv2.putText(hud, f"Focus Score: {int(focus_score)}%", (250, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(hud, f"Fatigue Index: {int(fatigue_score)}%", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # Telemetry Summary Line
        temp = context.get("temperature", 25.0)
        hum = context.get("humidity", 50.0)
        light = context.get("light", 400.0)
        cv2.putText(hud, f"Sensors: {temp}C | {hum}% Hum | {light} Lux", (250, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Live Recommendation Banner
        cv2.rectangle(hud, (15, 110), (w - 15, 132), (50, 50, 60), -1)
        cv2.putText(hud, f"REC: {recommendation}", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Render window
        cv2.imshow(window_name, hud)

    def _generate_demo_canvas(self) -> np.ndarray:
        """Generate a simulated camera canvas for demo execution."""
        canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # Add subtle grid background lines
        for y in range(0, self.frame_height, 40):
            cv2.line(canvas, (0, y), (self.frame_width, y), (30, 30, 35), 1)
        for x in range(0, self.frame_width, 40):
            cv2.line(canvas, (x, 0), (x, self.frame_height), (30, 30, 35), 1)

        cv2.putText(canvas, "[DEMO WEBCAM FEED]", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(canvas, "Simulated Camera Input Active", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return canvas

    def release(self):
        """Release camera hardware and close all OpenCV windows."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()


# Quick Standalone Test
if __name__ == "__main__":
    print("=== Testing Module 2: Camera ===")
    cam = Camera(demo=True)
    frame = cam.read_frame()
    print(f"Captured frame shape: {frame.shape if frame is not None else None}")
    cam.release()
