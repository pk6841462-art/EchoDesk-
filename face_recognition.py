"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 2 - Face Recognition Engine (Owner vs Guest)
File:    Python/face_recognition.py

Description:
This module recognizes whether the person sitting at the desk is the registered "Owner"
or a "Guest". 

Features:
1. Primary: Uses the `face_recognition` dlib-backed encoding library if installed.
2. Fallback: Uses OpenCV grayscale feature difference matrix similarity matching.
3. Automatically loads and caches the reference face from `faces/owner.jpg`.
================================================================================
"""

import os
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition as fr_lib
    if hasattr(fr_lib, "face_encodings"):
        FACE_REC_LIB_AVAILABLE = True
    else:
        FACE_REC_LIB_AVAILABLE = False
except Exception:
    FACE_REC_LIB_AVAILABLE = False


class FaceRecognition:
    """Recognize desk user identity: Owner vs Guest."""

    def __init__(self, owner_image_path: str = "faces/owner.jpg"):
        """
        Initialize face recognizer and load owner profile image.
        
        :param owner_image_path: Relative or absolute path to owner reference image.
        """
        self.owner_image_path = self._resolve_owner_image_path(owner_image_path)
        self.owner_encoding = None
        self.owner_gray_image = None
        self.use_lib = FACE_REC_LIB_AVAILABLE

        self._load_owner_profile()

    def _resolve_owner_image_path(self, owner_image_path: str) -> str:
        """Resolve owner image path from common locations and extensions."""
        if not owner_image_path:
            return ""

        candidate_paths = []
        provided = Path(owner_image_path)

        if provided.is_absolute():
            candidate_paths.append(provided)
        else:
            candidate_paths.extend([
                Path(owner_image_path),
                Path("faces") / owner_image_path,
                Path("Python") / owner_image_path,
                Path.cwd() / owner_image_path,
            ])

            base_name = provided.stem
            suffixes = [provided.suffix] if provided.suffix else [".jpg", ".jpeg", ".png", ".bmp"]
            for suffix in suffixes:
                candidate_paths.append(Path(owner_image_path))
                candidate_paths.append(Path("faces") / f"{base_name}{suffix}")
                candidate_paths.append(Path("faces") / owner_image_path)
                candidate_paths.append(Path.cwd() / "faces" / f"{base_name}{suffix}")

        seen = set()
        for path in candidate_paths:
            try:
                resolved = str(path)
            except Exception:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if os.path.exists(resolved):
                return resolved

        return str(provided)

    def _load_owner_profile(self):
        """Load owner face encoding or template image from disk."""
        if not os.path.exists(self.owner_image_path):
            print(f"[FaceRecognition] Warning: Owner reference image missing at '{self.owner_image_path}'. Identity defaults to 'Guest'.")
            return

        try:
            if self.use_lib:
                owner_bgr = cv2.imread(self.owner_image_path)
                if owner_bgr is not None:
                    owner_rgb = cv2.cvtColor(owner_bgr, cv2.COLOR_BGR2RGB)
                    encodings = fr_lib.face_encodings(owner_rgb)
                    if len(encodings) > 0:
                        self.owner_encoding = encodings[0]
                        print("[FaceRecognition] Successfully loaded Owner face encoding via face_recognition library.")
                        return
            
            # OpenCV Grayscale Template Fallback
            self.owner_gray_image = cv2.imread(self.owner_image_path, cv2.IMREAD_GRAYSCALE)
            if self.owner_gray_image is not None:
                self.owner_gray_image = cv2.resize(self.owner_gray_image, (120, 120))
                print("[FaceRecognition] Loaded Owner reference image template for OpenCV matching.")
        except Exception as exc:
            print(f"[FaceRecognition] Error initializing owner profile: {exc}")

    def identify(self, frame: np.ndarray) -> str:
        """
        Identify whether the live camera frame contains the 'Owner' or a 'Guest'.
        
        :param frame: Live video frame (BGR numpy array).
        :return: 'Owner' or 'Guest'
        """
        if frame is None or frame.size == 0:
            return "Guest"

        # 1. Primary: face_recognition 128D Deep Feature Vector Comparison
        if self.use_lib and self.owner_encoding is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = fr_lib.face_locations(rgb_frame)
                if len(face_locations) > 0:
                    encodings = fr_lib.face_encodings(rgb_frame, face_locations)
                    for enc in encodings:
                        matches = fr_lib.compare_faces([self.owner_encoding], enc, tolerance=0.55)
                        if True in matches:
                            return "Owner"
                    return "Guest"
            except Exception:
                pass

        # 2. Fallback: OpenCV Face Cascade Detection & Template Metric Comparison
        if self.owner_gray_image is not None:
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                if hasattr(face_cascade, 'empty') and not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
                else:
                    faces = []

                for (x, y, w, h) in faces:
                    face_roi = gray_frame[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (120, 120))
                    
                    # Compute mean absolute difference score
                    diff = cv2.absdiff(face_resized, self.owner_gray_image)
                    score = float(np.mean(diff))

                    # Lower score = higher similarity
                    if score < 75.0:
                        return "Owner"

                return "Guest"
            except Exception:
                return "Guest"

        return "Guest"


# Quick Standalone Test
if __name__ == "__main__":
    print("=== Testing Module 2: FaceRecognition ===")
    fr = FaceRecognition()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    identity = fr.identify(dummy_frame)
    print(f"Dummy Frame Recognized Identity: {identity}")
