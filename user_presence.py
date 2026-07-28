"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 2 - User Presence Detector
File:    Python/user_presence.py

Description:
This module determines if a person is actively seated in front of the desk setup.
It uses a hybrid approach:
1. Primary: MediaPipe Pose landmark presence checking.
2. Fallback: Motion contour delta tracking between consecutive video frames.

Output:
Returns True if user presence is detected, False otherwise.
================================================================================
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        MEDIAPIPE_AVAILABLE = True
    else:
        MEDIAPIPE_AVAILABLE = False
except Exception:
    MEDIAPIPE_AVAILABLE = False


class UserPresenceDetector:
    """Detect active user presence in front of the laptop/desk webcam."""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the User Presence Detector.
        
        :param min_detection_confidence: Confidence threshold for landmark presence.
        """
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.previous_frame = None
        
        if self.mediapipe_available:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )

    def detect(self, frame: np.ndarray) -> bool:
        """
        Evaluate if a user is present in the provided BGR image frame.
        
        :param frame: OpenCV BGR image numpy array.
        :return: True if user is present, False if absent.
        """
        if frame is None or frame.size == 0:
            return False

        # Attempt MediaPipe pose detection if library is installed
        if self.mediapipe_available:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks is not None:
                # Check visibility of nose and shoulder landmarks
                landmarks = results.pose_landmarks.landmark
                nose_visible = landmarks[0].visibility > 0.5
                left_shoulder_visible = landmarks[11].visibility > 0.3
                right_shoulder_visible = landmarks[12].visibility > 0.3
                
                if nose_visible or left_shoulder_visible or right_shoulder_visible:
                    return True

        # Fallback Heuristic: Frame motion and foreground contour area tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return True  # Default to present on first frame initialization

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        motion_pixel_count = cv2.countNonZero(thresh)
        self.previous_frame = gray

        # Also check standard face cascade detection in frame safely
        faces = []
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if hasattr(face_cascade, 'empty') and not face_cascade.empty():
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        except Exception:
            faces = []

        if len(faces) > 0 or motion_pixel_count > 3000:
            return True

        return False

    def close(self):
        """Release MediaPipe resources cleanly."""
        if self.mediapipe_available and hasattr(self, 'pose'):
            self.pose.close()


# Quick Standalone Test
if __name__ == "__main__":
    print("=== Testing Module 2: UserPresenceDetector ===")
    detector = UserPresenceDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    presence = detector.detect(dummy_frame)
    print(f"Dummy Frame Presence Result: {presence}")
    detector.close()
