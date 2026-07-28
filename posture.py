"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 2 - Posture Estimator Classifier
File:    Python/posture.py

Description:
This module evaluates sitting posture quality in real time using webcam frames.

Posture Classifications:
- 'Good': Straight spine, aligned neck and shoulders.
- 'Slight Lean': Slight head tilt or slight forward lean.
- 'Moderate Lean': Noticeable forward neck slump or uneven shoulders.
- 'Poor': Severe slouching, excessive forward head posture, or hunching.

Technique:
1. Primary: MediaPipe Pose landmark geometry (computing neck angle and shoulder slope).
2. Fallback: Upper-body contour aspect ratio analysis.
================================================================================
"""

import math
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


class PostureEstimator:
    """Analyze upper body geometry to classify sitting posture quality."""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize Posture Estimator.
        
        :param min_detection_confidence: Landmark detection confidence threshold.
        """
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.last_posture = "Good"

        if self.mediapipe_available:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )

    def estimate(self, frame: np.ndarray) -> str:
        """
        Classify sitting posture from the input camera frame.
        
        :param frame: Live BGR video frame.
        :return: Posture label string: 'Good', 'Slight Lean', 'Moderate Lean', or 'Poor'.
        """
        if frame is None or frame.size == 0:
            return self.last_posture

        # 1. Primary: MediaPipe 3D Pose Landmarks Analysis
        if self.mediapipe_available:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark

                    # Landmark indices:
                    # 0: Nose, 7: Left Ear, 8: Right Ear, 11: Left Shoulder, 12: Right Shoulder
                    nose = landmarks[0]
                    left_ear = landmarks[7]
                    right_ear = landmarks[8]
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]

                    # Check visibility
                    if (left_shoulder.visibility > 0.4 and right_shoulder.visibility > 0.4):
                        # Midpoint of shoulders
                        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
                        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0

                        # Calculate neck inclination angle (degrees from vertical)
                        dx = nose.x - mid_shoulder_x
                        dy = mid_shoulder_y - nose.y  # Inverted y-axis in image coordinates

                        # Avoid division by zero
                        angle_rad = math.atan2(abs(dx), max(dy, 1e-4))
                        neck_angle_deg = math.degrees(angle_rad)

                        # Shoulder tilt angle (slope between left & right shoulder)
                        shoulder_dy = abs(left_shoulder.y - right_shoulder.y)
                        shoulder_dx = max(abs(left_shoulder.x - right_shoulder.x), 1e-4)
                        shoulder_tilt_deg = math.degrees(math.atan2(shoulder_dy, shoulder_dx))

                        # Determine posture classification based on biomechanical thresholds
                        if neck_angle_deg < 12.0 and shoulder_tilt_deg < 6.0:
                            self.last_posture = "Good"
                        elif neck_angle_deg < 20.0 and shoulder_tilt_deg < 12.0:
                            self.last_posture = "Slight Lean"
                        elif neck_angle_deg < 30.0 or shoulder_tilt_deg < 20.0:
                            self.last_posture = "Moderate Lean"
                        else:
                            self.last_posture = "Poor"

                        return self.last_posture
            except Exception:
                pass

        # 2. Fallback: Upper-Body Bounding Ratio Heuristic
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                aspect_ratio = w / max(1.0, float(h))

                if aspect_ratio > 1.1:
                    self.last_posture = "Good"
                elif aspect_ratio > 0.8:
                    self.last_posture = "Slight Lean"
                elif aspect_ratio > 0.6:
                    self.last_posture = "Moderate Lean"
                else:
                    self.last_posture = "Poor"
        except Exception:
            pass

        return self.last_posture

    def close(self):
        """Release MediaPipe resources cleanly."""
        if self.mediapipe_available and hasattr(self, 'pose'):
            self.pose.close()


# Quick Standalone Test
if __name__ == "__main__":
    print("=== Testing Module 2: PostureEstimator ===")
    estimator = PostureEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    label = estimator.estimate(dummy_frame)
    print(f"Dummy Frame Posture Label: {label}")
    estimator.close()
