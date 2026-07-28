"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 6 - Main System Controller & Execution Loop
File:    Python/main.py

Description:
This is the main entry point orchestrating the entire industrial EchoDesk pipeline:
1. Webcam Capture (Camera)
2. Presence Detection (UserPresenceDetector)
3. Face Recognition (FaceRecognition)
4. Posture Estimation (PostureEstimator)
5. Telemetry Ingestion (SerialBridge)
6. Trained ML Model Inference (EchoDeskPredictor)
7. Personal Cognitive Environment Model Adaptation (PCEMEngine)
8. Actionable Recommendation Generation (RecommendationEngine)
9. Hardware Actuation & OLED Feedback Dispatch (SerialBridge)
10. Dataset Logging (DatasetCollector)
================================================================================
"""

import sys
import os
import time
import cv2

# Ensure Python search path includes project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Python.camera import Camera
from Python.face_recognition import FaceRecognition
from Python.posture import PostureEstimator
from Python.user_presence import UserPresenceDetector
from Python.serial_comm import SerialBridge
from Python.prediction import EchoDeskPredictor
from Python.pcem import PCEMEngine
from Python.recommendation import RecommendationEngine
from Python.dataset import DatasetCollector


class EchoDeskSystem:
    """Main EchoDesk Industrial Controller."""

    def __init__(self, owner_image_path: str = "faces/owner.jpg", demo_mode: bool = False):
        """
        Initialize all sub-modules.
        """
        print("\n=======================================================")
        print("  Initializing EchoDesk – Self-Learning Productivity   ")
        print("=======================================================\n")

        self.camera = Camera(demo=demo_mode)
        self.face_recognizer = FaceRecognition(owner_image_path)
        self.posture_estimator = PostureEstimator()
        self.presence_detector = UserPresenceDetector()
        self.serial_bridge = SerialBridge()
        self.predictor = EchoDeskPredictor()
        self.pcem = PCEMEngine()
        self.recommendation_engine = RecommendationEngine(self.pcem)
        self.dataset_collector = DatasetCollector()

        self.start_time = time.time()
        self.last_log_time = time.time()

    def get_study_duration_minutes(self) -> float:
        """Return elapsed study session duration in minutes."""
        return round((time.time() - self.start_time) / 60.0, 1)

    def run(self):
        """Execute the main system loop until user exits."""
        print("[EchoDeskSystem] System loop started. Press 'q' on video window or Ctrl+C to exit.\n")
        
        try:
            while True:
                # 1. Capture video frame
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # 2. Detect User Presence
                user_present = self.presence_detector.detect(frame)
                presence_int = 1 if user_present else 0

                # 3. Recognize Face Identity (Owner vs Guest)
                identity = self.face_recognizer.identify(frame) if user_present else "Guest"
                owner_mode_bin = 1 if identity == "Owner" else 0

                # 4. Estimate Posture
                posture = self.posture_estimator.estimate(frame) if user_present else "Good"

                # 5. Read ESP32 Sensor Telemetry
                sensor_data = self.serial_bridge.read_sensor_frame()
                temp = sensor_data.get("temperature", 25.0)
                hum = sensor_data.get("humidity", 50.0)
                light = sensor_data.get("light", 450.0)

                study_duration = self.get_study_duration_minutes()

                # 6. Build Payload Dictionary for ML Inference & PCEM
                payload = {
                    "temperature": temp,
                    "humidity": hum,
                    "light": light,
                    "study_duration": study_duration,
                    "posture": posture,
                    "user_presence": presence_int,
                    "identity": identity,
                    "owner_mode": owner_mode_bin
                }

                # 7. Execute Trained ML Model Prediction (Gradient Boosting)
                ml_result = self.predictor.predict(payload)
                focus_score = ml_result["focus_score"]
                fatigue_score = ml_result["fatigue_score"]
                prod_class = ml_result["productivity_class"]

                # 8. Compute PCEM Adaptation Score
                adaptation_score = self.pcem.calculate_adaptation_score(
                    identity=identity,
                    temperature=temp,
                    humidity=hum,
                    light=light,
                    study_duration=study_duration
                )

                # 9. Generate Recommendation & Buzzer Alert Code
                rec_text, alert_code = self.recommendation_engine.generate_recommendation(
                    focus_score=focus_score,
                    fatigue_score=fatigue_score,
                    posture=posture,
                    temperature=temp,
                    light=light,
                    study_duration=study_duration,
                    user_presence=presence_int
                )

                # 10. Send AI Recommendation back to ESP32 OLED & Buzzer
                self.serial_bridge.send_ai_feedback(
                    recommendation=rec_text,
                    focus_score=int(focus_score),
                    identity=identity,
                    alert_code=alert_code
                )

                # 11. Log Telemetry to CSV every 5 seconds
                if time.time() - self.last_log_time >= 5.0:
                    self.last_log_time = time.time()
                    self.dataset_collector.collect_row(
                        temperature=temp,
                        humidity=hum,
                        light=light,
                        study_duration=study_duration,
                        posture=posture,
                        user_presence=presence_int,
                        owner_mode=owner_mode_bin,
                        focus_score=focus_score,
                        fatigue_score=fatigue_score,
                        productivity_class=prod_class,
                        recommendation=rec_text
                    )
                    # Continual learning update for owner profile
                    if identity == "Owner":
                        self.pcem.update_owner_profile(
                            identity=identity,
                            temperature=temp,
                            humidity=hum,
                            light=light,
                            study_duration=study_duration,
                            posture=posture,
                            focus_score=focus_score
                        )

                # 12. Render OpenCV High-Tech HUD Display
                context_hud = {
                    "identity": identity,
                    "user_present": user_present,
                    "posture": posture,
                    "temperature": temp,
                    "humidity": hum,
                    "light": light
                }
                self.camera.show_frame(
                    frame=frame,
                    context=context_hud,
                    recommendation=rec_text,
                    focus_score=focus_score,
                    fatigue_score=fatigue_score
                )

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[EchoDeskSystem] Keyboard Interrupt received. Shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        """Release camera and serial connections cleanly."""
        print("[EchoDeskSystem] Closing resources...")
        self.camera.release()
        self.presence_detector.close()
        self.posture_estimator.close()
        self.serial_bridge.close()
        print("[EchoDeskSystem] Shutdown complete.")


if __name__ == "__main__":
    system = EchoDeskSystem()
    system.run()
