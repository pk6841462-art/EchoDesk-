"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 5 - Machine Learning Prediction Engine
File:    Python/prediction.py

Description:
This module handles live ML inference for EchoDesk.
It loads the serialized model (`ML/focus_model.pkl`) trained in Module 4
and calculates real-time predictions for:
1. Focus Score (0-100)
2. Fatigue Score (0-100)
3. Productivity Class ('Excellent Focus', 'Good Productivity', 'Needs Improvement', 'High Fatigue')
================================================================================
"""

import os
from typing import Dict, Any
import joblib
import pandas as pd


class EchoDeskPredictor:
    """Live Machine Learning Inference Engine using trained Gradient Boosting Regressor."""

    def __init__(self, model_path: str = "ML/focus_model.pkl"):
        """
        Initialize predictor and load ML model artifact.
        
        :param model_path: Path to serialized model pkl file.
        """
        self.model_path = self._resolve_model_path(model_path)
        self.model = self._load_model()
        self.feature_columns = [
            "temperature",
            "humidity",
            "light",
            "study_duration",
            "posture_score",
            "user_presence",
            "owner_mode_bin",
            "light_temperature_ratio",
            "temp_humidity_ratio",
            "stress_flag"
        ]

    def _resolve_model_path(self, model_path: str) -> str:
        """Search multiple relative locations for focus_model.pkl."""
        candidates = [
            model_path,
            os.path.join("ML", "focus_model.pkl"),
            os.path.join(".", "focus_model.pkl"),
            os.path.join("..", "ML", "focus_model.pkl"),
            os.path.join(os.path.dirname(__file__), "..", "ML", "focus_model.pkl"),
            os.path.join(os.path.dirname(__file__), "..", "focus_model.pkl")
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return model_path

    def _load_model(self):
        """Load scikit-learn model pipeline from joblib pickle file."""
        if not os.path.exists(self.model_path):
            print(f"[EchoDeskPredictor] Warning: Model file missing at '{self.model_path}'. Using fallback ML predictor.")
            return None

        try:
            model = joblib.load(self.model_path)
            print(f"[EchoDeskPredictor] Successfully loaded trained ML model from '{self.model_path}'.")
            return model
        except Exception as exc:
            print(f"[EchoDeskPredictor] Error loading model: {exc}")
            return None

    def _build_feature_frame(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Convert input payload dictionary into feature matrix DataFrame."""
        posture_str = str(payload.get("posture", "Good"))
        posture_map = {"Good": 3, "Slight Lean": 2, "Moderate Lean": 1, "Poor": 0}
        posture_score = posture_map.get(posture_str, 3)

        identity = str(payload.get("identity", payload.get("owner_mode", "Owner")))
        owner_bin = 1 if identity.lower() in {"owner", "1"} else 0

        temp = float(payload.get("temperature", 25.0))
        hum = float(payload.get("humidity", 50.0))
        light = float(payload.get("light", 450.0))
        duration = float(payload.get("study_duration", 30.0))
        presence = int(payload.get("user_presence", 1))

        # Interaction engineered features
        light_temp_ratio = light / (temp + 1.0)
        temp_hum_ratio = temp / (hum + 1.0)
        stress_flag = int((temp > 32.0) or (light < 180.0) or (duration > 90.0) or (posture_score == 0))

        feature_dict = {
            "temperature": temp,
            "humidity": hum,
            "light": light,
            "study_duration": duration,
            "posture_score": posture_score,
            "user_presence": presence,
            "owner_mode_bin": owner_bin,
            "light_temperature_ratio": light_temp_ratio,
            "temp_humidity_ratio": temp_hum_ratio,
            "stress_flag": stress_flag
        }

        return pd.DataFrame([feature_dict], columns=self.feature_columns)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict Focus Score, Fatigue Score, and Productivity Class for a given payload.
        
        :param payload: Dictionary containing telemetry & CV features.
        :return: Dictionary with keys 'focus_score', 'fatigue_score', 'productivity_class'.
        """
        presence = int(payload.get("user_presence", 1))
        if presence == 0:
            return {
                "focus_score": 0.0,
                "fatigue_score": 0.0,
                "productivity_class": "Inactive"
            }

        # Build feature DataFrame
        feature_frame = self._build_feature_frame(payload)

        # 1. Execute ML Model Inference if model is loaded
        if self.model is not None:
            try:
                raw_pred = float(self.model.predict(feature_frame)[0])
                focus_score = round(max(0.0, min(100.0, raw_pred)), 1)
            except Exception as exc:
                print(f"[EchoDeskPredictor] Inference warning ({exc}). Using heuristic fallback.")
                focus_score = self._heuristic_fallback(payload)
        else:
            focus_score = self._heuristic_fallback(payload)

        # Compute Fatigue Index with domain penalty modifiers
        duration = float(payload.get("study_duration", 30.0))
        posture = str(payload.get("posture", "Good"))
        temp = float(payload.get("temperature", 25.0))

        raw_fatigue = (100.0 - focus_score) + min(25.0, duration / 3.5)
        if posture == "Poor":
            raw_fatigue += 12.0
        if temp > 32.0:
            raw_fatigue += 8.0

        fatigue_score = round(max(0.0, min(100.0, raw_fatigue)), 1)

        # Categorize Productivity Class
        if fatigue_score >= 70.0:
            prod_class = "High Fatigue"
        elif focus_score >= 82.0:
            prod_class = "Excellent Focus"
        elif focus_score >= 65.0:
            prod_class = "Good Productivity"
        else:
            prod_class = "Needs Improvement"

        return {
            "focus_score": focus_score,
            "fatigue_score": fatigue_score,
            "productivity_class": prod_class
        }

    def _heuristic_fallback(self, payload: Dict[str, Any]) -> float:
        """Fallback scoring formula if model artifact is unreadable."""
        temp = float(payload.get("temperature", 25.0))
        hum = float(payload.get("humidity", 50.0))
        light = float(payload.get("light", 450.0))
        posture = str(payload.get("posture", "Good"))

        t_score = max(0, 100 - abs(temp - 24.5) * 4.0)
        l_score = max(0, 100 - abs(light - 450.0) / 6.0)
        p_map = {"Good": 100, "Slight Lean": 85, "Moderate Lean": 65, "Poor": 40}
        p_score = p_map.get(posture, 50)

        return round(0.35 * t_score + 0.35 * l_score + 0.30 * p_score, 1)


# Standalone Test Block
if __name__ == "__main__":
    print("=== Testing Module 5: EchoDeskPredictor ===")
    predictor = EchoDeskPredictor()
    sample_payload = {
        "temperature": 25.5,
        "humidity": 52.0,
        "light": 480.0,
        "study_duration": 35.0,
        "posture": "Good",
        "user_presence": 1,
        "identity": "Owner"
    }
    result = predictor.predict(sample_payload)
    print(f"Prediction Result: {result}")
