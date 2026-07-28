"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 5 - Personal Cognitive Environment Model (PCEM)
File:    Python/pcem.py

Description:
This module implements the Personal Cognitive Environment Model (PCEM).

Core Functionality:
1. Baseline vs Personalized Profiles:
   - Guest Mode: Always evaluates against fixed, default environment standards.
   - Owner Mode: Dynamically adapts to the owner's personal productivity patterns.
2. Continual Learning:
   - Accumulates session history in `pcem_profile.json`.
   - Updates running moving averages of optimal temperature, humidity, light,
     and session duration when high-focus study sessions occur.
3. Adaptation Affinity Scoring:
   - Calculates a 0-100 Adaptation Affinity Score based on how closely current
     workspace conditions match the owner's learned baseline preferences.
================================================================================
"""

import json
import os
from typing import Dict, Any, Tuple


class PCEMEngine:
    """Personal Cognitive Environment Model (PCEM) Adaptive Engine."""

    def __init__(self, profile_path: str = "pcem_profile.json"):
        """
        Initialize PCEM Engine and load or initialize profile.
        
        :param profile_path: Path to profile JSON file.
        """
        self.profile_path = profile_path
        self.profile: Dict[str, Any] = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        """Load profile JSON from disk if present, else return default baseline."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict) and "sessions" in data:
                        return data
            except Exception as exc:
                print(f"[PCEMEngine] Warning loading profile ({exc}). Using default profile.")

        return {
            "temperature": 24.5,
            "humidity": 50.0,
            "light": 450.0,
            "study_duration": 45.0,
            "posture": "Good",
            "sessions": 0,
            "owner_name": "Owner"
        }

    def save_profile(self) -> None:
        """Persist the updated profile to disk."""
        try:
            with open(self.profile_path, "w", encoding="utf-8") as handle:
                json.dump(self.profile, handle, indent=2)
        except Exception as exc:
            print(f"[PCEMEngine] Error saving profile: {exc}")

    def update_owner_profile(
        self,
        identity: str,
        temperature: float,
        humidity: float,
        light: float,
        study_duration: float,
        posture: str,
        focus_score: float
    ) -> bool:
        """
        Update the owner's learned preference profile when a successful study session completes.
        
        Note: Guest sessions never update the owner's PCEM profile!
        
        :return: True if profile was updated, False if skipped (e.g. Guest mode or low focus).
        """
        if identity != "Owner" or focus_score < 70.0:
            return False

        sessions = self.profile.get("sessions", 0) + 1
        self.profile["sessions"] = sessions

        # Incremental moving average update formula
        prev_t = float(self.profile.get("temperature", 24.5))
        prev_h = float(self.profile.get("humidity", 50.0))
        prev_l = float(self.profile.get("light", 450.0))
        prev_d = float(self.profile.get("study_duration", 45.0))

        self.profile["temperature"] = round((prev_t * (sessions - 1) + float(temperature)) / sessions, 1)
        self.profile["humidity"] = round((prev_h * (sessions - 1) + float(humidity)) / sessions, 1)
        self.profile["light"] = round((prev_l * (sessions - 1) + float(light)) / sessions, 1)
        self.profile["study_duration"] = round((prev_d * (sessions - 1) + float(study_duration)) / sessions, 1)
        self.profile["posture"] = posture

        self.save_profile()
        print(f"[PCEMEngine] Owner Profile Updated! Total Sessions Learned: {sessions}")
        return True

    def calculate_adaptation_score(
        self,
        identity: str,
        temperature: float,
        humidity: float,
        light: float,
        study_duration: float
    ) -> float:
        """
        Compute the Personal Adaptation Score (0-100).
        
        :return: Score float between 0.0 and 100.0.
        """
        if identity != "Owner":
            # Guest mode uses standard baseline static affinity (75.0)
            return 75.0

        target_temp = float(self.profile.get("temperature", 24.5))
        target_hum = float(self.profile.get("humidity", 50.0))
        target_light = float(self.profile.get("light", 450.0))
        target_duration = float(self.profile.get("study_duration", 45.0))

        # Absolute deviation penalties
        delta_temp = abs(float(temperature) - target_temp)
        delta_hum = abs(float(humidity) - target_hum)
        delta_light = abs(float(light) - target_light)
        delta_dur = abs(float(study_duration) - target_duration)

        score = 100.0
        score -= min(40.0, delta_temp * 2.5)
        score -= min(20.0, delta_hum * 0.8)
        score -= min(20.0, delta_light / 12.0)
        score -= min(20.0, delta_dur / 3.0)

        return round(max(0.0, min(100.0, score)), 1)


# Standalone Test
if __name__ == "__main__":
    print("=== Testing Module 5: PCEMEngine ===")
    pcem = PCEMEngine()
    score = pcem.calculate_adaptation_score("Owner", 25.0, 52.0, 420.0, 30.0)
    print(f"Sample Owner PCEM Adaptation Score: {score}%")
