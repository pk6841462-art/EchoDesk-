"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 3 - Dataset Collector & Telemetry Generator (Calibrated)
File:    Python/dataset.py

Description:
Manages dataset logging and generates realistic study session training samples
with calibrated indoor environmental distributions (Temp: 21-28.5°C, Humidity: 42-60%, Light: 350-650 Lux).
================================================================================
"""

import os
import csv
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np


class DatasetCollector:
    """Collect real-time study observations and generate ML training datasets."""

    def __init__(self, csv_path: str = "Dataset/study_dataset.csv"):
        """
        Initialize dataset collector.
        
        :param csv_path: Output CSV file path.
        """
        self.csv_path = csv_path
        self.fieldnames = [
            "timestamp",
            "session_id",
            "temperature",
            "humidity",
            "light",
            "study_duration",
            "posture",
            "user_presence",
            "owner_mode",
            "focus_score",
            "fatigue_score",
            "productivity_class",
            "recommendation"
        ]
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the CSV file and directory header if missing."""
        directory = os.path.dirname(self.csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def collect_row(
        self,
        temperature: float,
        humidity: float,
        light: float,
        study_duration: float,
        posture: str,
        user_presence: int,
        owner_mode: int,
        focus_score: float,
        fatigue_score: float,
        productivity_class: str,
        recommendation: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Append a single structured telemetry record to the dataset CSV.
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or datetime.now().strftime("SESS_%Y%m%d_%H%M%S"),
            "temperature": round(float(temperature), 1),
            "humidity": round(float(humidity), 1),
            "light": round(float(light), 1),
            "study_duration": round(float(study_duration), 1),
            "posture": posture,
            "user_presence": int(user_presence),
            "owner_mode": int(owner_mode),
            "focus_score": round(float(focus_score), 1),
            "fatigue_score": round(float(fatigue_score), 1),
            "productivity_class": productivity_class,
            "recommendation": recommendation
        }

        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)

        return row

    def generate_synthetic_dataset(self, num_samples: int = 1200) -> str:
        """
        Generate a realistic, diverse training dataset (1200+ samples) with calibrated room metrics.
        
        :param num_samples: Number of study session sample rows to generate.
        :return: Path to generated CSV file.
        """
        print(f"[DatasetCollector] Generating calibrated study session dataset ({num_samples} rows)...")

        np.random.seed(42)
        random.seed(42)

        start_time = datetime.now() - timedelta(days=30)
        rows: List[Dict[str, Any]] = []

        posture_choices = ["Good", "Slight Lean", "Moderate Lean", "Poor"]
        posture_weights = [0.55, 0.25, 0.12, 0.08]

        for i in range(num_samples):
            current_timestamp = start_time + timedelta(minutes=i * 25)
            session_id = f"SESS_{(i // 15) + 1:04d}"

            # Calibrated realistic indoor room distributions
            temperature = round(float(np.random.normal(24.5, 1.4)), 1)
            temperature = max(20.5, min(29.5, temperature))

            humidity = round(float(np.random.normal(51.0, 3.5)), 1)
            humidity = max(40.0, min(65.0, humidity))

            light = round(float(np.random.normal(480.0, 50.0)), 1)
            light = max(300.0, min(700.0, light))

            study_duration = round(float(np.random.exponential(40.0)), 1)
            study_duration = max(5.0, min(150.0, study_duration))

            posture = random.choices(posture_choices, weights=posture_weights)[0]
            user_presence = 1 if random.random() > 0.05 else 0
            owner_mode = 1 if random.random() > 0.15 else 0

            # Ground-truth metric calculation
            focus_score, fatigue_score, prod_class, rec = self._calculate_baseline_labels(
                temperature=temperature,
                humidity=humidity,
                light=light,
                study_duration=study_duration,
                posture=posture,
                user_presence=user_presence,
                owner_mode=owner_mode
            )

            row = {
                "timestamp": current_timestamp.isoformat(),
                "session_id": session_id,
                "temperature": temperature,
                "humidity": humidity,
                "light": light,
                "study_duration": study_duration,
                "posture": posture,
                "user_presence": user_presence,
                "owner_mode": owner_mode,
                "focus_score": focus_score,
                "fatigue_score": fatigue_score,
                "productivity_class": prod_class,
                "recommendation": rec
            }
            rows.append(row)

        # Write to CSV file
        directory = os.path.dirname(self.csv_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[DatasetCollector] Dataset created at '{self.csv_path}' with {len(rows)} calibrated samples.")
        return self.csv_path

    def _calculate_baseline_labels(
        self,
        temperature: float,
        humidity: float,
        light: float,
        study_duration: float,
        posture: str,
        user_presence: int,
        owner_mode: int
    ):
        if user_presence == 0:
            return 0.0, 0.0, "Inactive", "User Absent"

        temp_score = 100 - min(40, abs(temperature - 24.5) * 5.0)
        hum_score = 100 - min(30, abs(humidity - 50.0) * 1.5)
        light_score = 100 - min(50, abs(light - 500.0) / 6.0)
        env_score = 0.40 * temp_score + 0.20 * hum_score + 0.40 * light_score

        posture_map = {"Good": 100, "Slight Lean": 88, "Moderate Lean": 68, "Poor": 40}
        posture_score = posture_map.get(posture, 50)
        duration_score = max(20, 100 - max(0, study_duration - 45) * 0.7)
        behaviour_score = 0.60 * posture_score + 0.40 * duration_score

        affinity_score = 92.0 if owner_mode == 1 else 75.0

        raw_focus = 0.35 * env_score + 0.45 * behaviour_score + 0.20 * affinity_score
        focus_score = round(max(10.0, min(100.0, raw_focus + np.random.normal(0, 2))), 1)

        raw_fatigue = (100.0 - focus_score) + min(25.0, study_duration / 4.0)
        if posture == "Poor":
            raw_fatigue += 12.0
        if temperature > 29.0:
            raw_fatigue += 8.0

        fatigue_score = round(max(0.0, min(100.0, raw_fatigue + np.random.normal(0, 1.5))), 1)

        if focus_score >= 80.0:
            prod_class = "High Focus"
            rec = "Excellent Focus"
        elif focus_score >= 65.0:
            prod_class = "Moderate Focus"
            rec = "Good Productivity"
        elif fatigue_score >= 65.0:
            prod_class = "Fatigued"
            rec = "Take 5 Minute Break"
        elif posture == "Poor":
            prod_class = "Poor Posture"
            rec = "Sit Straight"
        elif temperature > 29.0:
            prod_class = "Uncomfortable"
            rec = "Room Too Hot"
        elif light < 250.0:
            prod_class = "Low Light"
            rec = "Increase Room Light"
        else:
            prod_class = "Low Focus"
            rec = "Needs Improvement"

        return focus_score, fatigue_score, prod_class, rec


if __name__ == "__main__":
    collector = DatasetCollector()
    collector.generate_synthetic_dataset(num_samples=1200)
