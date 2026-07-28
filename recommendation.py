"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 6 - Recommendation Engine
File:    Python/recommendation.py

Description:
This module translates trained Machine Learning predictions (Focus Score, Fatigue Score)
and PCEM adaptation affinity into human-readable action prompts for the student.

Output Recommendations:
- 'Sit Straight'
- 'Increase Room Light'
- 'Take 5 Minute Break'
- 'Room Too Hot'
- 'Turn On Fan'
- 'Excellent Focus'
- 'Good Productivity'
- 'Needs Improvement'
- 'High Fatigue'
================================================================================
"""

from typing import Dict, Any, Tuple


class RecommendationEngine:
    """Generate practical human-readable recommendations from ML predictions & telemetry."""

    def __init__(self, pcem_engine=None):
        """
        Initialize Recommendation Engine.
        
        :param pcem_engine: Instance of PCEMEngine.
        """
        self.pcem_engine = pcem_engine

    def generate_recommendation(
        self,
        focus_score: float,
        fatigue_score: float,
        posture: str,
        temperature: float,
        light: float,
        study_duration: float,
        user_presence: int = 1
    ) -> Tuple[str, str]:
        """
        Generate recommendation text string and hardware buzzer alert code.
        
        :return: Tuple of (recommendation_string, alert_code)
        """
        if user_presence == 0:
            return "User Absent", "NONE"

        # Priority 1: Physical Ergonomic Safety (Posture Correction)
        if posture == "Poor":
            return "Sit Straight", "BAD_POSTURE"

        # Priority 2: Critical Fatigue & Session Duration Alerts
        if fatigue_score >= 75.0 or study_duration >= 120.0:
            return "Take 5 Minute Break", "TAKE_BREAK"

        if fatigue_score >= 65.0:
            return "High Fatigue", "HIGH_FATIGUE"

        # Priority 3: Environmental Thermal & Lighting Adjustments
        if temperature >= 32.0:
            return "Turn On Fan", "HIGH_TEMP"

        if temperature >= 29.0:
            return "Room Too Hot", "HIGH_TEMP"

        if light < 200.0:
            return "Increase Room Light", "LOW_LIGHT"

        # Priority 4: High Productivity & Focus States
        if focus_score >= 82.0:
            return "Excellent Focus", "NONE"

        if focus_score >= 65.0:
            return "Good Productivity", "NONE"

        if fatigue_score >= 50.0:
            return "Take 5 Minute Break", "TAKE_BREAK"

        return "Needs Improvement", "NONE"
