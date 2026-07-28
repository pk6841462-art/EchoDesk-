"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 7 - Real-Time Dashboard (Laptop Webcam + ESP32 Hardware Integration)
File:    Python/dashboard.py

Description:
Presents the real-time Streamlit web dashboard for EchoDesk featuring:
1. Live Laptop Camera capture with MediaPipe/OpenCV AI HUD overlay.
2. Real-time ESP32 hardware telemetry ingestion (Temp, Humidity, Light).
3. Trained Machine Learning predictions (Gradient Boosting Focus & Fatigue Scores).
4. Personal Cognitive Environment Model (PCEM) adaptation metrics.
5. Automated session logging and interactive telemetry charts.

Run command:
  streamlit run Python/dashboard.py
================================================================================
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Setup search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python.camera import Camera
from Python.posture import PostureEstimator
from Python.user_presence import UserPresenceDetector
from Python.serial_comm import SerialBridge
from Python.prediction import EchoDeskPredictor
from Python.pcem import PCEMEngine
from Python.recommendation import RecommendationEngine
from Python.dataset import DatasetCollector

# Streamlit Page Setup
st.set_page_config(
    page_title="EchoDesk - Real-Time AI Productivity Companion",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Professional UI Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .status-badge-hw {
        background-color: #1a4d2e; color: #4ef081; padding: 6px 14px;
        border-radius: 6px; font-size: 0.88rem; font-weight: 600; margin-bottom: 12px;
    }
    .status-badge-sim {
        background-color: #1a365d; color: #63b3ed; padding: 6px 14px;
        border-radius: 6px; font-size: 0.88rem; font-weight: 600; margin-bottom: 12px;
    }
    .recommendation-banner {
        background: linear-gradient(90deg, #1f6beb 0%, #114bb8 100%);
        border-radius: 8px; padding: 16px; color: #ffffff; font-size: 1.25rem;
        font-weight: bold; text-align: center; margin-top: 8px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(31, 107, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system_components(port_override: Optional[str] = None):
    """Initialize and cache core AI & hardware components."""
    camera = Camera(demo=False)
    posture_estimator = PostureEstimator()
    presence_detector = UserPresenceDetector()
    serial_bridge = SerialBridge(port=port_override)
    st.success("✅ SerialBridge object created")
    st.write(serial_bridge)
    st.write("Hardware connected:", serial_bridge.is_hardware_connected)
    st.write("Connection status:", serial_bridge.connection_status)
    st.write("Last error:", serial_bridge.last_error)
    st.write("Port:", serial_bridge.port)
    predictor = EchoDeskPredictor("ML/focus_model.pkl")
    pcem = PCEMEngine("pcem_profile.json")
    rec_engine = RecommendationEngine(pcem)
    collector = DatasetCollector("Dataset/study_dataset.csv")
    
    return {
        "camera": camera,
        "posture_estimator": posture_estimator,
        "presence_detector": presence_detector,
        "serial_bridge": serial_bridge,
        "predictor": predictor,
        "pcem": pcem,
        "rec_engine": rec_engine,
        "collector": collector,
        "start_time": time.time(),
        "owner_start_time": time.time(),
        "guest_start_time": time.time(),
        "last_mode": "owner",
        "session_active": True,
        "last_session_result": None,
    }


def load_dataset(csv_path: str = "Dataset/study_dataset.csv") -> pd.DataFrame:
    """Load session log dataset."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def main():
    # Sidebar Controls
    st.sidebar.title("⚡ EchoDesk Controls")
    st.sidebar.markdown("---")

    available_ports = SerialBridge.list_available_ports()
    port_options = ["Auto"] + available_ports
    selected_port = st.sidebar.selectbox(
        "ESP32 COM Port",
        options=port_options,
        index=0 if "Auto" in port_options else 0,
    )
    port_override = None if selected_port == "Auto" else selected_port

    components = init_system_components(port_override=port_override)
    camera: Camera = components["camera"]
    posture_estimator: PostureEstimator = components["posture_estimator"]
    presence_detector: UserPresenceDetector = components["presence_detector"]
    serial_bridge: SerialBridge = components["serial_bridge"]
    predictor: EchoDeskPredictor = components["predictor"]
    pcem: PCEMEngine = components["pcem"]
    rec_engine: RecommendationEngine = components["rec_engine"]
    collector: DatasetCollector = components["collector"]

    enable_webcam = st.sidebar.checkbox("📷 Enable Laptop Webcam Feed", value=True)
    auto_refresh = st.sidebar.checkbox("🔄 Real-Time Dashboard Loop", value=True)
    refresh_rate = st.sidebar.slider("Refresh Speed (sec)", 0.5, 5.0, 0.5, step=0.5)

    st.sidebar.markdown("### 👤 User Mode & Config")
    user_override = st.sidebar.selectbox("Study Mode", ["Owner", "Guest"], index=0)
    session_active = st.sidebar.toggle(
        "Study Session Active",
        value=components.get("session_active", True),
        help="Turn off to pause model updates and ESP32 feedback. Turn on again to start a new session from zero.",
    )
    posture_override = st.sidebar.selectbox("Posture Override", ["Auto-Detect (CV)", "Good", "Slight Lean", "Moderate Lean", "Poor"], index=0)

    st.sidebar.markdown("---")
    if st.sidebar.button("📊 Regenerate Calibrated Dataset"):
        collector.generate_synthetic_dataset(1200)
        st.sidebar.success("Calibrated dataset generated!")

    # App Header
    st.title("🖥️ EchoDesk – A Self-Learning Productivity Companion")

    # 1. Real-Time Physical Hardware Status Indicator
    is_hw = getattr(serial_bridge, "is_hardware_connected", False)
    port_name = getattr(serial_bridge, "port", "COM3")
    connection_status = getattr(serial_bridge, "connection_status", "disconnected")
    if is_hw:
        st.markdown(f'<div class="status-badge-hw">🟢 Physical ESP32 Hardware Connected ({port_name}) - Real-Time Serial Telemetry Active</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge-sim">⚠️ {port_name} Standby / Auto-Scan Mode: {connection_status}. System is streaming calibrated room metrics.</div>', unsafe_allow_html=True)
    if not is_hw:
        st.caption(f"ESP32 connection status: {connection_status}")

    # 2. Capture Laptop Camera Frame & Execute Computer Vision Pipeline
    frame = camera.read_frame()
    
    if frame is not None and enable_webcam:
        # Detect presence
        user_present = presence_detector.detect(frame)
        presence_int = 1 if user_present else 0

        # Identity is selected manually from the sidebar.
        identity = user_override

        # Detect posture
        if posture_override == "Auto-Detect (CV)":
            posture = posture_estimator.estimate(frame) if user_present else "Good"
        else:
            posture = posture_override
    else:
        user_present = True
        presence_int = 1
        identity = user_override
        posture = "Good" if posture_override == "Auto-Detect (CV)" else posture_override

    # 3. Read Physical ESP32 Telemetry from Serial Port
    sensor_data = serial_bridge.read_sensor_frame()
    temp = sensor_data.get("temperature", 0.0)
    hum = sensor_data.get("humidity", 0.0)
    light = sensor_data.get("light", 0.0)

    current_mode = "owner" if identity == "Owner" else "guest"
    if components.get("last_mode") != current_mode:
        components["last_mode"] = current_mode
        if current_mode == "owner":
            components["owner_start_time"] = time.time()
        else:
            components["guest_start_time"] = time.time()

    # Switching the session control on starts a new session for the selected mode.
    if session_active != components.get("session_active", True):
        components["session_active"] = session_active
        if session_active:
            if current_mode == "owner":
                components["owner_start_time"] = time.time()
            else:
                components["guest_start_time"] = time.time()

    if current_mode == "owner":
        session_start_time = components.get("owner_start_time", time.time())
    else:
        session_start_time = components.get("guest_start_time", time.time())

    # A new Owner/Guest mode selection begins a new session at zero minutes.
    # An ended session remains at zero until the session control is turned on again.
    study_duration = (
        round((time.time() - session_start_time) / 60.0, 1)
        if session_active else 0.0
    )

    st.caption(f"Live telemetry from ESP32 • Temperature: {temp:.1f} °C • Humidity: {hum:.1f} % • Light: {light:.0f} Lux")

    # Do not pass a zero-minute paused session to the model. Retain the last
    # valid result so the dashboard remains informative while work is paused.
    if session_active:
        payload = {
            "temperature": temp,
            "humidity": hum,
            "light": light,
            "study_duration": study_duration,
            "posture": posture,
            "user_presence": presence_int,
            "identity": identity,
            "owner_mode": 1 if identity == "Owner" else 0,
        }
        ml_res = predictor.predict(payload)
        focus_score = ml_res["focus_score"]
        fatigue_score = ml_res["fatigue_score"]
        prod_class = ml_res["productivity_class"]
        adaptation_score = pcem.calculate_adaptation_score(
            identity=identity,
            temperature=temp,
            humidity=hum,
            light=light,
            study_duration=study_duration,
        )
        rec_text, alert_code = rec_engine.generate_recommendation(
            focus_score=focus_score,
            fatigue_score=fatigue_score,
            posture=posture,
            temperature=temp,
            light=light,
            study_duration=study_duration,
            user_presence=presence_int,
        )
        components["last_session_result"] = {
            "focus_score": focus_score,
            "fatigue_score": fatigue_score,
            "productivity_class": prod_class,
            "adaptation_score": adaptation_score,
            "recommendation": rec_text,
            "alert_code": alert_code,
        }
        serial_bridge.send_ai_feedback(
            recommendation=rec_text,
            focus_score=int(focus_score),
            identity=identity,
            alert_code=alert_code,
        )
    else:
        previous_result = components.get("last_session_result")
        if previous_result is None:
            focus_score = fatigue_score = adaptation_score = 0.0
            prod_class = "Paused"
            rec_text, alert_code = "No active study session.", "PAUSED"
        else:
            focus_score = previous_result["focus_score"]
            fatigue_score = previous_result["fatigue_score"]
            prod_class = previous_result["productivity_class"]
            adaptation_score = previous_result["adaptation_score"]
            rec_text, alert_code = previous_result["recommendation"], "PAUSED"

    # Render the live result or a paused-session status.
    if not session_active:
        st.info("Study session paused: model updates and ESP32 feedback are stopped. Displayed scores are the last active-session results.")
    st.markdown(f"""
    <div class="recommendation-banner">
        💡 AI Recommendation: {rec_text} &nbsp;|&nbsp; Alert Code: {alert_code}
    </div>
    """, unsafe_allow_html=True)

    # Layout: Top Row - Laptop Webcam Feed & Live Telemetry Metrics
    cam_col, metric_col = st.columns([1.2, 1.0])

    with cam_col:
        st.subheader("📷 Laptop Webcam Live Stream & AI HUD")
        if frame is not None and enable_webcam:
            # Render HUD overlay on video frame
            hud_context = {
                "identity": identity,
                "user_present": user_present,
                "posture": posture,
                "temperature": temp,
                "humidity": hum,
                "light": light
            }
            # Draw overlay metrics
            h, w = frame.shape[:2]
            overlay_frame = frame.copy()
            cv2.rectangle(overlay_frame, (10, 10), (w - 10, 90), (15, 15, 20), -1)
            cv2.putText(overlay_frame, f"User: {identity} | Posture: {posture}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay_frame, f"Focus: {int(focus_score)}% | Fatigue: {int(fatigue_score)}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            
            # Convert BGR to RGB for Streamlit rendering
            rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, channels="RGB", use_container_width=True)
        else:
            st.info("Laptop webcam disabled or standby mode. Toggle 'Enable Laptop Webcam Feed' in sidebar to start stream.")

    with metric_col:
        st.subheader("🌡️ ESP32 Sensor Telemetry")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.metric("Temperature", f"{temp:.1f} °C")
            st.metric("Ambient Light", f"{light:.0f} Lux")
            st.metric("Posture State", posture)
        with tcol2:
            st.metric("Humidity", f"{hum:.1f} %")
            st.metric("Identity Mode", identity)
            st.metric("Study Duration", f"{int(study_duration)} min")

    st.markdown("---")

    # ML Predictions & PCEM Adaptation Metrics Row
    st.subheader("🎯 Trained ML Model Predictions & PCEM Cognitive Affinity")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("Focus Score (ML)", f"{focus_score:.1f}%")
        st.progress(int(focus_score) / 100.0)
    with pcol2:
        st.metric("Fatigue Index", f"{fatigue_score:.1f}%")
        st.progress(int(fatigue_score) / 100.0)
    with pcol3:
        st.metric("Productivity Class", prod_class)
    with pcol4:
        st.metric("PCEM Adaptation Affinity", f"{adaptation_score:.1f}%")
        st.progress(int(adaptation_score) / 100.0)

    st.markdown("---")

    # Session Trends Analytics & Dataset Logs
    st.subheader("📊 Historical Analytics & Telemetry Logs")
    df = load_dataset()

    if not df.empty:
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            fig1, ax1 = plt.subplots(figsize=(6, 3.2))
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#141721')
            ax1.tick_params(colors='white')
            ax1.xaxis.label.set_color('white')
            ax1.yaxis.label.set_color('white')
            ax1.title.set_color('white')

            sns.lineplot(data=df.tail(60), x=df.tail(60).index, y="temperature", color="#58a6ff", ax=ax1, label="Temp (°C)")
            sns.lineplot(data=df.tail(60), x=df.tail(60).index, y="humidity", color="#00d4b1", ax=ax1, label="Humidity (%)")
            ax1.set_title("Temperature & Humidity vs Session Time")
            ax1.legend(facecolor='#1e222d', edgecolor='none', labelcolor='white')
            st.pyplot(fig1)

        with gcol2:
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#141721')
            ax2.tick_params(colors='white')
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
            ax2.title.set_color('white')

            sns.lineplot(data=df.tail(60), x=df.tail(60).index, y="focus_score", color="#00ff7f", ax=ax2, label="Focus Score")
            sns.lineplot(data=df.tail(60), x=df.tail(60).index, y="fatigue_score", color="#ff7f50", ax=ax2, label="Fatigue Index")
            ax2.set_title("ML Focus & Fatigue Score Trends")
            ax2.legend(facecolor='#1e222d', edgecolor='none', labelcolor='white')
            st.pyplot(fig2)

        st.subheader("📜 Recent Study Session History Logs")
        st.dataframe(df.tail(15), use_container_width=True)

    # Streamlit 1.60 has no built-in autorefresh API.  Rerun at the selected
    # interval so the dashboard stays live without delaying serial reads.
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
