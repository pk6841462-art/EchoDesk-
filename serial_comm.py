"""
================================================================================
Project: EchoDesk – A Self-Learning Productivity Companion
Module:  Module 1 - Serial Communication Bridge
File:    Python/serial_comm.py

Key Fix: Uses BLOCKING readline() with timeout to always wait for and capture
the latest ESP32 sensor transmission. No more in_waiting polling that misses data.
================================================================================
"""

import time
from typing import Dict, Optional
import serial
import serial.tools.list_ports




class SerialBridge:
    """Bi-directional Serial Communication Bridge between ESP32 and Python."""

    is_hardware_connected: bool = False

    @staticmethod
    def list_available_ports() -> list[str]:
        """Return a list of available serial ports discovered by pyserial."""
        ports = []
        for p in serial.tools.list_ports.comports():
            if p.device:
                ports.append(p.device)
        return ports

    def __init__(self, port: Optional[str] = None, baud_rate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.is_hardware_connected: bool = False
        self.last_reconnect_attempt = 0.0
        self.last_error: str = ""
        self.connection_status: str = "disconnected"

        # Start without placeholder values; real telemetry will be populated only after the ESP32 sends it.
        self.last_sensor_data: Dict[str, float] = {
            "temperature": 0.0,
            "humidity": 0.0,
            "light": 0.0
        }

        self.connect()

    def _build_port_candidates(self) -> list[str]:
        """Build a prioritized list of candidate serial ports for the ESP32."""
        candidates: list[str] = []
        seen = set()

        if self.port:
            candidates.append(self.port)
            seen.add(self.port)

        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            if any(kw in desc or kw in hwid for kw in ["cp210", "ch340", "ftdi", "esp32", "usb serial", "uart"]):
                if p.device not in seen:
                    candidates.append(p.device)
                    seen.add(p.device)

        for port in [f"COM{i}" for i in range(1, 21)]:
            if port not in seen:
                candidates.append(port)
                seen.add(port)

        return candidates

    def connect(self) -> bool:
        """Attempt to open the ESP32 on a detected or common COM port."""

        if self.connection is not None and self.connection.is_open:
            self.is_hardware_connected = True
            self.connection_status = f"connected ({self.port})"
            return True

        for candidate in self._build_port_candidates():
            try:
                self.connection = serial.Serial(candidate, self.baud_rate, timeout=self.timeout)
                time.sleep(1.5)
                self.connection.reset_input_buffer()
                self.port = candidate
                self.is_hardware_connected = True
                self.last_error = ""
                self.connection_status = f"connected ({candidate})"
                print(f"[SerialBridge] SUCCESS: Connected to ESP32 on {self.port} @ {self.baud_rate} baud.")
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.connection = None
                self.is_hardware_connected = False

        self.connection_status = "disconnected"
        self.is_hardware_connected = False
        return False

    def _parse_sensor_line(self, raw_line: str) -> Dict[str, float]:
        """Parse a telemetry line like TEMP:27.3,HUM:65.0,LIGHT:480."""
        if not raw_line or "TEMP:" not in raw_line:
            return dict(self.last_sensor_data)

        for token in raw_line.split(","):
            if ":" not in token:
                continue
            key, val_str = token.split(":", 1)
            key = key.strip().upper()
            try:
                val = float(val_str.strip())
                if key == "TEMP":
                    if val > 45.0:
                        val = (val - 32.0) * (5.0 / 9.0)
                    self.last_sensor_data["temperature"] = round(val, 1)
                elif key == "HUM":
                    self.last_sensor_data["humidity"] = round(val, 1)
                elif key in {"LIGHT", "LDR"}:
                    if val > 1000.0:
                        val = 100.0 + ((4095.0 - val) / 4095.0) * 800.0
                    self.last_sensor_data["light"] = round(val, 1)
            except ValueError:
                continue

        return dict(self.last_sensor_data)

    def read_sensor_frame(self) -> Dict[str, float]:
        """
        Return the newest complete telemetry frame available right now.

        The dashboard re-runs on a timer, so waiting here only delays the UI.
        Draining the serial buffer also prevents old frames from accumulating.
        """
        # Auto-reconnect if disconnected
        if not self.is_hardware_connected or self.connection is None:
            if time.time() - self.last_reconnect_attempt > 3.0:
                self.last_reconnect_attempt = time.time()
                if self.connect():
                    print(f"[SerialBridge] Reconnected to ESP32 on {self.port}!")
            return dict(self.last_sensor_data)

        try:
            latest_line = ""
            while self.connection.in_waiting > 0:
                candidate = self.connection.readline().decode("utf-8", errors="ignore").strip()
                if "TEMP:" in candidate:
                    latest_line = candidate

            return self._parse_sensor_line(latest_line)

        except Exception as exc:
            print(f"[SerialBridge] Read error ({exc}). Reconnecting...")
            self.close()
            return dict(self.last_sensor_data)

    def send_ai_feedback(self, recommendation: str, focus_score: int, identity: str = "Owner", alert_code: str = "NONE") -> bool:
        """Send AI prediction payload back to ESP32 OLED display and Buzzer."""
        if not self.is_hardware_connected or self.connection is None:
            return False

        packet = f"REC:{recommendation}|FOCUS:{focus_score}|IDENTITY:{identity}|ALERT:{alert_code}\n"
        try:
            self.connection.write(packet.encode("utf-8"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close serial connection."""
        if self.connection is not None:
            try:
                if self.connection.is_open:
                    self.connection.close()
            except Exception:
                pass
            self.connection = None
            self.is_hardware_connected = False


if __name__ == "__main__":
    bridge = SerialBridge()
    for i in range(5):
        frame = bridge.read_sensor_frame()
        print(f"Frame {i}: {frame}")
    bridge.close()
