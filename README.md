# EchoDesk

EchoDesk is a complete AI + IoT prototype for a self-learning productivity desk.
It combines:
- ESP32 hardware sensing with DHT11 and LDR
- Python-based computer vision and recommendation logic
- Serial communication between the laptop and the controller
- A simple PCEM model that learns from owner study sessions

## Project Structure
- main.py: orchestrates the whole system
- camera.py: webcam capture and visualization
- face_recognition.py: owner/guest recognition
- posture.py: posture estimation
- user_presence.py: presence detection
- pcem.py: personal cognitive environment model
- recommendation.py: recommendation engine
- serial_comm.py: serial bridge to the ESP32
- database.py: session logging placeholder
- Arduino/EchoDesk_ESP32.ino: ESP32 firmware
- faces/owner.jpg: reference owner image

## How to Run
1. Place an owner image at faces/owner.jpg.
2. Upload Arduino/EchoDesk_ESP32.ino to the ESP32.
3. Connect the ESP32 via USB.
4. Run python main.py.
