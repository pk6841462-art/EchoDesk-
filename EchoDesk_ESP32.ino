/*
  ================================================================================
  Project: EchoDesk – A Self-Learning Productivity Companion
  Module:  Module 1 - Hardware Sensing & Actuation (ESP32 Firmware)
  File:    Arduino/EchoDesk_ESP32/EchoDesk_ESP32.ino

  IMPORTANT: The values printed to Serial MUST be the SAME variables shown on the OLED.
  DHT11 requires: read humidity FIRST, then temperature (library reads both in one cycle).
  ================================================================================
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#define DHTPIN 27
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define LDR_PIN    34
#define BUZZER_PIN 25


String currentIdentity = "Owner";
int currentFocusScore = 85;
String currentRecommendation = "Good Productivity";

// Live sensor values (these are displayed on OLED AND sent via Serial)
float temperature = 0.0;
float humidity = 0.0;
int lightLux = 0;
bool sensorReady = false;

unsigned long lastSensorRead = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LDR_PIN, INPUT);
  digitalWrite(BUZZER_PIN, LOW);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("ERROR: OLED Display init failed!");
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(15, 10);
  display.println("=== ECHODESK ===");
  display.setCursor(10, 30);
  display.println("Real-Time Telemetry");
  display.setCursor(20, 48);
  display.println("Initializing...");
  display.display();

  dht.begin();
  delay(2000); // DHT11 needs 2s warm-up before first reliable read
}

void loop() {
  // Read sensors every 2000ms (DHT11 minimum reliable interval is ~2s)
  if (millis() - lastSensorRead >= 2000) {
    lastSensorRead = millis();

    // IMPORTANT: Read humidity FIRST - DHT library triggers full sensor read on readHumidity()
    float humRead = dht.readHumidity();
    float tempRead = dht.readTemperature(false); // false = Celsius
    int rawLDR = analogRead(LDR_PIN);

    // Update temperature if valid
    if (!isnan(tempRead)) {
      // Auto-convert if sensor returns Fahrenheit (> 45)
      if (tempRead > 45.0) {
        tempRead = (tempRead - 32.0) * (5.0 / 9.0);
      }
      temperature = tempRead;
      sensorReady = true;
    }

    // Update humidity if valid
    if (!isnan(humRead)) {
      humidity = humRead;
      sensorReady = true;
    }

    // Convert 12-bit ADC (0-4095) to Lux
    float normalizedADC = (float)rawLDR / 4095.0;
    lightLux = (int)(100.0 + (1.0 - normalizedADC) * 800.0);
    if (lightLux < 50) lightLux = 50;
    if (lightLux > 1000) lightLux = 1000;

    // Transmit the EXACT same values that are shown on the OLED display
    Serial.print("TEMP:");
    Serial.print(temperature, 1);
    Serial.print(",HUM:");
    Serial.print(humidity, 1);
    Serial.print(",LIGHT:");
    Serial.println(lightLux);
  }

  // Read AI feedback from Python dashboard via Serial
  if (Serial.available() > 0) {
    String serialPacket = Serial.readStringUntil('\n');
    serialPacket.trim();
    if (serialPacket.length() > 0) {
      parseSerialPacket(serialPacket);
    }
  }

  updateOLEDDisplay();
}

void parseSerialPacket(String packet) {
  int recIdx = packet.indexOf("REC:");
  int focusIdx = packet.indexOf("|FOCUS:");
  int idIdx = packet.indexOf("|IDENTITY:");
  int alertIdx = packet.indexOf("|ALERT:");

  if (recIdx != -1 && focusIdx != -1) {
    currentRecommendation = packet.substring(recIdx + 4, focusIdx);
    
    if (idIdx != -1) {
      currentFocusScore = packet.substring(focusIdx + 7, idIdx).toInt();
      if (alertIdx != -1) {
        currentIdentity = packet.substring(idIdx + 10, alertIdx);
        String alertCode = packet.substring(alertIdx + 7);
        triggerBuzzerAlert(alertCode);
      } else {
        currentIdentity = packet.substring(idIdx + 10);
      }
    } else {
      currentFocusScore = packet.substring(focusIdx + 7).toInt();
    }
  }
}

void triggerBuzzerAlert(String alertCode) {
  if (alertCode == "BAD_POSTURE") {
    digitalWrite(BUZZER_PIN, HIGH); delay(100); digitalWrite(BUZZER_PIN, LOW); delay(80);
    digitalWrite(BUZZER_PIN, HIGH); delay(100); digitalWrite(BUZZER_PIN, LOW);
  } else if (alertCode == "HIGH_FATIGUE" || alertCode == "TAKE_BREAK") {
    digitalWrite(BUZZER_PIN, HIGH); delay(400); digitalWrite(BUZZER_PIN, LOW);
  }
}

void updateOLEDDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("User: ");
  display.print(currentIdentity);

  display.setCursor(75, 0);
  display.print("F:");
  display.print(currentFocusScore);
  display.print("%");

  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  display.setCursor(0, 14);
  display.print("Rec: ");
  display.setCursor(0, 26);
  display.print(currentRecommendation);

  display.drawLine(0, 38, 128, 38, SSD1306_WHITE);

  // Display the SAME temperature/humidity/light variables sent over Serial
  display.setCursor(0, 42);
  display.print("T:");
  display.print(temperature, 1);
  display.print("C  H:");
  display.print(humidity, 0);
  display.print("%");

  display.setCursor(0, 54);
  display.print("Light: ");
  display.print(lightLux);
  display.print(" Lux");

  display.display();
}
