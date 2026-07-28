/*
  ================================================================================
  Project: EchoDesk – A Self-Learning Productivity Companion
  Module:  Module 1 - Hardware Sensing & Actuation (ESP32 Firmware)
  File:    Arduino/EchoDesk_ESP32.ino
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
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define LDR_PIN    34
#define BUZZER_PIN 16

String currentIdentity = "Owner";
int currentFocusScore = 85;
String currentRecommendation = "Good Productivity";
float temperature = 24.5;
float humidity = 50.0;
int lightLux = 480;
unsigned long lastSensorRead = 0;
const unsigned long sensorInterval = 1000;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ECHODESK_BOOT");
  Serial.println("ESP32 ready");
  
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
  display.println("Calibrated System");
  display.setCursor(20, 48);
  display.println("Initializing...");
  display.display();

  dht.begin();
  delay(2000);
  Serial.println("DHT sensor initialized");
  Serial.print("DHT model: ");
  Serial.println(DHTTYPE == DHT22 ? "DHT22" : "OTHER");
  Serial.print("DHT pin: ");
  Serial.println(DHTPIN);
}

void loop() {
  if (millis() - lastSensorRead >= sensorInterval) {
    lastSensorRead = millis();

    float tempRead = dht.readTemperature(false);
    float humRead = dht.readHumidity();
    int rawLDR = analogRead(LDR_PIN);

    Serial.print("RAW_DHT temp=");
    Serial.print(tempRead, 2);
    Serial.print(" hum=");
    Serial.print(humRead, 2);
    if (isnan(tempRead) || isnan(humRead)) {
      Serial.print(" | STATUS=NO_DATA");
    } else {
      Serial.print(" | STATUS=OK");
    }
    Serial.println();

    if (!isnan(tempRead) && tempRead > 45.0) {
      tempRead = (tempRead - 32.0) * (5.0 / 9.0);
    }

    if (!isnan(tempRead) && tempRead >= -40.0 && tempRead <= 80.0) {
      temperature = tempRead;
    } else {
      Serial.println("DHT temp invalid, keeping previous value");
    }

    if (!isnan(humRead) && humRead >= 0.0 && humRead <= 100.0) {
      humidity = humRead;
    } else {
      Serial.println("DHT humidity invalid, keeping previous value");
    }

    float normalizedADC = (float)rawLDR / 4095.0;
    lightLux = (int)(150.0 + (1.0 - normalizedADC) * 700.0);
    if (lightLux < 100) lightLux = 100;
    if (lightLux > 900) lightLux = 900;

    Serial.print("TEMP:");
    Serial.print(temperature, 1);
    Serial.print(",HUM:");
    Serial.print(humidity, 1);
    Serial.print(",LIGHT:");
    Serial.println(lightLux);
    Serial.println("---");
    Serial.flush();
  }

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
