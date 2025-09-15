// file: lora_gateway_esp32.ino
// Board: ESP32 + RFM95 (LoRa), uses WiFi to call inference server

#include <WiFi.h>
#include <HTTPClient.h>
#include <SPI.h>
#include <RH_RF95.h>

#define RFM95_CS 5
#define RFM95_RST 14
#define RFM95_INT 26

RH_RF95 rf95(RFM95_CS, RFM95_INT);

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASS";
const char* PREDICT_URL = "http://<server-ip>:8080/predict"; // FastAPI endpoint

// Pump control pin
#define PUMP_PIN 27

// Simple weather placeholders; replace with real sensors/APIs
float wind_speed = 2.0;
float solar_rad = 400.0;
float pressure = 1013.0;
float et0_mm = 4.0;            // compute locally or from API
float forecast_rain_mm = 0.5;  // from weather API
float last_irrig_mm = 0.0;

void setup() {
  Serial.begin(115200);
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);

  if (!rf95.init()) {
    Serial.println("RFM95 init failed");
    while(1);
  }
  rf95.setFrequency(433.0);
  rf95.setTxPower(13, false);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected");
}

bool callPredict(float soil, float t, float h, float& out_mm) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(PREDICT_URL);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"soil_moisture\":" + String(soil,1) + ",";
  payload += "\"air_temp\":" + String(t,1) + ",";
  payload += "\"air_humidity\":" + String(h,1) + ",";
  payload += "\"wind_speed\":" + String(wind_speed,1) + ",";
  payload += "\"solar_rad\":" + String(solar_rad,1) + ",";
  payload += "\"pressure\":" + String(pressure,1) + ",";
  payload += "\"et0_mm\":" + String(et0_mm,1) + ",";
  payload += "\"forecast_rain_mm\":" + String(forecast_rain_mm,1) + ",";
  payload += "\"last_irrig_mm\":" + String(last_irrig_mm,1);
  payload += "}";

  int code = http.POST(payload);
  if (code > 0) {
    String resp = http.getString();
    int pos = resp.indexOf(":");
    int end = resp.indexOf("}");
    if (pos > 0 && end > pos) {
      String val = resp.substring(pos+1, end);
      out_mm = val.toFloat();
      http.end();
      return true;
    }
  }
  http.end();
  return false;
}

void irrigateByVolume(float mm) {
  // Example: convert mm over area to pump runtime
  // Assume 1 mm over 1 m^2 = 1 liter. If zone area A m^2 and pump flow F L/min:
  const float area_m2 = 10.0;
  const float flow_Lpm = 8.0;
  float liters = mm * area_m2;
  float minutes = liters / flow_Lpm;
  unsigned long ms = (unsigned long)(minutes * 60000.0);
  Serial.printf("Irrigating %.1f mm -> %.1f L -> %.1f min\n", mm, liters, minutes);
  digitalWrite(PUMP_PIN, HIGH);
  delay(ms);
  digitalWrite(PUMP_PIN, LOW);
  last_irrig_mm = mm;
}

void loop() {
  if (rf95.available()) {
    uint8_t buf; uint8_t len = sizeof(buf);
    if (rf95.recv(buf, &len)) {
      String s = "";
      for (int i=0;i<len;i++) s += (char)buf[i];
      Serial.print("RX: "); Serial.println(s);

      // very light JSON parse
      float soil=0, t=0, h=0;
      int a = s.indexOf("\"soil\":"); if (a>=0) soil = s.substring(a+7).toFloat();
      a = s.indexOf("\"temp\":"); if (a>=0) t = s.substring(a+7).toFloat();
      a = s.indexOf("\"hum\":"); if (a>=0) h = s.substring(a+6).toFloat();

      float mm;
      if (callPredict(soil, t, h, mm)) {
        Serial.printf("Prescribed irrigation: %.2f mm\n", mm);
        if (mm > 0.5) { // deadband
          irrigateByVolume(mm);
        } else {
          Serial.println("No irrigation needed");
        }
      } else {
        Serial.println("Prediction call failed");
      }
    }
  }
  delay(100);
}
