// file: lora_node_transmitter.ino
// Board: Arduino Pro Mini / Uno (5V), with RFM95 (SX1278), DHT22, analog soil sensor

#include <SPI.h>
#include <RH_RF95.h>
#include "DHT.h"

#define DHTPIN 2
#define DHTTYPE DHT22
#define SOIL_PIN A0

// RFM95 wiring (typical for Arduino)
#define RFM95_CS 10
#define RFM95_RST 9
#define RFM95_INT 2  // adjust if needed; if DHT on 2, use 3 for INT and move DHT
// Use consistent pins; example assumes INT on 3, DHT on 4 instead:
#undef DHTPIN
#define DHTPIN 4
#undef RFM95_INT
#define RFM95_INT 3

RH_RF95 rf95(RFM95_CS, RFM95_INT);
DHT dht(DHTPIN, DHTTYPE);

float analogToMoisturePct(int av) {
  // Simple calibration; replace with field calibration
  int dry = 800;   // value in air
  int wet = 300;   // value in water
  av = constrain(av, wet, dry);
  float pct = 100.0 * (float)(dry - av) / (float)(dry - wet);
  return pct;
}

void setup() {
  Serial.begin(9600);
  delay(100);
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH);
  dht.begin();

  // manual reset
  digitalWrite(RFM95_RST, LOW); delay(10);
  digitalWrite(RFM95_RST, HIGH); delay(10);

  if (!rf95.init()) {
    Serial.println("RFM95 init failed");
    while (1);
  }
  rf95.setFrequency(433.0); // or 868/915 based on region
  rf95.setTxPower(13, false);
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  int raw = analogRead(SOIL_PIN);
  float soil_pct = analogToMoisturePct(raw);

  // Build JSON-like payload (compact)
  char buf;
  // Only include key fields; gateway can augment weather/ET0
  snprintf(buf, sizeof(buf), "{\"soil\":%.1f,\"temp\":%.1f,\"hum\":%.1f}", soil_pct, t, h);

  rf95.send((uint8_t*)buf, strlen(buf));
  rf95.waitPacketSent();

  Serial.println(buf);
  delay(10000); // 10 seconds; increase for power savings
}
