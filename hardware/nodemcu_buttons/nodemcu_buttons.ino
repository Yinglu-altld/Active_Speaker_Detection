#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* DEST_IP = "192.168.1.109";
const uint16_t DEST_PORT = 5005;

const uint8_t BUTTON0_PIN = D2;
const uint8_t BUTTON1_PIN = D6;
const bool ACTIVE_LOW = true;
const uint16_t DEBOUNCE_MS = 25;
const uint32_t SEND_INTERVAL_MS = 50;

struct DebouncedButton {
  uint8_t pin;
  int stableState;
  int rawState;
  uint32_t lastEdgeMs;
};

DebouncedButton button0 = {BUTTON0_PIN, HIGH, HIGH, 0};
DebouncedButton button1 = {BUTTON1_PIN, HIGH, HIGH, 0};

WiFiUDP udp;
uint32_t lastSendMs = 0;

int logicalPressed(const DebouncedButton& button) {
  if (ACTIVE_LOW) {
    return button.stableState == LOW ? 1 : 0;
  }
  return button.stableState == HIGH ? 1 : 0;
}

void initButton(DebouncedButton& button) {
  if (ACTIVE_LOW) {
    pinMode(button.pin, INPUT_PULLUP);
  } else {
    pinMode(button.pin, INPUT);
  }
  int value = digitalRead(button.pin);
  button.rawState = value;
  button.stableState = value;
  button.lastEdgeMs = millis();
}

bool updateButton(DebouncedButton& button, uint32_t nowMs) {
  bool changed = false;
  int raw = digitalRead(button.pin);
  if (raw != button.rawState) {
    button.rawState = raw;
    button.lastEdgeMs = nowMs;
  }
  if ((nowMs - button.lastEdgeMs) >= DEBOUNCE_MS && button.stableState != button.rawState) {
    button.stableState = button.rawState;
    changed = true;
  }
  return changed;
}

void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  uint32_t startMs = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startMs) < 12000) {
    delay(200);
  }
}

void sendState(uint32_t nowMs) {
  char payload[32];
  int b0 = logicalPressed(button0);
  int b1 = logicalPressed(button1);
  int len = snprintf(payload, sizeof(payload), "{\"u0\":%d,\"u1\":%d}", b0, b1);
  if (len <= 0) {
    return;
  }
  udp.beginPacket(DEST_IP, DEST_PORT);
  udp.write(reinterpret_cast<const uint8_t*>(payload), static_cast<size_t>(len));
  udp.endPacket();
  lastSendMs = nowMs;
}

void setup() {
  initButton(button0);
  initButton(button1);
  ensureWiFi();
  udp.begin(0);
  sendState(millis());
}

void loop() {
  ensureWiFi();
  uint32_t nowMs = millis();
  bool changed = updateButton(button0, nowMs) || updateButton(button1, nowMs);
  if (changed || (nowMs - lastSendMs) >= SEND_INTERVAL_MS) {
    sendState(nowMs);
  }
  delay(2);
}
