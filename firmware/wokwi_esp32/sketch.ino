#include <Arduino.h>
#include <DHT.h>
#include <math.h>

#define DHT_PIN 15
#define DHT_TYPE DHT22
#define POT_PIN 34

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  analogReadResolution(12);


  Serial.println(
    "time_ms,temperature_c,humidity_pct,adc_raw,voltage_v"
  );
}

void loop() {
  unsigned long timestamp = millis();

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int adcValue = analogRead(POT_PIN);

  float voltage = adcValue * 3.3f / 4095.0f;


  if (isnan(temperature) || isnan(humidity)) {
    delay(2000);
    return;
  }


  Serial.print(timestamp);
  Serial.print(",");

  Serial.print(temperature, 2);
  Serial.print(",");

  Serial.print(humidity, 2);
  Serial.print(",");

  Serial.print(adcValue);
  Serial.print(",");

  Serial.println(voltage, 3);

  delay(2000);
}
