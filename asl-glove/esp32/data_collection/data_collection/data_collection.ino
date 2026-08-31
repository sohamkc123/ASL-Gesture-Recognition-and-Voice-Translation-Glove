/*
  data_collection.ino
  --------------------
  Streams raw sensor values over Serial as CSV. No labeling, no
  normalization here - that's all handled on the PC side so nothing
  about the model or calibration ever needs a reflash.

  Pins match your actual wired hardware (from your existing firmware):
    Flex (analog):
      FLEX_PINKY  33
      FLEX_RING   32
      FLEX_MIDDLE 35
      FLEX_INDEX  34
      FLEX_THUMB  36 (VP)
    Touch (INPUT_PULLUP, reads LOW=0 when touched):
      TOUCH_INDEX  26
      TOUCH_MIDDLE 23
      TOUCH_RING   18
      TOUCH_PINKY   5   <-- verify this contact with a multimeter/continuity
                            test before trusting any pinky-related letter.
                            In your last dataset this sensor NEVER registered
                            a touch across 5600+ samples.
      TOUCH_R      25
      TOUCH_U      27
    MPU6050 (I2C, manual register access - no external library, so there's
    no ambiguity about what full-scale range is active):
      SDA 21, SCL 22

  IMPORTANT: this sketch explicitly sets the MPU6050 to +/-2g accel and
  +/-250 deg/s gyro at startup (the sensor's power-on default, but we set
  it explicitly rather than relying on a library's default so the exact
  same setup can never silently differ between this sketch and
  model_deploy.ino).

  Output line format (CSV):
    flex_pinky,flex_ring,flex_middle,flex_index,flex_thumb,
    touch_index,touch_middle,touch_ring,touch_pinky,touch_r,touch_u,
    ax,ay,az,gx,gy,gz
*/

#include <Wire.h>

#define MPU_ADDR 0x68

#define FLEX_PINKY   33
#define FLEX_RING    32
#define FLEX_MIDDLE  35
#define FLEX_INDEX   34
#define FLEX_THUMB   36

#define TOUCH_INDEX   26
#define TOUCH_MIDDLE  23
#define TOUCH_RING    19
#define TOUCH_PINKY   18
#define TOUCH_R       25
#define TOUCH_U       27

const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz
unsigned long lastSample = 0;

int16_t ax, ay, az, gx, gy, gz;

void mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00); // PWR_MGMT_1: wake up
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x00); // ACCEL_CONFIG: +/-2g
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); Wire.write(0x00); // GYRO_CONFIG: +/-250 deg/s
  Wire.endTransmission(true);
}

void mpuRead() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  ax = Wire.read() << 8 | Wire.read();
  ay = Wire.read() << 8 | Wire.read();
  az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read(); // skip temperature
  gx = Wire.read() << 8 | Wire.read();
  gy = Wire.read() << 8 | Wire.read();
  gz = Wire.read() << 8 | Wire.read();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  mpuInit();

  pinMode(TOUCH_INDEX, INPUT_PULLUP);
  pinMode(TOUCH_MIDDLE, INPUT_PULLUP);
  pinMode(TOUCH_RING, INPUT_PULLUP);
  pinMode(TOUCH_PINKY, INPUT_PULLUP);
  pinMode(TOUCH_R, INPUT_PULLUP);
  pinMode(TOUCH_U, INPUT_PULLUP);

  delay(200);
  Serial.println("flex_pinky,flex_ring,flex_middle,flex_index,flex_thumb,touch_index,touch_middle,touch_ring,touch_pinky,touch_r,touch_u,ax,ay,az,gx,gy,gz");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSample >= SAMPLE_INTERVAL_MS) {
    lastSample = now;
    mpuRead();

    String line = "";
    line += String(analogRead(FLEX_PINKY));  line += ",";
    line += String(analogRead(FLEX_RING));   line += ",";
    line += String(analogRead(FLEX_MIDDLE)); line += ",";
    line += String(analogRead(FLEX_INDEX));  line += ",";
    line += String(analogRead(FLEX_THUMB));  line += ",";

    line += String(digitalRead(TOUCH_INDEX));  line += ",";
    line += String(digitalRead(TOUCH_MIDDLE)); line += ",";
    line += String(digitalRead(TOUCH_RING));   line += ",";
    line += String(digitalRead(TOUCH_PINKY));  line += ",";
    line += String(digitalRead(TOUCH_R));      line += ",";
    line += String(digitalRead(TOUCH_U));      line += ",";

    line += String(ax) + "," + String(ay) + "," + String(az) + "," +
            String(gx) + "," + String(gy) + "," + String(gz);

    Serial.println(line);
  }
}
