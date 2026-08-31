/*
  model_deploy.ino (FINAL)
  ------------------------
  ESP32 ASL glove inference + DFPlayer audio.

  Modes:
    - LETTER mode: plays 0001..0026 (A..Z)
    - WORD mode  : plays 0101..0126 (A..Z words)

  Required in this folder:
    - model_data.h
    - calibration.h

  Arduino libraries:
    - MicroTFLite
    - DFRobotDFPlayerMini
*/

#include <Arduino.h>
#include <Wire.h>
#include <HardwareSerial.h>
#include <DFRobotDFPlayerMini.h>

#include <MicroTFLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "calibration.h"

// ---------------- Pins ----------------
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

#define MPU_ADDR 0x68
#define I2C_SDA 21
#define I2C_SCL 22

#define DFPLAYER_RX 16   // ESP32 RX2  <- DFPlayer TX
#define DFPLAYER_TX 17   // ESP32 TX2  -> DFPlayer RX

#define MODE_BUTTON_PIN 12 // button to GND, INPUT_PULLUP

// ---------------- Runtime config ----------------
const uint32_t INFER_INTERVAL_MS = 40;      // 25Hz
const uint32_t AUDIO_COOLDOWN_MS = 800;     // faster feedback
const uint32_t BUTTON_DEBOUNCE_MS = 250;

const int STABLE_REQUIRED = 2;              // easier trigger for live testing
const float MIN_CONFIDENCE = 0.20f;         // relaxed threshold

const int LETTER_TRACK_BASE = 1;            // A=1 ... Z=26
const int WORD_TRACK_BASE   = 101;          // A=101 ... Z=126

const char* LABELS[26] = {
  "A","B","C","D","E","F","G","H","I","J","K","L","M",
  "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
};

// ---------------- TFLite ----------------
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Increase arena if AllocateTensors fails.
  constexpr int kTensorArenaSize = 40 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// ---------------- Devices/state ----------------
HardwareSerial dfSerial(2);
DFRobotDFPlayerMini dfplayer;

bool dfReady = false;
bool wordMode = false;

uint32_t lastInferMs = 0;
uint32_t lastAudioMs = 0;
uint32_t lastButtonMs = 0;

int lastPred = -1;
int stableCount = 0;

int16_t ax, ay, az, gx, gy, gz;

// ---------------- Helpers ----------------
float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

float normalizeFlex(float raw, float flat, float fist) {
  if (fist == flat) return 0.5f;
  return clamp01((raw - flat) / (fist - flat));
}

int8_t quantizeToInt8(float x, float scale, int zero_point) {
  int32_t q = (int32_t)roundf(x / scale) + zero_point;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

float dequantizeInt8(int8_t q, float scale, int zero_point) {
  return ((int)q - zero_point) * scale;
}

// ---------------- MPU6050 ----------------
void mpuInit() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0x00); // wake
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C); Wire.write(0x00); // accel +/-2g
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B); Wire.write(0x00); // gyro +/-250 dps
  Wire.endTransmission(true);
}

void mpuRead() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  ax = Wire.read() << 8 | Wire.read();
  ay = Wire.read() << 8 | Wire.read();
  az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read(); // temp
  gx = Wire.read() << 8 | Wire.read();
  gy = Wire.read() << 8 | Wire.read();
  gz = Wire.read() << 8 | Wire.read();
}

void setupSensors() {
  pinMode(TOUCH_INDEX, INPUT_PULLUP);
  pinMode(TOUCH_MIDDLE, INPUT_PULLUP);
  pinMode(TOUCH_RING, INPUT_PULLUP);
  pinMode(TOUCH_PINKY, INPUT_PULLUP);
  pinMode(TOUCH_R, INPUT_PULLUP);
  pinMode(TOUCH_U, INPUT_PULLUP);
  pinMode(MODE_BUTTON_PIN, INPUT_PULLUP);

  Wire.begin(I2C_SDA, I2C_SCL);
  mpuInit();
}

// ---------------- DFPlayer ----------------
bool setupDFPlayer() {
  dfSerial.begin(9600, SERIAL_8N1, DFPLAYER_RX, DFPLAYER_TX);
  if (!dfplayer.begin(dfSerial)) return false;
  dfplayer.volume(25); // 0..30
  return true;
}

void playTrackForClass(int cls) {
  if (!dfReady || cls < 0 || cls > 25) return;

  int track = wordMode ? (WORD_TRACK_BASE + cls) : (LETTER_TRACK_BASE + cls);
  dfplayer.playMp3Folder(track);

  Serial.print(wordMode ? "WORD " : "LETTER ");
  Serial.print(LABELS[cls]);
  Serial.print("  track=");
  Serial.println(track);
}

// ---------------- Model ----------------
bool setupModel() {
  model = tflite::GetModel(asl_model_int8_tflite);
  if (!model) return false;
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;

  static tflite::MicroMutableOpResolver<3> resolver;
  static bool resolverReady = false;
  if (!resolverReady) {
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolverReady = true;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) return false;

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (!input || !output) return false;

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) return false;
  if (input->dims->size != 2 || input->dims->data[1] != 17) return false;

  return true;
}

void fillModelInput() {
  float flex_raw[5] = {
    (float)analogRead(FLEX_PINKY),
    (float)analogRead(FLEX_RING),
    (float)analogRead(FLEX_MIDDLE),
    (float)analogRead(FLEX_INDEX),
    (float)analogRead(FLEX_THUMB)
  };

  float touch[6] = {
    (float)digitalRead(TOUCH_INDEX),
    (float)digitalRead(TOUCH_MIDDLE),
    (float)digitalRead(TOUCH_RING),
    (float)digitalRead(TOUCH_PINKY),
    (float)digitalRead(TOUCH_R),
    (float)digitalRead(TOUCH_U)
  };

  mpuRead();

  float feat[17];
  feat[0] = normalizeFlex(flex_raw[0], flex_flat[0], flex_fist[0]);
  feat[1] = normalizeFlex(flex_raw[1], flex_flat[1], flex_fist[1]);
  feat[2] = normalizeFlex(flex_raw[2], flex_flat[2], flex_fist[2]);
  feat[3] = normalizeFlex(flex_raw[3], flex_flat[3], flex_fist[3]);
  feat[4] = normalizeFlex(flex_raw[4], flex_flat[4], flex_fist[4]);

  feat[5] = touch[0];
  feat[6] = touch[1];
  feat[7] = touch[2];
  feat[8] = touch[3];
  feat[9] = touch[4];
  feat[10] = touch[5];

  feat[11] = ax / 32768.0f;
  feat[12] = ay / 32768.0f;
  feat[13] = az / 32768.0f;
  feat[14] = gx / 32768.0f;
  feat[15] = gy / 32768.0f;
  feat[16] = gz / 32768.0f;

  const float inScale = input->params.scale;
  const int inZp = input->params.zero_point;
  for (int i = 0; i < 17; i++) {
    input->data.int8[i] = quantizeToInt8(feat[i], inScale, inZp);
  }
}

void inferAndSpeak() {
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  int bestIdx = 0;
  int8_t bestRaw = output->data.int8[0];
  for (int i = 1; i < 26; i++) {
    int8_t v = output->data.int8[i];
    if (v > bestRaw) {
      bestRaw = v;
      bestIdx = i;
    }
  }

  float conf = dequantizeInt8(bestRaw, output->params.scale, output->params.zero_point);

  // Live monitor output so you can verify model is actually running.
  Serial.print("Pred=");
  Serial.print(LABELS[bestIdx]);
  Serial.print(" conf=");
  Serial.print(conf, 3);
  Serial.print(" mode=");
  Serial.println(wordMode ? "WORD" : "LETTER");

  if (bestIdx == lastPred) stableCount++;
  else {
    lastPred = bestIdx;
    stableCount = 1;
  }

  uint32_t now = millis();
  if (stableCount >= STABLE_REQUIRED && conf >= MIN_CONFIDENCE && (now - lastAudioMs) > AUDIO_COOLDOWN_MS) {
    playTrackForClass(bestIdx);
    lastAudioMs = now;
    stableCount = 0;
  }
}

void handleModeButton() {
  if (digitalRead(MODE_BUTTON_PIN) == LOW) {
    uint32_t now = millis();
    if (now - lastButtonMs > BUTTON_DEBOUNCE_MS) {
      lastButtonMs = now;
      wordMode = !wordMode;
      Serial.println(wordMode ? "Mode: WORD" : "Mode: LETTER");
      delay(60);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  setupSensors();
  dfReady = setupDFPlayer();
  bool modelReady = setupModel();

  Serial.println("ASL glove start");
  Serial.println(dfReady ? "DFPlayer OK" : "DFPlayer FAIL");
  Serial.println(modelReady ? "Model OK" : "Model FAIL");

  if (!modelReady) {
    while (true) {
      delay(1000);
    }
  }

}

void loop() {
  handleModeButton();

  uint32_t now = millis();
  if (now - lastInferMs >= INFER_INTERVAL_MS) {
    lastInferMs = now;
    fillModelInput();
    inferAndSpeak();
  }
}
