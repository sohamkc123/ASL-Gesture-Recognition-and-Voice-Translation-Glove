/*
  model_deploy_rf.ino
  -------------------
  ESP32 ASL glove inference using Random Forest + DFPlayer playback.

  Modes:
    LETTER mode -> play 0001..0026 for A..Z
    WORD mode   -> play custom word tracks

  Required files:
    - rf_model.h
    - calibration.h

  IMPORTANT:
    DFPlayer should normally use:
      /mp3/0001.mp3
      /mp3/0002.mp3
      ...
      /mp3/0026.mp3

    Word tracks follow the numbers in WORD_TRACK_MAP.
*/

#include <Arduino.h>
#include <Wire.h>
#include <HardwareSerial.h>
#include <DFRobotDFPlayerMini.h>

#include "rf_model.h"
#include "calibration.h"


// ============================================================
// PINS
// ============================================================

#define FLEX_PINKY    33
#define FLEX_RING     32
#define FLEX_MIDDLE   35
#define FLEX_INDEX    34
#define FLEX_THUMB    36

#define TOUCH_INDEX   26
#define TOUCH_MIDDLE  23
#define TOUCH_RING    19
#define TOUCH_PINKY   18
#define TOUCH_R       25
#define TOUCH_U       27

#define MPU_ADDR      0x68
#define I2C_SDA       21
#define I2C_SCL       22

#define DFPLAYER_RX   16
#define DFPLAYER_TX   17

#define MODE_BUTTON_PIN 12


// ============================================================
// GENERAL TIMING
// ============================================================

const uint32_t INFER_INTERVAL_MS = 40;

// Gesture must remain stable this long before becoming valid.
const uint32_t GESTURE_HOLD_MS = 600;

// Button debounce.
const uint32_t BUTTON_DEBOUNCE_MS = 250;


// ============================================================
// RANDOM FOREST ACCEPTANCE
// ============================================================

const float MIN_CONFIDENCE = 0.12f;

// Kept low enough for C/Y/Z.
// Stability time is the main protection against transitions.
const float MIN_VOTE_MARGIN = 0.03f;


// ============================================================
// DFPPLAYER
// ============================================================

// DFPlayer software volume range is normally 0-30.
const uint8_t DFPLAYER_VOLUME = 30;

// Safety delay after a newly issued command.
const uint32_t DFPLAYER_COMMAND_GAP_MS = 300;


// ============================================================
// TRACK CONFIGURATION
// ============================================================

const int LETTER_TRACK_BASE = 1;

// A = 0
// B = 1
// ...
// Z = 25
//
// -1 means no word assigned.
const int WORD_TRACK_MAP[26] = {
  -1, 32, -1, 30, -1, 29, 37, 28, 35, -1, 27, -1, -1,
  -1, -1, -1, -1, 36, -1, 38, -1, 31, 34, -1, 33, -1
};


// ============================================================
// OBJECTS
// ============================================================

HardwareSerial dfSerial(2);
DFRobotDFPlayerMini dfplayer;


// ============================================================
// SYSTEM STATE
// ============================================================

bool dfReady = false;
bool wordMode = false;


// ============================================================
// TIMING
// ============================================================

uint32_t lastInferMs = 0;
uint32_t lastButtonMs = 0;
uint32_t lastDFPlayerCommandMs = 0;


// ============================================================
// GESTURE STATE
// ============================================================

int candidatePred = -1;
uint32_t candidateSinceMs = 0;


// This is the currently accepted/active stable gesture.
int activeGesture = -1;

// True after a gesture has become stable.
bool gestureAccepted = false;


// ============================================================
// AUDIO STATE
// ============================================================

// Track currently being played.
int currentPlayingTrack = -1;

// True when DFPlayer is expected to be playing.
bool audioPlaying = false;


// ============================================================
// BUTTON STATE
// ============================================================

bool lastButtonState = HIGH;


// ============================================================
// MPU6050
// ============================================================

int16_t ax, ay, az;
int16_t gx, gy, gz;


// ============================================================
// UTILITY FUNCTIONS
// ============================================================

float clamp01(float x) {

  if (x < 0.0f)
    return 0.0f;

  if (x > 1.0f)
    return 1.0f;

  return x;
}


float normalizeFlex(float raw, float flat, float fist) {

  if (fist == flat)
    return 0.5f;

  return clamp01(
    (raw - flat) / (fist - flat)
  );
}


// ============================================================
// MPU6050 INITIALIZATION
// ============================================================

void mpuInit() {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);


  // Accelerometer ±2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);


  // Gyroscope ±250 deg/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);
}


// ============================================================
// MPU6050 READ
// ============================================================

void mpuRead() {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, 14, true);


  if (Wire.available() < 14) {

    ax = 0;
    ay = 0;
    az = 0;

    gx = 0;
    gy = 0;
    gz = 0;

    return;
  }


  ax = Wire.read() << 8 | Wire.read();
  ay = Wire.read() << 8 | Wire.read();
  az = Wire.read() << 8 | Wire.read();


  // Temperature bytes
  Wire.read();
  Wire.read();


  gx = Wire.read() << 8 | Wire.read();
  gy = Wire.read() << 8 | Wire.read();
  gz = Wire.read() << 8 | Wire.read();
}


// ============================================================
// SENSOR SETUP
// ============================================================

void setupSensors() {

  pinMode(TOUCH_INDEX, INPUT_PULLUP);
  pinMode(TOUCH_MIDDLE, INPUT_PULLUP);
  pinMode(TOUCH_RING, INPUT_PULLUP);
  pinMode(TOUCH_PINKY, INPUT_PULLUP);
  pinMode(TOUCH_R, INPUT_PULLUP);
  pinMode(TOUCH_U, INPUT_PULLUP);

  pinMode(MODE_BUTTON_PIN, INPUT_PULLUP);

  lastButtonState = digitalRead(MODE_BUTTON_PIN);

  Wire.begin(I2C_SDA, I2C_SCL);

  mpuInit();
}


// ============================================================
// DFPPLAYER SETUP
// ============================================================

bool setupDFPlayer() {

  dfSerial.begin(
    9600,
    SERIAL_8N1,
    DFPLAYER_RX,
    DFPLAYER_TX
  );


  delay(300);


  if (!dfplayer.begin(dfSerial)) {

    Serial.println(
      "DFPlayer initialization FAILED"
    );

    return false;
  }


  Serial.println(
    "DFPlayer initialization OK"
  );


  // Maximum software volume.
  dfplayer.volume(DFPLAYER_VOLUME);


  // Give the module time to process startup commands.
  delay(300);


  return true;
}


// ============================================================
// DFPPLAYER FEEDBACK / ERROR HANDLING
// ============================================================

void handleDFPlayerMessages() {

  if (!dfReady)
    return;


  while (dfplayer.available()) {

    uint8_t type = dfplayer.readType();
    int value = dfplayer.read();


    switch (type) {

      case TimeOut:

        Serial.println(
          "DFPLAYER: TIMEOUT"
        );

        break;


      case WrongStack:

        Serial.println(
          "DFPLAYER: WRONG STACK"
        );

        break;


      case DFPlayerCardInserted:

        Serial.println(
          "DFPLAYER: SD CARD INSERTED"
        );

        break;


      case DFPlayerCardRemoved:

        Serial.println(
          "DFPLAYER: SD CARD REMOVED"
        );

        break;


      case DFPlayerCardOnline:

        Serial.println(
          "DFPLAYER: SD CARD ONLINE"
        );

        break;


      case DFPlayerPlayFinished:

        Serial.print(
          "DFPLAYER: PLAY FINISHED, track="
        );

        Serial.println(value);


        // Playback is finished.
        audioPlaying = false;

        currentPlayingTrack = -1;

        break;


      case DFPlayerError:

        Serial.print(
          "DFPLAYER ERROR: "
        );


        switch (value) {

          case Busy:

            Serial.println(
              "BUSY / CARD NOT FOUND"
            );

            break;


          case Sleeping:

            Serial.println(
              "SLEEPING"
            );

            break;


          case SerialWrongStack:

            Serial.println(
              "SERIAL WRONG STACK"
            );

            break;


          case CheckSumNotMatch:

            Serial.println(
              "CHECKSUM ERROR"
            );

            break;


          case FileIndexOut:

            Serial.println(
              "FILE INDEX OUT OF BOUNDS"
            );

            break;


          case FileMismatch:

            Serial.println(
              "FILE MISMATCH / FILE NOT FOUND"
            );

            break;


          case Advertise:

            Serial.println(
              "ADVERTISE MODE"
            );

            break;


          default:

            Serial.print(
              "UNKNOWN ERROR CODE="
            );

            Serial.println(value);

            break;
        }


        // We no longer consider the audio active after
        // receiving an error.
        audioPlaying = false;

        currentPlayingTrack = -1;

        break;


      default:

        break;
    }
  }
}


// ============================================================
// READ SENSOR FEATURES
// ============================================================

void readFeatures(float feat[17]) {

  float flex_raw[5] = {

    (float)analogRead(FLEX_PINKY),
    (float)analogRead(FLEX_RING),
    (float)analogRead(FLEX_MIDDLE),
    (float)analogRead(FLEX_INDEX),
    (float)analogRead(FLEX_THUMB)
  };


  // ----------------------------------------------------------
  // FLEX
  // ----------------------------------------------------------

  feat[0] =
    normalizeFlex(
      flex_raw[0],
      flex_flat[0],
      flex_fist[0]
    );


  feat[1] =
    normalizeFlex(
      flex_raw[1],
      flex_flat[1],
      flex_fist[1]
    );


  feat[2] =
    normalizeFlex(
      flex_raw[2],
      flex_flat[2],
      flex_fist[2]
    );


  feat[3] =
    normalizeFlex(
      flex_raw[3],
      flex_flat[3],
      flex_fist[3]
    );


  feat[4] =
    normalizeFlex(
      flex_raw[4],
      flex_flat[4],
      flex_fist[4]
    );


  // ----------------------------------------------------------
  // TOUCH
  // ----------------------------------------------------------

  feat[5]  = (float)digitalRead(TOUCH_INDEX);
  feat[6]  = (float)digitalRead(TOUCH_MIDDLE);
  feat[7]  = (float)digitalRead(TOUCH_RING);
  feat[8]  = (float)digitalRead(TOUCH_PINKY);
  feat[9]  = (float)digitalRead(TOUCH_R);
  feat[10] = (float)digitalRead(TOUCH_U);


  // ----------------------------------------------------------
  // MPU6050
  // ----------------------------------------------------------

  mpuRead();


  feat[11] = ax / 32768.0f;
  feat[12] = ay / 32768.0f;
  feat[13] = az / 32768.0f;

  feat[14] = gx / 32768.0f;
  feat[15] = gy / 32768.0f;
  feat[16] = gz / 32768.0f;
}


// ============================================================
// RANDOM FOREST TREE PREDICTION
// ============================================================

int predictTree(
  const RFNode* nodes,
  float feat[17]
) {

  int idx = 0;


  while (true) {

    const RFNode& n = nodes[idx];


    // Leaf node
    if (n.pred >= 0) {

      return n.pred;
    }


    // Safety check
    if (
      n.feature < 0 ||
      n.feature >= RF_NUM_FEATURES
    ) {

      return 0;
    }


    idx =
      (
        feat[n.feature] <= n.threshold
      )
      ? n.left
      : n.right;


    if (idx < 0) {

      return 0;
    }
  }
}


// ============================================================
// RANDOM FOREST PREDICTION
// ============================================================

int predictForest(
  float feat[17],
  float* confidence_out,
  float* margin_out
) {

  int votes[26];


  for (int i = 0; i < 26; i++) {

    votes[i] = 0;
  }


  // ----------------------------------------------------------
  // TREE VOTES
  // ----------------------------------------------------------

  for (
    int t = 0;
    t < RF_NUM_TREES;
    t++
  ) {

    int cls =
      predictTree(
        RF_TREES[t].nodes,
        feat
      );


    if (
      cls >= 0 &&
      cls < 26
    ) {

      votes[cls]++;
    }
  }


  // ----------------------------------------------------------
  // TOP TWO
  // ----------------------------------------------------------

  int bestCls = 0;
  int bestVotes = votes[0];
  int secondVotes = 0;


  for (
    int c = 1;
    c < 26;
    c++
  ) {

    int v = votes[c];


    if (v > bestVotes) {

      secondVotes = bestVotes;
      bestVotes = v;
      bestCls = c;

    } else if (v > secondVotes) {

      secondVotes = v;
    }
  }


  // ----------------------------------------------------------
  // CONFIDENCE
  // ----------------------------------------------------------

  if (confidence_out) {

    *confidence_out =
      (float)bestVotes /
      (float)RF_NUM_TREES;
  }


  // ----------------------------------------------------------
  // MARGIN
  // ----------------------------------------------------------

  if (margin_out) {

    *margin_out =
      (float)(
        bestVotes - secondVotes
      ) /
      (float)RF_NUM_TREES;
  }


  return bestCls;
}


// ============================================================
// CALCULATE TRACK
// ============================================================

int getTrackForClass(int cls) {

  if (
    cls < 0 ||
    cls > 25
  ) {

    return -1;
  }


  // WORD MODE
  if (wordMode) {

    return WORD_TRACK_MAP[cls];
  }


  // LETTER MODE
  return LETTER_TRACK_BASE + cls;
}


// ============================================================
// PLAY TRACK
// ============================================================

bool playTrackForClass(int cls) {

  if (!dfReady)
    return false;


  int track =
    getTrackForClass(cls);


  if (track < 1) {

    Serial.print(
      "No audio mapped for "
    );

    Serial.println(
      RF_LABELS[cls]
    );

    return false;
  }


  uint32_t now = millis();


  // ----------------------------------------------------------
  // COMMAND GAP
  // ----------------------------------------------------------

  if (
    now - lastDFPlayerCommandMs <
    DFPLAYER_COMMAND_GAP_MS
  ) {

    return false;
  }


  // ----------------------------------------------------------
  // DO NOT SEND ANOTHER COMMAND WHILE PLAYING
  // ----------------------------------------------------------

  if (audioPlaying) {

    return false;
  }


  // ----------------------------------------------------------
  // PLAY
  // ----------------------------------------------------------

  Serial.print(
    "PLAY "
  );

  Serial.print(
    wordMode
      ? "WORD "
      : "LETTER "
  );

  Serial.print(
    RF_LABELS[cls]
  );

  Serial.print(
    " -> track "
  );

  Serial.println(track);


  dfplayer.playMp3Folder(track);


  lastDFPlayerCommandMs = now;

  currentPlayingTrack = track;

  audioPlaying = true;


  return true;
}


// ============================================================
// MODE BUTTON
// ============================================================

void handleModeButton() {

  bool currentButtonState =
    digitalRead(MODE_BUTTON_PIN);


  // Detect new press: HIGH -> LOW
  if (
    lastButtonState == HIGH &&
    currentButtonState == LOW
  ) {

    uint32_t now = millis();


    if (
      now - lastButtonMs >=
      BUTTON_DEBOUNCE_MS
    ) {

      lastButtonMs = now;


      // Toggle mode
      wordMode = !wordMode;


      // Clear gesture detection
      candidatePred = -1;
      candidateSinceMs = 0;

      activeGesture = -1;
      gestureAccepted = false;


      // Stop the software's current audio state.
      audioPlaying = false;
      currentPlayingTrack = -1;


      Serial.println();
      Serial.println(
        "=============================="
      );


      if (wordMode) {

        Serial.println(
          "MODE CHANGED -> WORD"
        );

      } else {

        Serial.println(
          "MODE CHANGED -> LETTER"
        );
      }


      Serial.println(
        "=============================="
      );

      Serial.println();
    }
  }


  lastButtonState =
    currentButtonState;
}


// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(500);


  setupSensors();


  dfReady =
    setupDFPlayer();


  Serial.println();

  Serial.println(
    "================================"
  );

  Serial.println(
    "RF ASL GLOVE START"
  );

  Serial.println(
    "================================"
  );


  if (dfReady) {

    Serial.println(
      "DFPlayer: OK"
    );

  } else {

    Serial.println(
      "DFPlayer: FAIL"
    );
  }


  Serial.print(
    "Volume: "
  );

  Serial.println(
    DFPLAYER_VOLUME
  );


  Serial.print(
    "Gesture hold: "
  );

  Serial.print(
    GESTURE_HOLD_MS
  );

  Serial.println(
    " ms"
  );


  Serial.print(
    "Minimum confidence: "
  );

  Serial.println(
    MIN_CONFIDENCE,
    3
  );


  Serial.print(
    "Minimum vote margin: "
  );

  Serial.println(
    MIN_VOTE_MARGIN,
    3
  );


  Serial.println(
    "Mode: LETTER"
  );


  Serial.println();
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

  // ----------------------------------------------------------
  // ALWAYS PROCESS DFPPLAYER FEEDBACK
  // ----------------------------------------------------------

  handleDFPlayerMessages();


  // ----------------------------------------------------------
  // MODE BUTTON
  // ----------------------------------------------------------

  handleModeButton();


  uint32_t now = millis();


  // ----------------------------------------------------------
  // INFERENCE TIMER
  // ----------------------------------------------------------

  if (
    now - lastInferMs <
    INFER_INTERVAL_MS
  ) {

    return;
  }


  lastInferMs = now;


  // ----------------------------------------------------------
  // READ FEATURES
  // ----------------------------------------------------------

  float feat[17];

  readFeatures(feat);


  // ----------------------------------------------------------
  // PREDICTION
  // ----------------------------------------------------------

  float conf = 0.0f;
  float margin = 0.0f;


  int pred =
    predictForest(
      feat,
      &conf,
      &margin
    );


  // ----------------------------------------------------------
  // SERIAL DEBUG
  // ----------------------------------------------------------

  Serial.print(
    "Pred="
  );

  Serial.print(
    RF_LABELS[pred]
  );

  Serial.print(
    " conf="
  );

  Serial.print(
    conf,
    3
  );

  Serial.print(
    " margin="
  );

  Serial.println(
    margin,
    3
  );


  // ==========================================================
  // CONFIDENCE
  // ==========================================================

  bool conf_ok =
    (conf >= MIN_CONFIDENCE);


  bool margin_ok =
    (margin >= MIN_VOTE_MARGIN);


  /*
    OR is intentionally retained.

    This is important for C/Y/Z because they may have
    lower vote margins.

    The 600 ms hold requirement provides the transition
    protection.
  */

  bool decision_ok =
    conf_ok || margin_ok;


  // ==========================================================
  // UNCERTAIN PREDICTION
  // ==========================================================

  if (!decision_ok) {

    Serial.println(
      "Uncertain -> candidate reset"
    );


    candidatePred = -1;
    candidateSinceMs = 0;

    return;
  }


  // ==========================================================
  // NEW PREDICTION
  // ==========================================================

  if (
    pred != candidatePred
  ) {

    /*
      New gesture detected.

      DO NOT PLAY AUDIO YET.
    */

    candidatePred =
      pred;

    candidateSinceMs =
      now;


    Serial.print(
      "New candidate -> "
    );

    Serial.println(
      RF_LABELS[pred]
    );


    return;
  }


  // ==========================================================
  // SAME PREDICTION
  // ==========================================================

  uint32_t heldTime =
    now - candidateSinceMs;


  Serial.print(
    "Holding "
  );

  Serial.print(
    RF_LABELS[pred]
  );

  Serial.print(
    " for "
  );

  Serial.print(
    heldTime
  );

  Serial.println(
    " ms"
  );


  // ==========================================================
  // GESTURE STABLE
  // ==========================================================

  if (
    heldTime >=
    GESTURE_HOLD_MS
  ) {


    // --------------------------------------------------------
    // FIRST ACCEPT THE NEW GESTURE
    // --------------------------------------------------------

    if (
      !gestureAccepted ||
      activeGesture != pred
    ) {

      activeGesture =
        pred;

      gestureAccepted =
        true;


      Serial.print(
        "GESTURE ACCEPTED -> "
      );

      Serial.println(
        RF_LABELS[pred]
      );


      // New gesture is allowed to start audio.
      //
      // We do not play here if DFPlayer says previous audio
      // is still active. It will play when available.
    }


    // --------------------------------------------------------
    // AUDIO REPEAT
    // --------------------------------------------------------

    /*
      Same gesture remains stable.

      If the previous audio finished, play it again.

      This gives:
        A held -> A audio -> finish -> A audio -> finish...

      But:
        A -> transition -> B

      will NOT play B until B has been stable for 600 ms.
    */

    if (
      gestureAccepted &&
      activeGesture == pred &&
      !audioPlaying
    ) {

      playTrackForClass(pred);
    }
  }
}