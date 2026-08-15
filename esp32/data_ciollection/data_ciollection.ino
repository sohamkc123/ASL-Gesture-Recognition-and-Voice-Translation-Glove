#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

// ================= FLEX SENSOR PINS =================

#define FLEX_PINKY   33
#define FLEX_RING    32
#define FLEX_MIDDLE  35
#define FLEX_INDEX   34
#define FLEX_THUMB   36   // VP / GPIO36


// ================= TOUCH SENSOR PINS =================

#define TOUCH_INDEX   26
#define TOUCH_MIDDLE  23
#define TOUCH_RING    18
#define TOUCH_PINKY    5
#define TOUCH_R       25
#define TOUCH_U       27


// ================= MPU6050 =================

#define SDA_PIN 21
#define SCL_PIN 22


// ================= MODE BUTTON =================

#define MODE_BUTTON 12


// true  = Alphabet mode
// false = Word mode

bool alphabetMode = true;

int lastButtonState = HIGH;


void setup() {

  Serial.begin(115200);

  // ==================================================
  // FLEX SENSORS
  // ==================================================

  pinMode(FLEX_PINKY, INPUT);
  pinMode(FLEX_RING, INPUT);
  pinMode(FLEX_MIDDLE, INPUT);
  pinMode(FLEX_INDEX, INPUT);
  pinMode(FLEX_THUMB, INPUT);


  // ==================================================
  // TOUCH SENSORS
  // INPUT_PULLUP prevents floating inputs
  // ==================================================

  pinMode(TOUCH_INDEX, INPUT_PULLUP);
  pinMode(TOUCH_MIDDLE, INPUT_PULLUP);
  pinMode(TOUCH_RING, INPUT_PULLUP);
  pinMode(TOUCH_PINKY, INPUT_PULLUP);
  pinMode(TOUCH_R, INPUT_PULLUP);
  pinMode(TOUCH_U, INPUT_PULLUP);


  // ==================================================
  // MODE BUTTON
  // ==================================================

  pinMode(MODE_BUTTON, INPUT_PULLUP);


  // ==================================================
  // MPU6050
  // ==================================================

  Wire.begin(SDA_PIN, SCL_PIN);

  mpu.initialize();

  if (mpu.testConnection()) {
    Serial.println("#MPU6050_CONNECTED");
  }
  else {
    Serial.println("#MPU6050_FAILED");
  }

  delay(1000);


  // ==================================================
  // INITIAL MODE
  // ==================================================

  Serial.println("#MODE=ALPHABET");


  // ==================================================
  // CSV HEADER
  // ==================================================

  Serial.println(
    "Mode,"
    "FlexPinky,FlexRing,FlexMiddle,FlexIndex,FlexThumb,"
    "TouchIndex,TouchMiddle,TouchRing,TouchPinky,TouchR,TouchU,"
    "AccX,AccY,AccZ,GyroX,GyroY,GyroZ"
  );
}


void loop() {

  // ==================================================
  // MODE BUTTON
  // ==================================================

  int buttonState = digitalRead(MODE_BUTTON);


  // Detect a new button press
  if (lastButtonState == HIGH && buttonState == LOW) {

    // Toggle mode
    alphabetMode = !alphabetMode;


    if (alphabetMode) {
      Serial.println("#MODE=ALPHABET");
    }
    else {
      Serial.println("#MODE=WORD");
    }


    // Button debounce
    delay(250);
  }


  lastButtonState = buttonState;


  // ==================================================
  // FLEX SENSOR READINGS
  // ==================================================

  int flexPinky  = analogRead(FLEX_PINKY);
  int flexRing   = analogRead(FLEX_RING);
  int flexMiddle = analogRead(FLEX_MIDDLE);
  int flexIndex  = analogRead(FLEX_INDEX);
  int flexThumb  = analogRead(FLEX_THUMB);


  // ==================================================
  // TOUCH SENSOR READINGS
  //
  // INPUT_PULLUP:
  // 1 = not touched
  // 0 = touched
  // ==================================================

  int touchIndex  = digitalRead(TOUCH_INDEX);
  int touchMiddle = digitalRead(TOUCH_MIDDLE);
  int touchRing   = digitalRead(TOUCH_RING);
  int touchPinky  = digitalRead(TOUCH_PINKY);
  int touchR      = digitalRead(TOUCH_R);
  int touchU      = digitalRead(TOUCH_U);


  // ==================================================
  // MPU6050
  // ==================================================

  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(
    &ax,
    &ay,
    &az,
    &gx,
    &gy,
    &gz
  );


  // ==================================================
  // PRINT MODE
  //
  // 1 = ALPHABET
  // 0 = WORD
  // ==================================================

  if (alphabetMode) {
    Serial.print("1,");
  }
  else {
    Serial.print("0,");
  }


  // ==================================================
  // PRINT FLEX
  // ==================================================

  Serial.print(flexPinky);
  Serial.print(",");

  Serial.print(flexRing);
  Serial.print(",");

  Serial.print(flexMiddle);
  Serial.print(",");

  Serial.print(flexIndex);
  Serial.print(",");

  Serial.print(flexThumb);
  Serial.print(",");


  // ==================================================
  // PRINT TOUCH
  // ==================================================

  Serial.print(touchIndex);
  Serial.print(",");

  Serial.print(touchMiddle);
  Serial.print(",");

  Serial.print(touchRing);
  Serial.print(",");

  Serial.print(touchPinky);
  Serial.print(",");

  Serial.print(touchR);
  Serial.print(",");

  Serial.print(touchU);
  Serial.print(",");


  // ==================================================
  // PRINT MPU6050
  // ==================================================

  Serial.print(ax);
  Serial.print(",");

  Serial.print(ay);
  Serial.print(",");

  Serial.print(az);
  Serial.print(",");

  Serial.print(gx);
  Serial.print(",");

  Serial.print(gy);
  Serial.print(",");

  Serial.println(gz);


  // ==================================================
  // 20 SAMPLES / SECOND
  // ==================================================

  delay(50);
}