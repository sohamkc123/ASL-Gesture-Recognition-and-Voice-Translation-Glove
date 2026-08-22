#define TOUCH_INDEX   26
#define TOUCH_MIDDLE  23
#define TOUCH_RING    18
#define TOUCH_PINKY    5
#define TOUCH_R       25
#define TOUCH_U       27

void setup() {
  Serial.begin(115200);

  pinMode(TOUCH_INDEX, INPUT_PULLUP);
  pinMode(TOUCH_MIDDLE, INPUT_PULLUP);
  pinMode(TOUCH_RING, INPUT_PULLUP);
  pinMode(TOUCH_PINKY, INPUT_PULLUP);
  pinMode(TOUCH_R, INPUT_PULLUP);
  pinMode(TOUCH_U, INPUT_PULLUP);

  Serial.println("Touch Test");
}

void loop() {

  Serial.print("Index: ");
  Serial.print(digitalRead(TOUCH_INDEX));

  Serial.print(" | Middle: ");
  Serial.print(digitalRead(TOUCH_MIDDLE));

  Serial.print(" | Ring: ");
  Serial.print(digitalRead(TOUCH_RING));

  Serial.print(" | Pinky: ");
  Serial.print(digitalRead(TOUCH_PINKY));

  Serial.print(" | R: ");
  Serial.print(digitalRead(TOUCH_R));

  Serial.print(" | U: ");
  Serial.println(digitalRead(TOUCH_U));

  delay(300);
}