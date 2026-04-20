/*
hardware/traffic.ino

Receives action (0-3) from serial and controls 4 lanes of LEDs.
Each lane has RED, YELLOW, GREEN LED pins.
Only one lane is green at a time.
*/

const int LANE_COUNT = 4;

// Pin mapping: lanePins[lane][0=RED, 1=YELLOW, 2=GREEN]
const int lanePins[LANE_COUNT][3] = {
  {2, 3, 4},     // Lane 1: R,Y,G
  {5, 6, 7},     // Lane 2: R,Y,G
  {8, 9, 10},    // Lane 3: R,Y,G
  {11, 12, 13}   // Lane 4: R,Y,G
};

const unsigned long YELLOW_TIME_MS = 700;
int currentGreenLane = -1;

void setAllRed() {
  for (int lane = 0; lane < LANE_COUNT; lane++) {
    digitalWrite(lanePins[lane][0], HIGH);  // RED ON
    digitalWrite(lanePins[lane][1], LOW);   // YELLOW OFF
    digitalWrite(lanePins[lane][2], LOW);   // GREEN OFF
  }
}

void setLaneGreen(int lane) {
  // Safety transition: all red first
  setAllRed();

  if (lane >= 0 && lane < LANE_COUNT) {
    digitalWrite(lanePins[lane][0], LOW);   // RED OFF
    digitalWrite(lanePins[lane][2], HIGH);  // GREEN ON
    currentGreenLane = lane;
  }
}

void transitionToLane(int nextLane) {
  if (nextLane < 0 || nextLane >= LANE_COUNT) {
    return;
  }

  if (currentGreenLane == nextLane) {
    // Keep same lane green.
    setLaneGreen(nextLane);
    return;
  }

  // Turn current green to yellow briefly.
  if (currentGreenLane >= 0 && currentGreenLane < LANE_COUNT) {
    digitalWrite(lanePins[currentGreenLane][2], LOW);  // GREEN OFF
    digitalWrite(lanePins[currentGreenLane][1], HIGH); // YELLOW ON
    delay(YELLOW_TIME_MS);
    digitalWrite(lanePins[currentGreenLane][1], LOW);  // YELLOW OFF
    digitalWrite(lanePins[currentGreenLane][0], HIGH); // RED ON
  }

  setLaneGreen(nextLane);
}

void setup() {
  Serial.begin(115200);

  for (int lane = 0; lane < LANE_COUNT; lane++) {
    for (int color = 0; color < 3; color++) {
      pinMode(lanePins[lane][color], OUTPUT);
    }
  }

  setAllRed();
  Serial.println("Traffic controller ready. Send 0,1,2,3 over serial.");
}

void loop() {
  if (Serial.available() > 0) {
    int action = Serial.parseInt();

    // Flush any trailing chars like newline
    while (Serial.available() > 0) {
      Serial.read();
    }

    if (action >= 0 && action < LANE_COUNT) {
      transitionToLane(action);
      Serial.print("Applied action: ");
      Serial.println(action);
    } else {
      Serial.print("Invalid action: ");
      Serial.println(action);
    }
  }
}
