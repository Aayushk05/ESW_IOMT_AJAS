/*
 * ESP32 COMBINED ECG AND FALL DETECTION SYSTEM
 * 
 * PIN CONNECTIONS:
 * 
 * MPU6050 (Accelerometer/Gyroscope - I2C Communication):
 * --------------------------------------------------------
 * VCC  -> 3.3V (ESP32)
 * GND  -> GND (ESP32)
 * SDA  -> GPIO 21 (ESP32)
 * SCL  -> GPIO 22 (ESP32)
 * 
 * Note: MPU6050 ONLY supports I2C communication protocol.
 * The SDA and SCL pins MUST be used - there is no alternative.
 * 
 * 
 * AD8232 (ECG Sensor - Analog/Digital):
 * --------------------------------------------------------
 * VCC      -> 3.3V (ESP32)
 * GND      -> GND (ESP32)
 * OUTPUT   -> GPIO 34 (ESP32 ADC1_CH6 - Analog Input)
 * LO+      -> GPIO 32 (ESP32 - Digital Input)
 * LO-      -> GPIO 33 (ESP32 - Digital Input)
 * 
 * ELECTRODE CONNECTIONS (AD8232):
 * --------------------------------------------------------
 * RA (Right Arm)  -> Right side of chest or right arm
 * LA (Left Arm)   -> Left side of chest or left arm  
 * RL (Right Leg)  -> Lower right torso (reference/ground)
 * 
 * These two sensors do NOT conflict because they use different
 * communication protocols and different GPIO pins.
 */

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// -------- WiFi credentials --------
const char* ssid = "iPhone";
const char* password = "12345678";

// -------- MQTT broker (your computer) --------
const char* mqtt_server = "172.20.10.2"; // <-- change to your computer's IP
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP32_TestClient";

// Create WiFi and MQTT clients
WiFiClient espClient;
PubSubClient client(espClient);

//String t;
//char msg[10];

// ==================== MPU6050 CONFIGURATION ====================
const int MPU_ADDR = 0x68;

// Raw sensor data variables
int16_t accelerometer_x, accelerometer_y, accelerometer_z;
int16_t gyroscope_x, gyroscope_y, gyroscope_z;

// Converted data variables
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;

// Fall detection thresholds
#define FALL_THRESHOLD 3.5
#define IMPACT_DURATION 500
#define STILLNESS_THRESHOLD 0.5
#define STILLNESS_DURATION 4000
#define DEBOUNCE_TIME 10000
#define GYRO_THRESHOLD 400
#define POST_FALL_GYRO_THRESH 20
#define FALL_CONFIRM_TIMEOUT 5000
#define MOVEMENT_CANCEL_THRESH 2.0
#define FALL_ALERT_DURATION 60000

// Fall detection variables
unsigned long lastFallTime = 0;
bool potentialFall = false;
unsigned long fallDetectedTime = 0;
bool stillnessDetected = false;
unsigned long stillnessTime = 0;
unsigned long fallConfirmedTime = 0;

// ==================== AD8232 CONFIGURATION ====================
const int ECG_OUTPUT_PIN = 34;   // Analog pin for ECG signal
const int LO_PLUS_PIN = 32;      // Leads-off detect +
const int LO_MINUS_PIN = 33;     // Leads-off detect -

// ==================== TIMING VARIABLES ====================
unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL = 1000; // Print every 1 second

// Connect to Wi-Fi
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// Connect to MQTT broker
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect(mqtt_client_id)) {
      Serial.println("connected!");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Initialize MPU6050 (I2C)
  Wire.begin(4, 5); // SDA = GPIO21, SCL = GPIO22
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Wake up the MPU6050
  Wire.endTransmission(true);
  
  // Initialize AD8232 pins
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  
  Serial.println("==========================================");
  Serial.println("ECG + Fall Detection System Initialized");
  Serial.println("==========================================");
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);

  delay(1000);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();
  
  client.setKeepAlive(60);
  WiFi.setSleep(false);


  unsigned long currentTime = millis();
  
  // ==================== READ MPU6050 DATA ====================
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  
  // Read accelerometer
  accelerometer_x = Wire.read() << 8 | Wire.read();
  accelerometer_y = Wire.read() << 8 | Wire.read();
  accelerometer_z = Wire.read() << 8 | Wire.read();
  
  // Skip temperature (2 bytes)
  Wire.read();
  Wire.read();
  
  // Read gyroscope
  gyroscope_x = Wire.read() << 8 | Wire.read();
  gyroscope_y = Wire.read() << 8 | Wire.read();
  gyroscope_z = Wire.read() << 8 | Wire.read();
  
  // Convert to meaningful units
  accel_x = (accelerometer_x / 16384.0) * 9.81;
  accel_y = (accelerometer_y / 16384.0) * 9.81;
  accel_z = (accelerometer_z / 16384.0) * 9.81;
  
  gyro_x = gyroscope_x / 131.0;
  gyro_y = gyroscope_y / 131.0;
  gyro_z = gyroscope_z / 131.0;
  
  // ==================== READ ECG DATA ====================
  int ecgValue = 0;
  bool leadsOff = false;
  
  if (digitalRead(LO_PLUS_PIN) == HIGH || digitalRead(LO_MINUS_PIN) == HIGH) {
    ecgValue = 0;
    leadsOff = true;
  } else {
    ecgValue = analogRead(ECG_OUTPUT_PIN);
  }
  
  // ==================== FALL DETECTION ALGORITHM ====================
  float accel_x_g = accel_x / 9.81;
  float accel_y_g = accel_y / 9.81;
  float accel_z_g = accel_z / 9.81;
  float totalAcceleration = sqrt(accel_x_g * accel_x_g + accel_y_g * accel_y_g + accel_z_g * accel_z_g);
  float totalGyro = sqrt(gyro_x * gyro_x + gyro_y * gyro_y + gyro_z * gyro_z);
  
  // Step 1: Detect potential fall
  if (!potentialFall && !stillnessDetected && 
      (totalAcceleration > FALL_THRESHOLD || totalGyro > GYRO_THRESHOLD) && 
      (currentTime - lastFallTime > DEBOUNCE_TIME)) {
      
    potentialFall = true;
    fallDetectedTime = currentTime;
    Serial.println("\n----- POTENTIAL FALL DETECTED! -----");
    Serial.println("Monitoring for post-fall stillness...\n");
  }
  
  // Step 2: Check for post-fall stillness
  if (potentialFall && !stillnessDetected && 
      (currentTime - fallDetectedTime > IMPACT_DURATION)) {
      
    if (abs(totalAcceleration - 1.0) < STILLNESS_THRESHOLD && 
        totalGyro < POST_FALL_GYRO_THRESH) {
        
      stillnessDetected = true;
      stillnessTime = currentTime;
      Serial.println("\n----- POST-FALL STILLNESS DETECTED -----");
      Serial.println("Monitoring for confirmation...\n");
    }
    
    if (currentTime - fallDetectedTime > FALL_CONFIRM_TIMEOUT) {
      potentialFall = false;
      stillnessDetected = false;
      Serial.println("\n----- FALSE ALARM -----\n");
    }
  }
  
  // Step 3: Monitor stillness period
  if (stillnessDetected) {
    if (totalAcceleration > MOVEMENT_CANCEL_THRESH || totalGyro > POST_FALL_GYRO_THRESH*2) {
      stillnessDetected = false;
      potentialFall = false;
      Serial.println("\n----- FALL ALERT CANCELED -----");
      Serial.println("Movement detected - person recovered\n");
    }
    else if (currentTime - stillnessTime > STILLNESS_DURATION) {
      triggerAlert();
      lastFallTime = currentTime;
      fallConfirmedTime = currentTime;
      potentialFall = false;
      stillnessDetected = false;
    }
  }
  
  // ==================== PRINT DATA ====================
  if (currentTime - lastPrintTime >= PRINT_INTERVAL) {
    lastPrintTime = currentTime;
    
    Serial.println("======================================");
    
    // ECG Data
    Serial.println("ECG SENSOR:");
    if (leadsOff) {
      Serial.println("  Status: LEADS OFF - Check electrode connections!");
      Serial.println("  Raw ECG: 0");
    } else {
      Serial.println("  Status: Reading");
      Serial.print("  Raw ECG: ");
      Serial.println(ecgValue);
    }
    
    Serial.println();
    
    // Fall Detection Status
    Serial.println("FALL DETECTION:");
    if (potentialFall) {
      Serial.println("  Status: Potential fall detected, monitoring");
    } else if (stillnessDetected) {
      Serial.println("  Status: Stillness detected, confirming fall");
    } else {
      Serial.println("  Status: Normal monitoring");
    }
    
    int fallStatus = (fallConfirmedTime > 0 && currentTime - fallConfirmedTime <= FALL_ALERT_DURATION) ? 1 : 0;
    Serial.print("  Fall Alert: ");
    Serial.println(fallStatus);
    
    Serial.println();
    
    // Acceleration Data
    Serial.println("ACCELERATION (m/s²):");
    Serial.print("  X: ");
    Serial.print(accel_x, 2);
    Serial.print("  Y: ");
    Serial.print(accel_y, 2);
    Serial.print("  Z: ");
    Serial.println(accel_z, 2);

    /*StaticJsonDocument<200> doc;
    doc["accel_x"] = accel_x;
    doc["accel_y"] = accel_y;
    doc["accel_z"] = accel_z;
    doc["fallStatus"] = fallStatus;
    doc["ecgValue"] = ecgValue;

    char buffer[256];
    serializeJson(doc, buffer);
    client.publish("test/sensors", buffer);*/

    char msg[128];
    sprintf(msg, "accel accel_x=%f,accel_y=%f,accel_z=%f,fallStatus=%d,ecgValue=%d", accel_x, accel_y, accel_z, fallStatus, ecgValue);
    client.publish("test/sensors", msg);
    
    Serial.println("======================================\n");
  }
  
  delay(10);
}

void triggerAlert() {
  Serial.println("\n\n!!!!!! FALL CONFIRMED - SENDING ALERT !!!!!!");
  Serial.println("Fall detection triggered!");
  Serial.println("Alert completed.\n\n");
}
