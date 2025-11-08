#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "Galaxy M15 5G E367";
const char* password = "2210@Joshua";

String t;

// Create two I2C instances
TwoWire I2C_MPU = TwoWire(0);  // I2C bus for MPU6050
TwoWire I2C_MAX = TwoWire(1);  // I2C bus for MAX30102

MAX30105 particleSensor;

// MPU6050 pins (default)
#define MPU_SDA 21
#define MPU_SCL 22

// MAX30102 pins (custom)
#define MAX_SDA 25
#define MAX_SCL 26

// MPU6050 I2C address
const int MPU_ADDR = 0x68;

// MPU6050 Raw sensor data variables
int16_t accelerometer_x, accelerometer_y, accelerometer_z;
int16_t gyroscope_x, gyroscope_y, gyroscope_z;
int16_t temperature;

// MPU6050 Converted data variables
float accel_x, accel_y, accel_z;
float gyro_x, gyro_y, gyro_z;
float temp_c;

// MPU6050 Calibration offsets (adjust these after calibration)
float accel_offset_x = 0;
float accel_offset_y = 0;
float accel_offset_z = 0;
float gyro_offset_x = 0;
float gyro_offset_y = 0;
float gyro_offset_z = 0;

// MAX30102 Variables
#define MAX_BRIGHTNESS 255
uint32_t irBuffer[100];
uint32_t redBuffer[100];
int32_t bufferLength;
int32_t spo2;
int8_t validSPO2;
int32_t heartRate;
int8_t validHeartRate;

// MAX30102 Averaging variables (not used for printing)
int validHRCount = 0;
int validSpO2Count = 0;
long sumHR = 0;
long sumSpO2 = 0;

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
unsigned long lastPrintTime = 0;
unsigned long lastMaxPrintTime = 0;

void postit(String url, String value) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(url);
    http.addHeader("X-M2M-Origin", "admin:admin");
    http.addHeader("X-M2M-RI", "req123");
    http.addHeader("Content-Type", "application/json;ty=4");

    // JSON body with contentInstance structure
    String payload = "{\"m2m:cin\": {\"con\": \"" + value + "\"}}";

    int httpResponseCode = http.POST(payload);

    Serial.println("POST URL: " + url);
    Serial.println("Payload: " + payload);
    Serial.println("HTTP Response code: " + String(httpResponseCode));
    Serial.println("Response: " + http.getString());

    http.end();
  } else {
    Serial.println("WiFi not connected");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing sensors...");
  
  // Initialize MPU6050 on pins 21,22
  I2C_MPU.begin(MPU_SDA, MPU_SCL, 100000);
  I2C_MPU.beginTransmission(MPU_ADDR);
  I2C_MPU.write(0x6B);
  I2C_MPU.write(0);
  I2C_MPU.endTransmission(true);
  Serial.println("MPU6050 initialized on pins 21,22");
  
  // Initialize MAX30102 on pins 25,26
  I2C_MAX.begin(MAX_SDA, MAX_SCL, 100000);
  
  if (!particleSensor.begin(I2C_MAX, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Check wiring/power.");
    while (1);
  }
  
  // Optimized MAX30102 settings for better accuracy
  byte ledBrightness = 60;   // 0-255, higher = better SNR but more power
  byte sampleAverage = 4;    // 1, 2, 4, 8, 16, 32
  byte ledMode = 2;          // 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  byte sampleRate = 100;     // 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411;      // 69, 118, 215, 411 (μs)
  int adcRange = 4096;       // 2048, 4096, 8192, 16384
  
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  
  // Set RED LED amplitude for SpO2 (increase for better signal)
  particleSensor.setPulseAmplitudeRed(0x1F);  // 0x00 to 0xFF
  particleSensor.setPulseAmplitudeIR(0x1F);   // 0x00 to 0xFF
  particleSensor.setPulseAmplitudeGreen(0);   // Turn off Green LED
  
  Serial.println("MAX30102 initialized with optimized settings on pins 25,26");
  
  Serial.println("\n==========================================");
  Serial.println("CALIBRATING MPU6050...");
  Serial.println("Place MPU6050 FLAT and STILL for 5 seconds");
  Serial.println("==========================================");
  delay(2000);
  
  // Calibrate MPU6050
  calibrateMPU6050();
  
  Serial.println("Calibration complete!");
  Serial.println("==========================================\n");
  
  // Fill initial buffer for MAX30102
  bufferLength = 100;
  for (int i = 0; i < bufferLength; i++) {
    while (particleSensor.available() == false)
      particleSensor.check();
    
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }
  
  lastPrintTime = millis();
  Serial.println("Both sensors ready!");
  Serial.println("==========================================\n");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
      delay(1000);
      Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");
  
  delay(1000);
}

void loop() {
  unsigned long currentTime = millis();
  
  // ========== Read MPU6050 Data ==========
  I2C_MPU.beginTransmission(MPU_ADDR);
  I2C_MPU.write(0x3B);
  I2C_MPU.endTransmission(false);
  I2C_MPU.requestFrom(MPU_ADDR, 14, true);
  
  accelerometer_x = I2C_MPU.read() << 8 | I2C_MPU.read();
  accelerometer_y = I2C_MPU.read() << 8 | I2C_MPU.read();
  accelerometer_z = I2C_MPU.read() << 8 | I2C_MPU.read();
  temperature = I2C_MPU.read() << 8 | I2C_MPU.read();
  gyroscope_x = I2C_MPU.read() << 8 | I2C_MPU.read();
  gyroscope_y = I2C_MPU.read() << 8 | I2C_MPU.read();
  gyroscope_z = I2C_MPU.read() << 8 | I2C_MPU.read();
  
  // Convert MPU6050 data with calibration offsets
  accel_x = ((accelerometer_x / 16384.0) * 9.81) - accel_offset_x;
  accel_y = ((accelerometer_y / 16384.0) * 9.81) - accel_offset_y;
  accel_z = ((accelerometer_z / 16384.0) * 9.81) - accel_offset_z;
  gyro_x = (gyroscope_x / 131.0) - gyro_offset_x;
  gyro_y = (gyroscope_y / 131.0) - gyro_offset_y;
  gyro_z = (gyroscope_z / 131.0) - gyro_offset_z;
  temp_c = temperature / 340.0 + 36.53;
  
  // ========== Read MAX30102 Data ==========
  while (particleSensor.available() == false)
    particleSensor.check();
  
  // Shift data in buffers (rolling window)
  for (int i = 0; i < bufferLength - 1; i++) {
    redBuffer[i] = redBuffer[i + 1];
    irBuffer[i] = irBuffer[i + 1];
  }
  
  redBuffer[bufferLength - 1] = particleSensor.getRed();
  irBuffer[bufferLength - 1] = particleSensor.getIR();
  particleSensor.nextSample();
  
  // ========== Fall Detection Algorithm ==========
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
    Serial.println("\n\n\n\n\n");
    Serial.println("----- POTENTIAL FALL DETECTED! -----");
    Serial.println("\n\n\n");
    Serial.println("Monitoring for post-fall stillness...");
    Serial.println("\n\n\n\n\n");
  }
  
  // Step 2: Check for post-fall stillness
  if (potentialFall && !stillnessDetected && 
      (currentTime - fallDetectedTime > IMPACT_DURATION)) {
    if (abs(totalAcceleration - 1.0) < STILLNESS_THRESHOLD && 
        totalGyro < POST_FALL_GYRO_THRESH) {
      stillnessDetected = true;
      stillnessTime = currentTime;
      Serial.println("\n\n\n\n\n");
      Serial.println("----- POST-FALL STILLNESS DETECTED -----");
      Serial.println("\n\n\n");
      Serial.println("Monitoring for confirmation... will alert in " + 
                    String(STILLNESS_DURATION/1000) + " seconds if stillness continues");
      Serial.println("\n\n\n\n\n");
    }
    
    if (currentTime - fallDetectedTime > FALL_CONFIRM_TIMEOUT) {
      potentialFall = false;
      stillnessDetected = false;
      Serial.println("\n\n\n\n\n");
      Serial.println("----- FALSE ALARM -----");
      Serial.println("\n\n\n");
      Serial.println("No sustained stillness within timeout period");
      Serial.println("\n\n\n\n\n");
    }
  }
  
  // Step 3: Monitor stillness period
  if (stillnessDetected) {
    if (totalAcceleration > MOVEMENT_CANCEL_THRESH || totalGyro > POST_FALL_GYRO_THRESH*2) {
      stillnessDetected = false;
      potentialFall = false;
      Serial.println("\n\n\n\n\n");
      Serial.println("----- FALL ALERT CANCELED -----");
      Serial.println("\n\n\n");
      Serial.println("Movement detected during confirmation period - person likely recovered");
      Serial.println("\n\n\n\n\n");
    }
    else if (currentTime - stillnessTime > STILLNESS_DURATION) {
      triggerAlert();
      lastFallTime = currentTime;
      fallConfirmedTime = currentTime;
      potentialFall = false;
      stillnessDetected = false;
    }
  }
  
  // ========== Print Data Every 2 Seconds ==========
  if (currentTime - lastMaxPrintTime >= 2000) {
    maxim_heart_rate_and_oxygen_saturation(
      irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);
    
    lastMaxPrintTime = currentTime;

    //Sending data to server
    if (validHeartRate) {
      t=String(heartRate);
      postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/heart_rate", t);
    } else {
      postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/heart_rate", "Invalid");
    }
    if (validSPO2) {
      t=String(spo2);
      postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/spo2", t);
    } else {
      postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/spo2", "Invalid");
    }
    t=String(temp_c);
    postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/temp", t);
    t=String(sqrt(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z));
    postit("http://10.54.250.29:5089/~/in-cse/in-name/Esw_AE/acceleration", t);
    //End of data sending
    
    Serial.println("--- MAX30102 Readings ---");
    Serial.print("Heart Rate: ");
    if (validHeartRate) {
      Serial.print(heartRate);
    } else {
      Serial.print("Invalid");
    }
    Serial.print(" bpm | SpO2: ");
    if (validSPO2) {
      Serial.print(spo2);
    } else {
      Serial.print("Invalid");
    }
    Serial.println(" %");
    
    Serial.println("\n--- MPU6050 Sensor Readings ---");
    
    Serial.print("Temperature: ");
    Serial.print(temp_c, 2);
    Serial.println(" °C");
    
    Serial.print("Linear Acceleration (m/s²): X=");
    Serial.print(accel_x, 2);
    Serial.print(" Y=");
    Serial.print(accel_y, 2);
    Serial.print(" Z=");
    Serial.print(accel_z, 2);
    Serial.print(" Total=");
    Serial.print(sqrt(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z), 2);
    Serial.println(" m/s²");
    
    Serial.println("FALL DETECTION STATUS:");
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
    
    Serial.println("================================\n");
  }
  
  delay(10);
}

void triggerAlert() {
  Serial.println("\n\n\n\n\n");
  Serial.println("!!!!!! FALL CONFIRMED - SENDING ALERT !!!!!");
  Serial.println("\n\n\n");
  Serial.println("Fall detection triggered!");
  Serial.print("Temperature at time of fall: ");
  Serial.print(temp_c, 2);
  Serial.println(" °C");
  
  Serial.println("Fall alert activated - emergency response recommended");
  Serial.println("Alert completed.");
  
  Serial.println("\n\n\n\n\n");
}

void calibrateMPU6050() {
  const int numSamples = 200;
  long sum_ax = 0, sum_ay = 0, sum_az = 0;
  long sum_gx = 0, sum_gy = 0, sum_gz = 0;
  
  for (int i = 0; i < numSamples; i++) {
    I2C_MPU.beginTransmission(MPU_ADDR);
    I2C_MPU.write(0x3B);
    I2C_MPU.endTransmission(false);
    I2C_MPU.requestFrom(MPU_ADDR, 14, true);
    
    int16_t ax = I2C_MPU.read() << 8 | I2C_MPU.read();
    int16_t ay = I2C_MPU.read() << 8 | I2C_MPU.read();
    int16_t az = I2C_MPU.read() << 8 | I2C_MPU.read();
    I2C_MPU.read(); I2C_MPU.read(); // Skip temperature
    int16_t gx = I2C_MPU.read() << 8 | I2C_MPU.read();
    int16_t gy = I2C_MPU.read() << 8 | I2C_MPU.read();
    int16_t gz = I2C_MPU.read() << 8 | I2C_MPU.read();
    
    sum_ax += ax;
    sum_ay += ay;
    sum_az += az;
    sum_gx += gx;
    sum_gy += gy;
    sum_gz += gz;
    
    delay(10);
  }
  
  // Calculate average offsets
  accel_offset_x = ((sum_ax / numSamples) / 16384.0) * 9.81;
  accel_offset_y = ((sum_ay / numSamples) / 16384.0) * 9.81;
  accel_offset_z = (((sum_az / numSamples) / 16384.0) * 9.81) - 9.81; // Subtract gravity
  
  gyro_offset_x = (sum_gx / numSamples) / 131.0;
  gyro_offset_y = (sum_gy / numSamples) / 131.0;
  gyro_offset_z = (sum_gz / numSamples) / 131.0;
  
  Serial.print("Accel Offsets (m/s²): X=");
  Serial.print(accel_offset_x, 3);
  Serial.print(" Y=");
  Serial.print(accel_offset_y, 3);
  Serial.print(" Z=");
  Serial.println(accel_offset_z, 3);
  
  Serial.print("Gyro Offsets (°/s): X=");
  Serial.print(gyro_offset_x, 3);
  Serial.print(" Y=");
  Serial.print(gyro_offset_y, 3);
  Serial.print(" Z=");
  Serial.println(gyro_offset_z, 3);
}