import asyncio
import numpy as np
import pandas as pd
from touch_sdk import Watch
import joblib
from collections import deque
import time
import warnings
from tensorflow.keras.models import load_model
import tensorflow as tf
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
WINDOW_SIZE = 100 

# CRITICAL FIX: Must match the 16 features used for training (mag_cal_x/y/z removed)
FEATURES = [
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_x", "gyro_y", "gyro_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
    "mag_x", "mag_y", "mag_z"
]
NUM_FEATURES = len(FEATURES) # Should be 16
EPSILON = 1e-7 # Used for division stability during scaling

# ------------------------------------------------------------
# Load trained model and scaling parameters (loaded into global scope)
# ------------------------------------------------------------
print("🔄 Loading model and scaling parameters...")

# Load the Keras model using the standard Keras function
try:
    model = load_model("gesture_rec_model.h5")
except Exception as e:
    print(f"ERROR loading Keras model: {e}")
    print("Please ensure 'gesture_rec_model.h5' is correctly saved.")
    exit()

# Load the scaling parameters (mean and std dev)
try:
    mean, std = joblib.load("scaling_params.pkl")
    if mean.ndim == 3 and std.ndim == 3:
        if mean.shape[-1] != NUM_FEATURES:
             raise ValueError(f"Scaling array feature count mismatch. Expected {NUM_FEATURES}, got {mean.shape[-1]}")
    else:
        print("Warning: Scaling parameters are not in expected (1, 1, N) shape. Proceeding with caution.")
        mean = mean.reshape(1, 1, NUM_FEATURES)
        std = std.reshape(1, 1, NUM_FEATURES)
        
except Exception as e:
    print(f"ERROR loading scaling parameters: {e}")
    print("Please ensure 'scaling_params.pkl' is correctly saved as a tuple (mean, std).")
    exit()

# ------------------------------------------------------------
# Class for gesture prediction (Now handles communication and events)
# ------------------------------------------------------------
class GesturePredictor(Watch):
    def __init__(self, model, mean, std, epsilon):
        super().__init__()
        
        # Store model and parameters as instance attributes
        self.model = model
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        
        # 🔑 NEW: Base URL for all gesture commands
        self.base_url = "http://192.168.0.105:8000/"
        
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.gesture_history = deque(maxlen=5) 
        
        self.last_prediction_time = 0
        self.prediction_interval = 0.5 
        
        # 0: Backward, 1: Forward, 2: Handwave, 3: Random
        self.label_map = {0: "Backward", 1: "Forward", 2: "Handwave", 3: "Random"} 
        
        # Command map uses the desired URL path/command name
        self.command_map = {
            "Backward": "move-backward",
            "Forward": "move-forward",
            "Handwave": "wave-hand" 
        }
        
        self.wave_active = False 
    
    async def on_connect(self):
        print("✅ Connected to Doublepoint Watch! Starting gesture recognition and event listening...")
        print(f"Model configured for {WINDOW_SIZE} time steps and {NUM_FEATURES} features.")

    # 🔑 MODIFIED: Network communication method uses the command to build the URL
    def send_command(self, command):
        """Sends a POST request command to the specific URL matching the command."""
        # Construct the URL: e.g., http://192.168.0.105:8000/move-backward
        url = f"{self.base_url}{command}" 
        
        # The payload contains the command name
        payload = {"command": command}
        
        try:
            response = requests.post(url, json=payload, timeout=3)
            print(f"Sent command: {command} to {url}. Response: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending command to {url}: {e}")

    # Tap event handler (No command sent, as requested)
    def on_tap(self):
        print("👆 Tap detected! No command sent.")

    # Pinch event handler
    def on_pinch(self):
        print("🤏 Pinch detected!")

    # Release event handler
    def on_release(self):
        print("🖐️ Release detected!")

    def on_sensors(self, sensors):
        # Data collection MUST match the 16 features used for training
        sample = {
            "accel_x": sensors.acceleration[0], "accel_y": sensors.acceleration[1], "accel_z": sensors.acceleration[2],
            "gravity_x": sensors.gravity[0], "gravity_y": sensors.gravity[1], "gravity_z": sensors.gravity[2],
            "gyro_x": sensors.angular_velocity[0], "gyro_y": sensors.angular_velocity[1], "gyro_z": sensors.angular_velocity[2],
            "orient_x": sensors.orientation[0], "orient_y": sensors.orientation[1], "orient_z": sensors.orientation[2], "orient_w": sensors.orientation[3],
            "mag_x": sensors.magnetic_field[0] if sensors.magnetic_field else 0,
            "mag_y": sensors.magnetic_field[1] if sensors.magnetic_field else 0,
            "mag_z": sensors.magnetic_field[2] if sensors.magnetic_field else 0,
        }
        
        self.buffer.append(sample)

        current_time = time.time()
        # Predict when the buffer is full and the interval has passed
        if len(self.buffer) == self.buffer.maxlen and (current_time - self.last_prediction_time > self.prediction_interval):
            self.predict_gesture()
            self.last_prediction_time = current_time
            # SLIDING WINDOW: Remove half the samples to allow 50% overlap for continuous detection
            for _ in range(WINDOW_SIZE // 2): 
                 if self.buffer:
                    self.buffer.popleft()


    def predict_gesture(self):
        
        # 1. Data processing
        df = pd.DataFrame(self.buffer)
        X = df[FEATURES].values 
        
        # Use self.mean, self.std, self.epsilon
        X_scaled = (X - self.mean) / (self.std + self.epsilon) 
        X_scaled_cnn_input = X_scaled.reshape(1, WINDOW_SIZE, NUM_FEATURES)

        # 2. Prediction
        # Use self.model
        probabilities = self.model.predict(X_scaled_cnn_input, verbose=0)[0] 
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index] 
        
        # 3. CONSENSUS BUFFER LOGIC 
        self.gesture_history.append((predicted_class_index, confidence))

        if len(self.gesture_history) == self.gesture_history.maxlen:
            
            indices = [item[0] for item in self.gesture_history]
            confidences = [item[1] for item in self.gesture_history]

            is_consensus = all(idx == indices[0] for idx in indices)
            median_confidence = np.median(confidences)

            if is_consensus and median_confidence > 0.70:
                final_index = indices[0]
                final_gesture_label = self.label_map.get(final_index, "UNKNOWN")
                
                print(f"✅ CONFIRMED: {final_gesture_label} (Median Conf: {median_confidence:.2f})")
                
                # --- Command Dispatch ---
                command_to_send = self.command_map.get(final_gesture_label)
                
                if command_to_send:
                    self.send_command(command_to_send)
                elif final_gesture_label in ["Random", "UNKNOWN"]:
                    print(f"🚫 {final_gesture_label} confirmed. No command sent.")

                # Clear the history buffer to look for the START of the next distinct gesture
                self.gesture_history.clear()

# ------------------------------------------------------------
# Run the gesture prediction loop
# ------------------------------------------------------------
async def main():
    print("🔗 Searching for Doublepoint Watch...")
    # Pass the globally loaded variables to the instance
    predictor = GesturePredictor(model, mean, std, EPSILON) 
    await predictor.run() 

if __name__ == "__main__":
    try:
        # Check if TensorFlow is working and imported
        _ = tf.random.uniform((1,)) 
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Stopping gesture prediction...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")