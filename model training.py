import asyncio
import numpy as np
import pandas as pd
from touch_sdk import Watch
import joblib
from collections import deque
import time
import warnings
import inspect

# suppress sklearn version warnings shown when unpickling across versions
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ------------------------------------------------------------
# Load trained model and scaler
# ------------------------------------------------------------
print("🔄 Loading model and scaler...")
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")

# The exact feature order used during training
FEATURES = [
    "accel_x", "accel_y", "accel_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyro_x", "gyro_y", "gyro_z",
    "orient_x", "orient_y", "orient_z", "orient_w",
    "mag_x", "mag_y", "mag_z"
]

# ------------------------------------------------------------
# Class to handle watch connection and prediction
# ------------------------------------------------------------
class GesturePredictor(Watch):
    def __init__(self, window_size=30):
        super().__init__()
        self.buffer = deque(maxlen=window_size)
        self.last_prediction_time = 0.0
        self.prediction_interval = 1.0  # seconds between predictions
        self.target_name = None  # optional: set to "DPSQYI" or other

    async def record_and_predict(self):
        """Main async loop for connecting to the watch and streaming."""
        print("🔗 Searching for Doublepoint Watch...")
        # try using target_name if set, otherwise explicit id
        device_id = self.target_name or "DPSQYI"
        await self.connect(device_id)
        print("✅ Connected! Starting real-time prediction...")

        # call run() — handle if it's a coroutine or sync function
        try:
            result = self.run()
            if asyncio.iscoroutine(result):
                await result
            # else assume the SDK started internal streaming and returned None or blocking result
        except Exception as e:
            # If run() raised because it's meant to be scheduled differently, show a helpful message
            print(f"⚠️ Error while starting SDK run(): {e}")
            raise

    def on_sensors(self, sensors):
        """Callback triggered when new sensor data is received."""
        try:
            # Defensive indexing in case some fields are missing
            accel = getattr(sensors, "acceleration", [0, 0, 0])
            gravity = getattr(sensors, "gravity", [0, 0, 0])
            gyro = getattr(sensors, "angular_velocity", [0, 0, 0])
            orient = getattr(sensors, "orientation", [0, 0, 0, 1])
            mag = getattr(sensors, "magnetic_field", None)

            sample = {
                "accel_x": accel[0], "accel_y": accel[1], "accel_z": accel[2],
                "gravity_x": gravity[0], "gravity_y": gravity[1], "gravity_z": gravity[2],
                "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
                "orient_x": orient[0], "orient_y": orient[1], "orient_z": orient[2], "orient_w": orient[3] if len(orient) > 3 else 1,
                "mag_x": mag[0] if mag else 0, "mag_y": mag[1] if mag else 0, "mag_z": mag[2] if mag else 0,
            }

            self.buffer.append(sample)

            # Make a prediction every `prediction_interval` seconds if buffer is full
            current_time = time.time()
            if len(self.buffer) == self.buffer.maxlen and (current_time - self.last_prediction_time > self.prediction_interval):
                try:
                    self.predict_gesture()
                except Exception as e:
                    print(f"⚠️ Prediction failed: {e}")
                self.last_prediction_time = current_time
        except Exception as e:
            print(f"⚠️ Error in on_sensors: {e}")

    def predict_gesture(self):
        """Use the trained model to predict the gesture."""
        if len(self.buffer) == 0:
            return
        df = pd.DataFrame(self.buffer)
        # ensure all features exist; if not, fill missing with zeros
        for f in FEATURES:
            if f not in df.columns:
                df[f] = 0
        X = df[FEATURES].mean().to_numpy().reshape(1, -1)
        X_scaled = scaler.transform(X)
        gesture = model.predict(X_scaled)[0]
        print(f"🟢 Detected Gesture: {gesture}")

    async def connect(self, device_name=None):
        """
        Handles BLE connection explicitly (overrides base).
        Tries super().connect if available, otherwise tries common alternatives.
        Tries both (device_name) and no-argument calls tolerantly.
        """
        try:
            # Build list of candidate methods (bound to instance or via super)
            candidates = []

            # Prefer the base class implementation if present (bound to this instance)
            try:
                base_connect = getattr(super(GesturePredictor, self), "connect", None)
                if base_connect:
                    candidates.append(base_connect)
            except Exception:
                pass

            # Add possible method names on the instance (bound methods)
            alt_names = ("connect", "connect_watch", "connect_device", "open", "start", "connect_to")
            for name in alt_names:
                # avoid re-adding our override 'connect' from GesturePredictor class
                if name == "connect" and hasattr(super(GesturePredictor, self), "connect"):
                    # base connect already added
                    continue
                method = getattr(self, name, None)
                if method and callable(method):
                    candidates.append(method)

            # iterate candidates and try invoking (device_name first, then without args)
            for method in candidates:
                if method is None:
                    continue
                # skip if method is this very override (unbound)
                if getattr(method, "__func__", None) is getattr(GesturePredictor, "connect", None):
                    continue

                is_coro = asyncio.iscoroutinefunction(method)
                # Try call with device_name if provided
                if device_name is not None:
                    try:
                        if is_coro:
                            await method(device_name)
                        else:
                            method(device_name)
                        return
                    except TypeError:
                        # method may not accept device_name, fall through to try without args
                        pass
                    except Exception as e:
                        # other exceptions are likely real connection failures
                        print(f"❌ Candidate method {method} raised: {e}")
                        raise

                # Try calling without args
                try:
                    if is_coro:
                        await method()
                    else:
                        method()
                    return
                except Exception as e:
                    # try next candidate
                    # If this candidate errors, show debug info and continue
                    print(f"⚠️ Candidate {method} failed when called without args: {e}")
                    continue

            raise AttributeError("No connect-like method found on Watch (tried common alternatives)")
        except Exception as e:
            print(f"❌ Failed to connect to watch: {e}")
            raise


# ------------------------------------------------------------
# Run the gesture prediction loop
# ------------------------------------------------------------
if __name__ == "__main__":
    predictor = GesturePredictor(window_size=30)
    try:
        # Use a new event loop (avoids DeprecationWarning about get_event_loop on some setups)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(predictor.record_and_predict())
    except KeyboardInterrupt:
        print("🛑 Stopping gesture prediction...")
    except Exception as e:
        print(f"⚠️ Error during execution: {e}")
