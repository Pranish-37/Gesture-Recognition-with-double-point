import asyncio
import csv
import time
from touch_sdk import Watch


class SensorRecorder(Watch):
    def __init__(self, label, filename="gesture_data.csv"):
        super().__init__()
        self.label = label
        self.filename = filename
        self.data = []
        self.recording = False

    def on_connect(self):
        print("✅ Watch connected.")

    def on_disconnect(self):
        print("🔌 Watch disconnected.")

    def on_sensors(self, sensors):
        if not self.recording:
            return

        timestamp = time.time()
        accel = sensors.acceleration or (None, None, None)
        gravity = sensors.gravity or (None, None, None)
        gyro = sensors.angular_velocity or (None, None, None)
        orient = sensors.orientation or (None, None, None, None)
        mag = sensors.magnetic_field or (None, None, None)
        mag_cal = sensors.magnetic_field_calibration or (None, None, None)

        row = [
            timestamp,
            *accel, *gravity, *gyro, *orient, *mag, *mag_cal,
            self.label
        ]
        self.data.append(row)

    async def record_gesture(self, duration=5):
        print("🔄 Searching for watch and starting connection...")
        task = asyncio.create_task(self.run())

        # wait a moment for connection and sensor stream setup
        await asyncio.sleep(2)
        print("📡 Sensors streaming — starting recording...")
        self.recording = True

        await asyncio.sleep(duration)
        self.recording = False
        print("🛑 Recording stopped.")

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.save_data()

    def save_data(self):
        header = [
            "timestamp",
            "accel_x", "accel_y", "accel_z",
            "gravity_x", "gravity_y", "gravity_z",
            "gyro_x", "gyro_y", "gyro_z",
            "orient_x", "orient_y", "orient_z", "orient_w",
            "mag_x", "mag_y", "mag_z",
            "mag_cal_x", "mag_cal_y", "mag_cal_z",
            "label"
        ]

        try:
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(header)
                writer.writerows(self.data)
            print(f"✅ Saved {len(self.data)} samples for '{self.label}' in {self.filename}")
        except Exception as e:
            print("❌ Error saving data:", e)

        self.data = []


if __name__ == "__main__":
    gesture_label = input("Enter gesture name (e.g., wave, tap, pinch): ")
    try:
        duration = int(input("Enter duration to record (seconds): "))
    except ValueError:
        print("Invalid duration; using 5 seconds.")
        duration = 5

    file_name = f"./gesture_data_{gesture_label}.csv"
    recorder = SensorRecorder(label=gesture_label, filename=file_name)

    try:
        asyncio.run(recorder.record_gesture(duration))
    except KeyboardInterrupt:
        print("⛔ Interrupted by user.")
