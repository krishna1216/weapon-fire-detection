import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import simpleaudio as sa
import threading

# ---------------------------------------
# Load YOLO model
# ---------------------------------------
model = YOLO("yolov8n.pt")

# Classes considered dangerous
DANGER_CLASSES = ["gun", "knife", "fire", "smoke", "pistol", "rifle"]

# Global alarm flag
alarm_on = False


# ---------------------------------------
# Alarm Function
# ---------------------------------------
def play_alarm():
    """
    Play alarm sound only once until danger stops.
    Uses WAV format for compatibility.
    """
    global alarm_on
    if alarm_on:
        return

    alarm_on = True
    try:
        wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print("Alarm Error:", e)

    time.sleep(2)
    alarm_on = False


# ---------------------------------------
# Object Detection
# ---------------------------------------
def detect_objects(frame):
    results = model(frame)
    alerts = []
    danger_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            color = (255, 0, 0)  # Red Box

            # Draw object box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Check for dangerous object
            if cls_name.lower() in DANGER_CLASSES:
                alerts.append(f"⚠️ ALERT: {cls_name.upper()} detected!")
                danger_detected = True

    return frame, alerts, danger_detected


# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.set_page_config(page_title="Weapon & Fire Detection", layout="wide")

st.title("🔴 YOLOv8 Real-Time Weapon & Fire Detection System")
st.write("Detects guns, knives, fire, smoke and dangerous activities using YOLOv8.")

start = st.button("Start Detection")
stop = st.button("Stop Detection")

frame_window = st.image([])
log_box = st.empty()

# ---------------------------------------
# Streamlit Webcam Loop
# ---------------------------------------
if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        st.stop()

    st.warning("Webcam started… Press **Stop Detection** to exit.")

    while start and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Error reading webcam.")
            break

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects
        processed_frame, alerts, danger = detect_objects(frame)

        # Update Streamlit window
        frame_window.image(processed_frame)

        # Show alerts
        if alerts:
            log_box.error("\n".join(alerts))
        else:
            log_box.info("No threats detected.")

        # Play alarm only if dangerous object detected
        if danger:
            threading.Thread(target=play_alarm).start()

    cap.release()
    cv2.destroyAllWindows()

if stop:
    st.success("🟢 Detection stopped!")
