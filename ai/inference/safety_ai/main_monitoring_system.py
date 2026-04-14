import os
import cv2
import csv
import time
import requests
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================
PPE_MODEL_PATH = "models/safety_best.pt"

# Video source
SOURCE = 0

CONFIDENCE_THRESHOLD = 0.40
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Construction Safety Monitoring System"

SAVE_VIOLATION_IMAGES = True
VIOLATION_IMAGE_DIR = "violations"
LOG_FILE = "violations_log.csv"

EVENT_CONFIRMATION_FRAMES = 15
EVENT_COOLDOWN_SECONDS = 10

CAMERA_NAME = "Laptop_Webcam"
CAMERA_IP = "Local_Webcam"

BACKEND_INCIDENT_URL = "http://127.0.0.1:8000/incident"

# =========================================================
# MODEL CLASS DEFINITIONS
# =========================================================
PERSON_CLASSES = {"Human"}

VIOLATION_CLASSES = {
    "no boot",
    "no boots",
    "no gloves",
    "no hat",
    "no vest",
}

CLASS_COLORS = {
    "Human": (0,255,0),

    "Gloves": (255,255,0),
    "Helmet": (255,255,0),
    "Safety Boot": (255,255,0),
    "Safety Vest": (255,255,0),
    "boots": (255,255,0),
    "glasses": (255,255,0),
    "gloves": (255,255,0),
    "hat": (255,255,0),
    "helmet": (255,255,0),
    "vest": (255,255,0),

    "no boot": (0,0,255),
    "no boots": (0,0,255),
    "no gloves": (0,0,255),
    "no hat": (0,0,255),
    "no vest": (0,0,255),
}

# =========================================================
# EVENT MANAGER
# =========================================================
class EventManager:

    def __init__(self, confirmation_frames=15, cooldown_seconds=10):
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        self.violation_frame_counts = defaultdict(int)
        self.last_event_time = defaultdict(float)

    def update(self, current_violations):

        confirmed_events = []
        current_time = time.time()

        for violation_name, count in current_violations.items():

            if count > 0:
                self.violation_frame_counts[violation_name] += 1
            else:
                self.violation_frame_counts[violation_name] = 0

        for violation_name, frame_count in self.violation_frame_counts.items():

            if frame_count >= self.confirmation_frames:

                time_since_last = current_time - self.last_event_time[violation_name]

                if time_since_last >= self.cooldown_seconds:

                    confirmed_events.append(violation_name)

                    self.last_event_time[violation_name] = current_time
                    self.violation_frame_counts[violation_name] = 0

        return confirmed_events

# =========================================================
# SETUP
# =========================================================
def ensure_directories_and_files():

    if SAVE_VIOLATION_IMAGES:
        os.makedirs(VIOLATION_IMAGE_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):

        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "event_id",
                "timestamp",
                "camera_name",
                "camera_ip",
                "camera_source",
                "total_persons",
                "total_violations",
                "violation_type",
                "violation_details",
                "image_path",
                "status",
            ])

def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return YOLO(model_path)

def open_video_source(source):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    return cap

# =========================================================
# HELPERS
# =========================================================
def get_color(label):

    return CLASS_COLORS.get(label,(255,255,255))

def draw_text(frame,text,x,y,color=(255,255,255),scale=0.7,thickness=2):

    cv2.putText(
        frame,
        text,
        (x,y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def save_violation_image(frame):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(
        VIOLATION_IMAGE_DIR,
        f"violation_{timestamp}.jpg"
    )

    cv2.imwrite(image_path,frame)

    return image_path

def append_log(
    event_id,
    total_persons,
    total_violations,
    violation_type,
    violation_summary,
    image_path=""
):

    with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            event_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            CAMERA_NAME,
            CAMERA_IP,
            str(SOURCE),
            total_persons,
            total_violations,
            violation_type,
            str(violation_summary),
            image_path,
            "open",
        ])

def send_incident_to_backend(violation_type,total_persons,image_path=""):

    try:

        payload = {
            "camera_name": CAMERA_NAME,
            "violation_type": violation_type,
            "persons": total_persons,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": image_path,
            "status": "open",
            "camera_ip": CAMERA_IP
        }

        response = requests.post(
            BACKEND_INCIDENT_URL,
            json=payload,
            timeout=5
        )

        print(f"Backend response: {response.status_code}")

    except Exception as e:

        print(f"Backend connection failed: {e}")

# =========================================================
# DETECTION
# =========================================================
def process_detections(frame,results):

    total_persons = 0
    total_violations = 0

    violation_summary = {
        "no boot":0,
        "no boots":0,
        "no gloves":0,
        "no hat":0,
        "no vest":0,
    }

    safe_summary = {
        "Helmet":0,
        "Safety Vest":0,
        "Gloves":0,
        "Safety Boot":0,
    }

    if not results:
        return frame,total_persons,total_violations,violation_summary,safe_summary

    result = results[0]

    if result.boxes is None:
        return frame,total_persons,total_violations,violation_summary,safe_summary

    names = result.names

    for box in result.boxes:

        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if conf < CONFIDENCE_THRESHOLD:
            continue

        label = names.get(cls_id,str(cls_id))

        color = get_color(label)

        x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())

        if label in PERSON_CLASSES:
            total_persons += 1

        if label in VIOLATION_CLASSES:
            total_violations += 1
            violation_summary[label]+=1

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        label_text = f"{label} {conf:.2f}"

        cv2.putText(
            frame,
            label_text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame,total_persons,total_violations,violation_summary,safe_summary

# =========================================================
# MAIN
# =========================================================
def main():

    ensure_directories_and_files()

    model = load_model(PPE_MODEL_PATH)

    cap = open_video_source(SOURCE)

    print("Model loaded successfully.")
    print("Press 'q' to quit.")

    prev_time = time.time()

    event_manager = EventManager(
        confirmation_frames=EVENT_CONFIRMATION_FRAMES,
        cooldown_seconds=EVENT_COOLDOWN_SECONDS
    )

    while True:

        ret,frame = cap.read()

        if not ret:
            break

        results = model(frame,verbose=False)

        frame,total_persons,total_violations,violation_summary,safe_summary = process_detections(frame,results)

        confirmed_events = event_manager.update(violation_summary)

        for violation_type in confirmed_events:

            event_id = f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            image_path=""

            if SAVE_VIOLATION_IMAGES:
                image_path = save_violation_image(frame)

            append_log(
                event_id,
                total_persons,
                total_violations,
                violation_type,
                violation_summary,
                image_path
            )

            send_incident_to_backend(
                violation_type,
                total_persons,
                image_path
            )

        current_time=time.time()
        fps=1/(current_time-prev_time)
        prev_time=current_time

        draw_text(frame,f"FPS: {fps:.2f}",20,30)

        cv2.imshow(WINDOW_NAME,frame)

        if cv2.waitKey(1)&0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()