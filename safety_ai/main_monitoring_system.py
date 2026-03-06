import cv2
from ultralytics import YOLO
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import csv
from datetime import datetime
import datetime as dt

# ---------------- LOAD MODELS ----------------

ppe_model = YOLO("models/safety_best.pt")
phone_model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=30)

workers_path = "workers"
threshold = 0.65

cap = cv2.VideoCapture(0)

marked_today = set()

os.makedirs("alerts", exist_ok=True)

# ---------------- CREATE CSV ----------------

if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name","Date","Time"])

# ---------------- ATTENDANCE FUNCTION ----------------

def mark_attendance(name):

    global marked_today

    if name in marked_today:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    with open("attendance.csv","a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name,today,time])

    marked_today.add(name)

    print("Attendance stored:", name)

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    # -------- Define Danger Zone --------

    zone_x1 = int(width * 0.6)
    zone_y1 = int(height * 0.5)

    zone_x2 = width
    zone_y2 = height

    cv2.rectangle(frame,(zone_x1,zone_y1),(zone_x2,zone_y2),(0,0,255),2)
    cv2.putText(frame,"DANGER ZONE",(zone_x1,zone_y1-10),
    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    # ---------------- PPE DETECTION ----------------

    ppe_results = ppe_model(frame)

    annotated_frame = ppe_results[0].plot()

    boxes = ppe_results[0].boxes
    names = ppe_model.names

    detected_objects = []
    detections = []

    if boxes is not None:

        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):

            label = names[int(cls)]

            x1,y1,x2,y2 = map(int,box)

            w = x2-x1
            h = y2-y1

            detected_objects.append(label)

            if label == "Human":
                detections.append(([x1,y1,w,h],conf,label))

    # ---------------- TRACKING ----------------

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l,t,r,b = map(int,track.to_ltrb())

        cv2.rectangle(annotated_frame,(l,t),(r,b),(0,255,255),2)

        cv2.putText(
            annotated_frame,
            f"Worker {track_id}",
            (l,t-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        # -------- Danger Zone Check --------

        center_x = int((l+r)/2)
        center_y = int((t+b)/2)

        if zone_x1 < center_x < zone_x2 and zone_y1 < center_y < zone_y2:

            print("⚠ DANGER ZONE VIOLATION")

            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"alerts/danger_zone_{timestamp}.jpg"

            cv2.imwrite(filename, frame)

    # ---------------- PHONE DETECTION ----------------

    phone_results = phone_model(frame)

    phone_boxes = phone_results[0].boxes
    phone_names = phone_model.names

    phone_detected = False

    if phone_boxes is not None:

        for box, cls in zip(phone_boxes.xyxy, phone_boxes.cls):

            label = phone_names[int(cls)]

            if label == "cell phone":

                phone_detected = True

                x1,y1,x2,y2 = map(int,box)

                cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,0,255),2)

                cv2.putText(
                    annotated_frame,
                    "PHONE",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2
                )

    # ---------------- FACE RECOGNITION ----------------

    try:

        results = DeepFace.find(
            img_path=frame,
            db_path=workers_path,
            enforce_detection=False,
            silent=True
        )

        if len(results)>0 and len(results[0])>0:

            best_match = results[0].iloc[0]

            distance = best_match["distance"]

            identity = best_match["identity"]

            name = os.path.basename(identity).split(".")[0]

            if distance < threshold:

                cv2.putText(
                    annotated_frame,
                    name,
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                mark_attendance(name)

    except:
        pass

    # ---------------- EMOTION ----------------

    try:

        analysis = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = analysis[0]["dominant_emotion"]

        cv2.putText(
            annotated_frame,
            f"Emotion: {emotion}",
            (50,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

    except:
        pass

    cv2.imshow("Construction AI Monitoring System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()