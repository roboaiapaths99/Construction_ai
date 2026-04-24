import os
import csv
from datetime import datetime


def ensure_log_file(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_id",
                "timestamp",
                "camera_name",
                "camera_ip",
                "total_persons",
                "total_violations",
                "violation_type",
                "violation_details",
                "image_path",
                "status",
            ])


def append_event_log(
    log_file,
    event_id,
    camera_name,
    camera_ip,
    total_persons,
    total_violations,
    violation_type,
    violation_details,
    image_path,
    status="open",
):
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            event_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            camera_name,
            camera_ip,
            total_persons,
            total_violations,
            violation_type,
            str(violation_details),
            image_path,
            status,
        ])