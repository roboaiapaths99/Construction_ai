"""
AI Worker - Separate AI processing pipeline
Pulls frames from camera RTSP and runs YOLO detection
Sends detection metadata via print (for now, will be WebSocket/REST API)
"""
import cv2
import time
import json
import os
import sys
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch torch.load for weights_only=False compatibility (YOLO model loading)
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# Configuration
RTSP_URL = "rtsp://192.168.1.36?tcp"
MODEL_PATH = "../ai/models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
DETECTION_INTERVAL = 5  # Process AI every N frames

# Detection classes (COCO dataset)
DETECTION_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

class AIWorker:
    def __init__(self):
        print("🤖 AI Worker initializing...")
        
        # Load YOLO model
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ YOLO model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
        
        self.frame_count = 0
        self.clients = set()
        self.running = True
        
    def process_frame(self, frame):
        """Run AI detection on a frame and return detection metadata"""
        if self.model is None:
            return None
        
        try:
            results = self.model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = DETECTION_CLASSES.get(class_id, f'unknown_{class_id}')
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        })
            
            return {
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'frame_count': self.frame_count
            }
            
        except Exception as e:
            print(f"❌ AI detection error: {e}")
            return None
    
    def broadcast_detections(self, detection_data):
        """Send detection metadata (simplified version - just print for now)"""
        print(f"📡 Detection data: {json.dumps(detection_data)}")
        # In production, this would send to WebSocket, Redis, or message queue
    
    def run_ai_processing(self):
        """Main AI processing loop - pulls frames from MediaMTX RTSP"""
        print(f"🎥 Connecting to MediaMTX RTSP: {RTSP_URL}")
        
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"❌ Failed to connect to RTSP stream")
            return
        
        print(f"✅ Connected to RTSP stream")
        
        while self.running:
            success, frame = cap.read()
            if not success:
                print("⚠️  Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            self.frame_count += 1
            
            # Run AI detection on frame (only every N frames for performance)
            if self.frame_count % DETECTION_INTERVAL == 0:
                detection_data = self.process_frame(frame)
                if detection_data:
                    # Broadcast detection data
                    self.broadcast_detections(detection_data)
                    
                    # Log detections
                    if detection_data['detections']:
                        print(f"🔍 Detected {len(detection_data['detections'])} objects")
        
        cap.release()
        print("🛑 AI Worker stopped")
    
    def start(self):
        """Start the AI worker"""
        print("🚀 Starting AI Worker...")
        self.run_ai_processing()

if __name__ == "__main__":
    worker = AIWorker()
    worker.start()
