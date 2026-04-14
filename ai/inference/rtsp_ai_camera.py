import cv2
import numpy as np
import time
import requests
import base64
from ultralytics import YOLO
import json

# =========================================================
# RTSP AI CAMERA SYSTEM
# =========================================================

# Camera Configuration
RTSP_URL = "rtsp://192.168.1.71:554/11?tcp"  # Your working URL with TCP protocol
BUFFER_SIZE = 1  # IMPORTANT: Reduces lag
TCP_PROTO = "tcp"  # IMPORTANT: More stable connection

# AI Model Configuration
MODEL_PATH = "../models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4
DISPLAY_SIZE = (640, 480)  # IMPORTANT: Resize for performance

# API Configuration
BACKEND_URL = "http://localhost:8001/detect_base64"

class RTSPAICamera:
    def __init__(self):
        self.cap = None
        self.model = None
        self.frame_count = 0
        self.detection_count = 0
        
    def connect_camera(self):
        """Connect to RTSP camera with optimized settings"""
        try:
            print(f"🎥 Connecting to RTSP camera: {RTSP_URL}")
            
            # Create video capture with optimized settings
            self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            
            # CRITICAL: Set buffer size to reduce lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
            
            # Set TCP protocol for stability
            if "?" in RTSP_URL:
                rtsp_url_with_tcp = RTSP_URL.replace("rtsp://", "rtsp://") + f"?{TCP_PROTO}"
                self.cap = cv2.VideoCapture(rtsp_url_with_tcp, cv2.CAP_FFMPEG)
            
            if self.cap.isOpened():
                print("✅ RTSP Camera Connected Successfully!")
                print(f"📊 Buffer Size: {BUFFER_SIZE}")
                print(f"🌐 Protocol: {TCP_PROTO}")
                print(f"📱 Display Size: {DISPLAY_SIZE}")
                return True
            else:
                print("❌ Failed to open RTSP camera")
                return False
                
        except Exception as e:
            print(f"❌ Camera connection error: {e}")
            return False
    
    def load_ai_model(self):
        """Load YOLO model for detection"""
        try:
            print(f"🤖 Loading AI model: {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            print("✅ AI Model Loaded Successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            return False
    
    def send_to_backend(self, frame):
        """Send frame to backend for AI processing"""
        try:
            # Resize frame for performance
            resized_frame = cv2.resize(frame, DISPLAY_SIZE)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare API request
            data = {
                "image": f"data:image/jpeg;base64,{frame_base64}"
            }
            
            # Send to backend
            response = requests.post(
                BACKEND_URL, 
                json=data, 
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.detection_count += 1
                
                # Display results
                if result.get('success') and result.get('detections'):
                    print(f"🎯 Detections: {len(result['detections'])}")
                    for detection in result['detections']:
                        class_name = detection.get('class_name', 'unknown')
                        confidence = detection.get('confidence', 0)
                        print(f"  📦 {class_name}: {confidence:.2f}")
                
                if result.get('violations'):
                    print(f"⚠️ Violations: {len(result['violations'])}")
                    for violation in result['violations']:
                        violation_type = violation.get('type', 'unknown')
                        print(f"  🚨 {violation_type}")
                
                return result
            else:
                print(f"❌ Backend error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def draw_detections(self, frame, api_result):
        """Draw detection boxes on frame"""
        if not api_result or not api_result.get('detections'):
            return frame
        
        result_frame = frame.copy()
        
        for detection in api_result.get('detections', []):
            bbox = detection.get('bbox', {})
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0)
            
            if bbox and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                
                # Choose color based on class
                if 'person' in class_name.lower():
                    color = (0, 255, 0)  # Green
                elif 'hard_hat' in class_name.lower():
                    color = (0, 215, 255)  # Gold
                elif 'safety_vest' in class_name.lower():
                    color = (255, 105, 180)  # Pink
                else:
                    color = (255, 255, 255)  # White
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{class_name.upper()} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_frame, (x, y - 25), (x + label_size[0], y), color, -1)
                cv2.putText(result_frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw violations
        for violation in api_result.get('violations', []):
            bbox = violation.get('bbox', {})
            if bbox and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
                
                # Red violation box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Violation label
                violation_text = f"VIOLATION: {violation.get('type', 'UNKNOWN')}"
                cv2.putText(result_frame, violation_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result_frame
    
    def run(self):
        """Main camera loop with AI detection"""
        print("🚀 Starting RTSP AI Camera System...")
        
        # Connect to camera
        if not self.connect_camera():
            return
        
        # Load AI model
        if not self.load_ai_model():
            return
        
        print("🎮 Starting camera loop (Press 'q' to quit)")
        print(f"📊 Frame processing size: {DISPLAY_SIZE}")
        
        while True:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ No frame received from camera")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Resize frame for performance
                resized_frame = cv2.resize(frame, DISPLAY_SIZE)
                
                # Send to backend for AI processing every 5 frames to reduce API calls
                if self.frame_count % 5 == 0:
                    print(f"🤖 Processing frame {self.frame_count}...")
                    api_result = self.send_to_backend(frame)
                    
                    if api_result:
                        # Draw detections on frame
                        display_frame = self.draw_detections(resized_frame, api_result)
                    else:
                        display_frame = resized_frame
                else:
                    display_frame = resized_frame
                
                # Show frame
                cv2.imshow("Construction AI Camera - RTSP", display_frame)
                
                # Add frame counter
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add detection counter
                cv2.putText(display_frame, f"Detections: {self.detection_count}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 User interrupted")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(0.1)
                continue
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("=" * 60)
    print("🏗️ AI CONSTRUCTION SAFETY SYSTEM - RTSP CAMERA")
    print("=" * 60)
    print(f"📹 Camera URL: {RTSP_URL}")
    print(f"🤖 Backend API: {BACKEND_URL}")
    print(f"📱 Display Size: {DISPLAY_SIZE}")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    
    camera = RTSPAICamera()
    camera.run()

if __name__ == "__main__":
    main()
