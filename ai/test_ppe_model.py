"""
PPE Detection Model Test - Construction Safety
Tests Hansung-Cho/yolov8-ppe-detection model on IP camera
"""
import cv2
from ultralytics import YOLO
import json
import os
import sys
from datetime import datetime
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

class PPEModelTester:
    """Test PPE detection model on IP camera (RTSP)"""
    
    def __init__(self, model_source="Hansung-Cho/yolov8-ppe-detection"):
        print(f"🤖 Loading PPE model: {model_source}")
        try:
            # Download model from HuggingFace Hub
            print("📥 Downloading from HuggingFace Hub...")
            model_path = hf_hub_download(
                repo_id=model_source,
                filename="best.pt",
                cache_dir="./models"
            )
            print(f"✅ Model downloaded to: {model_path}")
            
            # Load PPE detection model
            self.model = YOLO(model_path)
            print(f"✅ PPE Model loaded successfully!")
            print(f"📊 Available classes:")
            for class_id, class_name in self.model.names.items():
                print(f"   {class_id}: {class_name}")
        except Exception as e:
            print(f"❌ Failed to load PPE model: {e}")
            self.model = None
    
    def test_camera(self, rtsp_url, duration_seconds=60, confidence=0.40):
        """Test PPE model on IP camera feed"""
        if not self.model:
            print("❌ Model not loaded!")
            return False
        
        print(f"\n📹 Connecting to IP Camera: {rtsp_url}")
        print(f"⏱️  Test Duration: {duration_seconds} seconds")
        print(f"🔍 Confidence Threshold: {confidence*100:.0f}%")
        print("-" * 60)
        
        # Open video capture
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera")
            return False
        
        print("✅ Camera connected successfully!")
        print("-" * 60)
        print("\n🎬 Starting PPE detection analysis...")
        print("Press 'q' to quit early")
        print("Press 's' to save current frame")
        print("-" * 60)
        
        # Statistics
        frame_count = 0
        start_time = time.time()
        detections_log = []
        ppe_violations = {
            "no_hardhat": 0,
            "no_vest": 0,
            "no_gloves": 0,
            "no_mask": 0
        }
        fps_list = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_time_start = time.time()
                results = self.model(frame, conf=confidence, imgsz=640, verbose=False)
                frame_time_end = time.time()
                fps = 1.0 / (frame_time_end - frame_time_start) if (frame_time_end - frame_time_start) > 0 else 0
                fps_list.append(fps)
                
                # Prepare frame for display
                display_frame = frame.copy()
                
                # Extract detections
                if results[0].boxes is not None:
                    detections_in_frame = []
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence_score = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0].tolist()]
                        
                        # Determine color based on violation type
                        if "NO-" in class_name:
                            color = (0, 0, 255)  # Red for violations
                            thickness = 3
                        else:
                            color = (0, 255, 0)  # Green for safe
                            thickness = 2
                        
                        # Draw bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add label
                        label = f"{class_name}: {confidence_score:.0%}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        label_y = y1 - 10 if y1 > 30 else y2 + 20
                        cv2.rectangle(display_frame, (x1, label_y - label_size[1] - 5), 
                                    (x1 + label_size[0] + 5, label_y + 5), color, -1)
                        cv2.putText(display_frame, label, (x1 + 2, label_y - 2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        detections_in_frame.append({
                            "class": class_name,
                            "confidence": confidence_score,
                            "bbox": [float(x) for x in box.xyxy[0].tolist()]
                        })
                        
                        # Track violations
                        if class_name == "NO-Hardhat":
                            ppe_violations["no_hardhat"] += 1
                        elif class_name == "NO-Safety Vest":
                            ppe_violations["no_vest"] += 1
                        elif class_name == "NO-Gloves":
                            ppe_violations["no_gloves"] += 1
                        elif class_name == "NO-Mask":
                            ppe_violations["no_mask"] += 1
                        
                        print(f"Frame {frame_count}: {class_name}: {confidence_score:.1%}")
                    
                    if detections_in_frame:
                        detections_log.append({
                            "frame": frame_count,
                            "detections": detections_in_frame
                        })
                
                # Add FPS counter to frame
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("🎬 PPE Detection - Press 'q' to quit, 's' to save", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⛔ User quit")
                    break
                elif key == ord('s'):
                    # Save frame
                    save_path = f"ppe_detection_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, display_frame)
                    print(f"💾 Frame saved: {save_path}")
                
                frame_count += 1
                
                # Check duration
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    print(f"\n⏱️  Duration ({duration_seconds}s) reached")
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Generate report
        self._print_report(frame_count, len(detections_log), fps_list, ppe_violations, detections_log)
        self._save_report(frame_count, len(detections_log), fps_list, ppe_violations, detections_log)
        
        return True
    
    def _print_report(self, total_frames, total_detections, fps_list, violations, detections_log):
        """Print test report to console"""
        print("\n" + "=" * 60)
        print("📊 PPE DETECTION TEST REPORT")
        print("=" * 60)
        
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"\n⏱️  Total Frames: {total_frames}")
        print(f"📍 Total Detections: {total_detections}")
        print(f"📈 Avg Detections/Frame: {total_detections/max(total_frames, 1):.2f}")
        print(f"🎬 Avg FPS: {avg_fps:.1f}")
        
        print("\n🚨 PPE VIOLATIONS DETECTED:")
        print("-" * 60)
        for violation, count in violations.items():
            if count > 0:
                print(f"  ⚠️  {violation.replace('_', ' ').upper()}: {count} incidents")
        
        if sum(violations.values()) == 0:
            print("  ✅ No PPE violations detected!")
        
        print("\n🎯 Detection Details:")
        print("-" * 60)
        
        # Aggregate detections by class
        class_stats = {}
        for det_frame in detections_log:
            for detection in det_frame["detections"]:
                class_name = detection["class"]
                confidence = detection["confidence"]
                
                if class_name not in class_stats:
                    class_stats[class_name] = {
                        "count": 0,
                        "confidences": [],
                        "min": 1.0,
                        "max": 0.0
                    }
                
                class_stats[class_name]["count"] += 1
                class_stats[class_name]["confidences"].append(confidence)
                class_stats[class_name]["min"] = min(class_stats[class_name]["min"], confidence)
                class_stats[class_name]["max"] = max(class_stats[class_name]["max"], confidence)
        
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            avg_conf = sum(stats["confidences"]) / len(stats["confidences"])
            print(f"\n  {class_name}")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Confidence: {avg_conf:.1%}")
            print(f"    Range: {stats['min']:.1%} - {stats['max']:.1%}")
        
        print("\n" + "=" * 60)
    
    def _save_report(self, total_frames, total_detections, fps_list, violations, detections_log):
        """Save report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"ppe_detection_test_{timestamp}.json"
        
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        
        report = {
            "test_type": "PPE Detection",
            "timestamp": timestamp,
            "model": "Hansung-Cho/yolov8-ppe-detection",
            "total_frames": total_frames,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / max(total_frames, 1),
            "avg_fps": round(avg_fps, 2),
            "violations": violations,
            "detections_log": detections_log
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved: {report_path}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 PPE DETECTION MODEL TEST")
    print("=" * 60)
    
    # Get camera URL
    if len(sys.argv) > 1:
        # From command line
        rtsp_url = sys.argv[1]
        print(f"📍 Using provided URL: {rtsp_url}")
    else:
        # Interactive mode
        print("\nNo URL provided. Using default:")
        rtsp_url = "rtsp://192.168.1.71:554/11?tcp"
        print(f"📍 URL: {rtsp_url}")
        print("\nAlternatively, run with: python test_ppe_model.py <RTSP_URL> [DURATION]")
    
    # Get duration
    duration = 60
    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
            print(f"⏱️  Duration: {duration} seconds")
        except ValueError:
            print(f"⏱️  Duration: {duration} seconds (default)")
    
    # Run test
    tester = PPEModelTester()
    if tester.model:
        tester.test_camera(rtsp_url, duration)
        print("✅ Test completed successfully!")
    else:
        print("❌ Failed to initialize model")
        sys.exit(1)
