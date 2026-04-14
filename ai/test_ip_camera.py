import cv2
from ultralytics import YOLO
import json
import os
import sys
from datetime import datetime
import time

class IPCameraTester:
    """Test YOLO model on IP camera (RTSP)"""
    
    def __init__(self, model_path="ai/models/yolov8n.pt"):
        print(f"🤖 Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Available classes:")
            for class_id, class_name in self.model.names.items():
                print(f"   {class_id}: {class_name}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def test_camera(self, rtsp_url, duration_seconds=60, confidence=0.40):
        """Test model on IP camera feed"""
        if not self.model:
            print("❌ Model not loaded!")
            return False
        
        print(f"\n📹 Connecting to IP Camera: {rtsp_url}")
        print(f"⏱️  Test Duration: {duration_seconds} seconds")
        print(f"🔍 Confidence Threshold: {confidence:.0%}")
        print("-" * 60)
        
        # Connect to camera
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimize buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to connect to camera!")
            print("🔧 Troubleshooting tips:")
            print("   1. Check RTSP URL is correct")
            print("   2. Check camera is online and accessible")
            print("   3. Try adding '?tcp' to URL for stability")
            print("   4. Check firewall allows port 554")
            print("   5. Check network connectivity")
            return False
        
        print("✅ Camera connected successfully!")
        print("-" * 60)
        
        start_time = datetime.now()
        frame_count = 0
        detection_stats = {}
        detections_list = []
        fps_list = []
        
        print("\n🎬 Starting analysis...")
        print("Press 'q' to quit early")
        print("Press 's' to save current frame")
        print("-" * 60)
        
        frame_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️  Camera stream ended or dropped")
                break
            
            # Measure FPS
            frame_time = time.time() - frame_start_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                fps_list.append(fps)
            frame_start_time = time.time()
            
            # Resize for display
            display_frame = cv2.resize(frame, (1024, 768))
            
            # Run inference
            try:
                results = self.model(display_frame, conf=confidence, verbose=False)
            except Exception as e:
                print(f"❌ Inference error: {e}")
                break
            
            # Process detections
            frame_detections = []
            for r in results:
                for box in r.boxes:
                    class_name = r.names[int(box.cls)]
                    confidence_val = float(box.conf)
                    
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence_val
                    })
                    
                    # Update stats
                    if class_name not in detection_stats:
                        detection_stats[class_name] = {
                            'count': 0,
                            'confidences': []
                        }
                    detection_stats[class_name]['count'] += 1
                    detection_stats[class_name]['confidences'].append(confidence_val)
            
            # Record detections
            if frame_detections:
                detections_list.append({
                    'frame': frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'detections': frame_detections
                })
            
            # Draw on frame
            annotated_frame = results[0].plot()
            
            # Add info text
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(frame_detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps_list[-1]:.1f}" if fps_list else "FPS: --", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show detections
            if frame_detections:
                y_offset = 150
                for det in frame_detections:
                    text = f"{det['class']}: {det['confidence']:.1%}"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 35
                    print(f"Frame {frame_count}: {text}")
            
            # Show frame
            cv2.imshow("IP Camera - Model Test (Press 'q' to quit, 's' to save)", annotated_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Test stopped by user")
                break
            elif key == ord('s'):
                filename = f"camera_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✅ Saved: {filename}")
            
            frame_count += 1
            
            # Check if duration exceeded
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration_seconds:
                print(f"\n⏱️  Duration ({duration_seconds}s) reached")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print report
        self._print_report(frame_count, detection_stats, fps_list)
        
        # Save report
        self._save_report(rtsp_url, frame_count, detection_stats, fps_list, detections_list)
        
        return True
    
    def _print_report(self, frame_count, stats, fps_list):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("📊 IP CAMERA TEST REPORT")
        print("="*60)
        print(f"\n⏱️  Total Frames: {frame_count}")
        
        total_detections = sum(s['count'] for s in stats.values())
        print(f"📍 Total Detections: {total_detections}")
        print(f"📈 Avg Detections/Frame: {total_detections/max(frame_count, 1):.2f}")
        
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            print(f"🎬 Avg FPS: {avg_fps:.1f}")
        
        if stats:
            print(f"\n🎯 Detected Classes:")
            print("-" * 60)
            for class_name in sorted(stats.keys()):
                data = stats[class_name]
                count = data['count']
                confs = data['confidences']
                avg_conf = sum(confs) / len(confs)
                min_conf = min(confs)
                max_conf = max(confs)
                
                print(f"\n  {class_name}")
                print(f"    Count: {count}")
                print(f"    Avg Confidence: {avg_conf:.1%}")
                print(f"    Range: {min_conf:.1%} - {max_conf:.1%}")
        else:
            print("\n⚠️  No detections found!")
            print("   This might mean:")
            print("   • No people/violations in camera view")
            print("   • Confidence threshold too high")
            print("   • Poor camera image quality")
        
        print("\n" + "="*60)
    
    def _save_report(self, rtsp_url, frame_count, stats, fps_list, detections):
        """Save test report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'ip_camera',
            'camera_url': rtsp_url,
            'frames_processed': frame_count,
            'total_detections': sum(s['count'] for s in stats.values()),
            'fps': {
                'average': sum(fps_list) / len(fps_list) if fps_list else 0,
                'min': min(fps_list) if fps_list else 0,
                'max': max(fps_list) if fps_list else 0
            },
            'class_statistics': {},
            'detections': detections[:50]  # Keep first 50 for file size
        }
        
        for class_name, data in stats.items():
            confs = data['confidences']
            report['class_statistics'][class_name] = {
                'count': data['count'],
                'avg_confidence': sum(confs) / len(confs),
                'min_confidence': min(confs),
                'max_confidence': max(confs)
            }
        
        # Save JSON
        filename = f"ip_camera_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved: {filename}")


def get_camera_url():
    """Get camera URL from user"""
    print("\n" + "="*60)
    print("🎥 IP CAMERA CONFIGURATION")
    print("="*60)
    
    print("\nCommon camera formats:")
    print("  • RTSP: rtsp://192.168.1.71:554/stream")
    print("  • RTSP TCP: rtsp://192.168.1.71:554/stream?tcp")
    print("  • With auth: rtsp://user:pass@192.168.1.71:554/stream")
    print("  • Hikvision: rtsp://user:pass@192.168.1.71:554/Streaming/Channels/101")
    
    url = input("\n📍 Enter your camera RTSP URL: ").strip()
    
    if not url:
        print("❌ No URL provided!")
        return None
    
    if not url.startswith('rtsp://'):
        print("⚠️  URL doesn't start with 'rtsp://'")
        print("   Assuming RTSP format...")
        if not url.startswith('rtsp://'):
            url = 'rtsp://' + url
    
    return url


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🚀 IP CAMERA AI MODEL TEST")
    print("="*60)
    
    # Check if model exists
    model_path = "ai/models/yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    # Get camera URL
    if len(sys.argv) > 1:
        # From command line
        rtsp_url = sys.argv[1]
        print(f"📍 Using provided URL: {rtsp_url}")
    else:
        # Get from user
        rtsp_url = get_camera_url()
        if not rtsp_url:
            sys.exit(1)
    
    # Get duration
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    # Run test
    tester = IPCameraTester(model_path)
    success = tester.test_camera(rtsp_url, duration_seconds=duration)
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Verify camera is online: ping 192.168.1.71")
        print("2. Check RTSP URL format")
        print("3. Try with ?tcp for stability")
        print("4. Check firewall/network settings")
