import cv2
from ultralytics import YOLO
import json
import os
from datetime import datetime

class ModelTester:
    """Test YOLO model detection"""
    
    def __init__(self, model_path="models/yolov8n.pt"):
        print(f"🤖 Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def test_webcam(self, duration_seconds=30):
        """Test model with webcam feed"""
        if not self.model:
            print("❌ Model not loaded!")
            return
        
        print(f"\n🎥 Starting webcam test for {duration_seconds} seconds...")
        print("Press 'q' to quit early")
        print("Press 's' to save a detection frame")
        print("-" * 60)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        start_time = datetime.now()
        frame_count = 0
        detection_stats = {}
        detections_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for display
            display_frame = cv2.resize(frame, (800, 600))
            
            # Run inference
            results = self.model(display_frame, conf=0.40, verbose=False)
            
            # Process detections
            frame_detections = []
            for r in results:
                for box in r.boxes:
                    class_name = r.names[int(box.cls)]
                    confidence = float(box.conf)
                    
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
                    
                    # Update stats
                    if class_name not in detection_stats:
                        detection_stats[class_name] = {
                            'count': 0,
                            'confidences': []
                        }
                    detection_stats[class_name]['count'] += 1
                    detection_stats[class_name]['confidences'].append(confidence)
            
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
            
            if frame_detections:
                y_offset = 110
                for det in frame_detections:
                    text = f"{det['class']}: {det['confidence']:.1%}"
                    cv2.putText(annotated_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 35
            
            # Show frame
            cv2.imshow("Model Test - Press 'q' to quit, 's' to save", annotated_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Test stopped by user")
                break
            elif key == ord('s'):
                filename = f"detection_frame_{frame_count}.jpg"
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
        self._print_report(frame_count, detection_stats, detections_list)
        
        # Save report
        self._save_report(frame_count, detection_stats, detections_list)
    
    def _print_report(self, frame_count, stats, detections):
        """Print formatted test report"""
        print("\n" + "="*60)
        print("📊 MODEL TEST REPORT")
        print("="*60)
        print(f"\n⏱️  Total Frames: {frame_count}")
        
        total_detections = sum(s['count'] for s in stats.values())
        print(f"📍 Total Detections: {total_detections}")
        print(f"📈 Avg Detections/Frame: {total_detections/max(frame_count, 1):.2f}")
        
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
            print("   • No people in frame")
            print("   • Confidence threshold too high")
            print("   • Model needs retraining")
        
        print("\n" + "="*60)
    
    def _save_report(self, frame_count, stats, detections):
        """Save test report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'webcam',
            'frames_processed': frame_count,
            'total_detections': sum(s['count'] for s in stats.values()),
            'class_statistics': {}
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
        filename = f"model_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved: {filename}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🚀 AI MODEL DETECTION TEST")
    print("="*60)
    
    # Check if model exists
    model_path = "models/yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please check the path and try again")
        sys.exit(1)
    
    # Run test
    tester = ModelTester(model_path)
    
    # Get duration from command line or use default
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    tester.test_webcam(duration_seconds=duration)
