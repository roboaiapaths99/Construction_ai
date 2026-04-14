import cv2
from ultralytics import YOLO
import json
import os
import glob
from datetime import datetime

class ImageBatchTester:
    """Test YOLO model on batch of images"""
    
    def __init__(self, model_path="models/yolov8n.pt"):
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
    
    def test_images(self, image_folder="test_images", confidence=0.40):
        """Test model on all images in a folder"""
        if not self.model:
            print("❌ Model not loaded!")
            return
        
        # Find images
        image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(image_folder, pattern)))
        
        if not image_files:
            print(f"❌ No images found in: {image_folder}")
            print(f"   Searched for: {', '.join(image_patterns)}")
            return
        
        print(f"\n📁 Found {len(image_files)} images")
        print(f"🔍 Confidence threshold: {confidence:.0%}")
        print("-" * 60)
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'batch_images',
            'image_folder': image_folder,
            'confidence_threshold': confidence,
            'total_images': len(image_files),
            'results': [],
            'summary': {}
        }
        
        class_stats = {}
        total_detections = 0
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"  ❌ Failed to read image")
                    continue
                
                # Run inference
                results = self.model(img, conf=confidence, verbose=False)
                
                # Extract detections
                detections = []
                for r in results:
                    for box in r.boxes:
                        class_name = r.names[int(box.cls)]
                        confidence_val = float(box.conf)
                        bbox = box.xyxy.tolist()[0]
                        
                        detection = {
                            'class': class_name,
                            'confidence': confidence_val,
                            'bbox': bbox
                        }
                        detections.append(detection)
                        
                        # Update stats
                        if class_name not in class_stats:
                            class_stats[class_name] = {
                                'count': 0,
                                'confidences': [],
                                'images': []
                            }
                        class_stats[class_name]['count'] += 1
                        class_stats[class_name]['confidences'].append(confidence_val)
                        class_stats[class_name]['images'].append(os.path.basename(image_path))
                
                total_detections += len(detections)
                
                # Print detections
                if detections:
                    print(f"  ✅ Found {len(detections)} violation(s):")
                    for det in detections:
                        print(f"     • {det['class']}: {det['confidence']:.1%}")
                else:
                    print(f"  ℹ️  No violations detected")
                
                # Save result
                all_results['results'].append({
                    'image': os.path.basename(image_path),
                    'detection_count': len(detections),
                    'detections': detections
                })
                
                # Save annotated image
                annotated_frame = results[0].plot()
                output_filename = f"annotated_{os.path.basename(image_path)}"
                cv2.imwrite(output_filename, annotated_frame)
                print(f"  💾 Saved annotated: {output_filename}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Print summary
        self._print_summary(len(image_files), total_detections, class_stats)
        
        # Save results
        self._save_results(all_results, class_stats)
    
    def _print_summary(self, total_images, total_detections, class_stats):
        """Print summary report"""
        print("\n" + "="*60)
        print("📊 BATCH TEST SUMMARY")
        print("="*60)
        print(f"\n📁 Images Processed: {total_images}")
        print(f"📍 Total Detections: {total_detections}")
        print(f"📈 Avg Detections/Image: {total_detections/max(total_images, 1):.2f}")
        
        if class_stats:
            print(f"\n🎯 Detected Classes:")
            print("-" * 60)
            for class_name in sorted(class_stats.keys()):
                data = class_stats[class_name]
                count = data['count']
                confs = data['confidences']
                avg_conf = sum(confs) / len(confs)
                min_conf = min(confs)
                max_conf = max(confs)
                images_with_detection = len(set(data['images']))
                
                print(f"\n  {class_name}")
                print(f"    Total: {count} detections")
                print(f"    In {images_with_detection} images")
                print(f"    Avg Confidence: {avg_conf:.1%}")
                print(f"    Range: {min_conf:.1%} - {max_conf:.1%}")
        else:
            print("\n⚠️  No violations detected in any images!")
        
        print("\n" + "="*60)
        print("💡 Annotated images saved as: annotated_*.jpg")
        print("="*60)
    
    def _save_results(self, all_results, class_stats):
        """Save detailed results"""
        # Add summary stats
        all_results['summary'] = {}
        for class_name, data in class_stats.items():
            confs = data['confidences']
            all_results['summary'][class_name] = {
                'total': data['count'],
                'images': len(set(data['images'])),
                'avg_confidence': sum(confs) / len(confs),
                'min_confidence': min(confs),
                'max_confidence': max(confs)
            }
        
        # Save JSON report
        filename = f"batch_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved: {filename}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🚀 BATCH IMAGE TEST")
    print("="*60)
    
    # Check if model exists
    model_path = "models/yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    # Get folder from command line or use default
    image_folder = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    
    print(f"\n📁 Test Images Folder: {image_folder}")
    print("   If you don't have test images, create a 'test_images' folder")
    print("   and add construction site photos there\n")
    
    tester = ImageBatchTester(model_path)
    tester.test_images(image_folder=image_folder)
