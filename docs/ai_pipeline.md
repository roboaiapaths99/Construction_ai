# AI Pipeline Documentation

## Overview
The AI Pipeline is the core component responsible for real-time object detection, violation recognition, and alert generation in the construction safety monitoring system.

## Pipeline Architecture Diagram

```
Input Frame → Preprocessing → YOLO Detection → Post-processing → Violation Analysis → Alert Generation
     │              │              │               │                │                   │
Video Stream   Resizing &     Object          Bounding        Rule-based          Severity
Webcam Feed    Normalizing   Detection       Box Filtering   Evaluation           Classification
               Format Conv.  (YOLOv8)        Confidence      Spatial              Notification
                              Classes         Threshold       Analysis             WebSocket
```

## Pipeline Stages

### 1. Input Processing Stage
```
Input Sources:
├── Webcam Feed (Real-time)
├── Video Files (Batch Processing)
├── Image Files (Single Frame)
└── API Upload (Base64 Images)

Processing Steps:
1. Frame capture (30 FPS)
2. Resolution validation (min 640x480)
3. Format conversion (RGB)
4. Quality check (blur detection)
```

### 2. Preprocessing Stage
```
Preprocessing Pipeline:
├── Resize to 640x640 (YOLO input size)
├── Normalization (0-1 pixel values)
├── Color space conversion (RGB)
├── Batch preparation (batch_size=1)
└── Device transfer (CPU/GPU)

Code Example:
```python
def preprocess_frame(frame):
    # Resize to model input size
    resized = cv2.resize(frame, (640, 640))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    # Convert to tensor format
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)
```

### 3. YOLO Detection Stage
```
YOLOv8 Model Configuration:
├── Model: yolov8n.pt (Nano version)
├── Input Size: 640x640 pixels
├── Classes: 3 (person, hard_hat, safety_vest)
├── Confidence Threshold: 0.4
├── IoU Threshold: 0.5
└── Device: CPU (upgradable to GPU)

Detection Process:
1. Forward pass through neural network
2. Non-maximum suppression (NMS)
3. Bounding box generation
4. Class probability calculation
5. Confidence scoring
```

### 4. Post-processing Stage
```
Post-processing Pipeline:
├── Bounding box validation
│   ├── Size filtering (min 20x20 pixels)
│   ├── Aspect ratio check
│   └── Boundary validation
├── Confidence filtering (> 0.4)
├── Class mapping (ID → Name)
├── Duplicate removal (IoU > 0.5)
└── Tracking assignment (if enabled)

Output Format:
```python
detections = [
    {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.95,
        "bbox": {
            "x": 100, "y": 100,
            "width": 200, "height": 300
        },
        "center": {"x": 200, "y": 250}
    }
]
```

### 5. Violation Analysis Stage
```
Violation Detection Rules:

1. No Hard Hat Violation:
   - Person detected
   - No hard hat in proximity (< 50 pixels)
   - Confidence > 0.7

2. No Safety Vest Violation:
   - Person detected
   - No safety vest in proximity (< 50 pixels)
   - Confidence > 0.6

3. Multiple Violations:
   - Person missing both equipment
   - High severity alert

Spatial Analysis Algorithm:
```python
def detect_violations(detections):
    violations = []
    persons = [d for d in detections if d["class_name"] == "person"]
    hard_hats = [d for d in detections if d["class_name"] == "hard_hat"]
    safety_vests = [d for d in detections if d["class_name"] == "safety_vest"]
    
    for person in persons:
        person_violations = []
        
        # Check hard hat
        if not has_equipment(person, hard_hats, 50):
            person_violations.append("no_hard_hat")
        
        # Check safety vest
        if not has_equipment(person, safety_vests, 50):
            person_violations.append("no_safety_vest")
        
        if person_violations:
            violations.append({
                "person_bbox": person["bbox"],
                "violations": person_violations,
                "severity": "high" if len(person_violations) > 1 else "medium"
            })
    
    return violations
```

### 6. Alert Generation Stage
```
Alert Classification:
├── Low Severity: Informational
├── Medium Severity: Single violation
├── High Severity: Multiple violations
└── Critical: Repeated violations

Alert Generation Rules:
1. Cooldown period (30 seconds per person)
2. Rate limiting (max 10 alerts/minute)
3. Severity-based routing
4. Real-time WebSocket broadcast

Alert Format:
```python
alert = {
    "id": generate_alert_id(),
    "type": "safety_violation",
    "severity": "medium",
    "message": "Worker without hard hat detected",
    "camera": "Main Entrance",
    "location": {"x": 100, "y": 100},
    "confidence": 0.95,
    "timestamp": datetime.utcnow(),
    "image_path": save_violation_image(frame, bbox)
}
```

## Performance Optimization

### Processing Optimization
```
Optimization Techniques:
├── Frame skipping (process every 2nd frame)
├── Region of interest (ROI) processing
├── Multi-threading (detection + I/O)
├── Model quantization (FP16)
└── Batch processing (when possible)

Performance Metrics:
├── Inference time: ~45ms per frame
├── Throughput: ~22 FPS (single thread)
├── Memory usage: ~500MB (CPU)
└── CPU usage: ~60% (single core)
```

### Memory Management
```
Memory Optimization:
├── Frame buffer management (circular buffer)
├── Model loading (single instance)
├── Tensor caching (reusable tensors)
└── Garbage collection (periodic cleanup)

Memory Usage Breakdown:
├── YOLO Model: ~6MB (disk), ~50MB (RAM)
├── Frame Buffer: ~10MB (3 frames)
├── Detection Cache: ~5MB
└── System Overhead: ~20MB
```

## Error Handling & Recovery

### Error Scenarios
```
Common Errors:
1. Model Loading Failure
   - Fallback to mock detection
   - Log error and alert admin
   
2. Invalid Frame Input
   - Skip frame and continue
   - Log frame metadata
   
3. Detection Timeout
   - Reduce confidence threshold
   - Skip current frame
   
4. Memory Overflow
   - Clear frame buffer
   - Reduce processing frequency
```

### Recovery Strategies
```
Automatic Recovery:
├── Model reload on failure
├── Graceful degradation (reduced accuracy)
├── Service restart (if critical)
└── Fallback to previous frame

Manual Intervention:
├── Admin notification system
├── Debug mode activation
├── Manual model reload
└── Configuration adjustment
```

## Quality Assurance

### Validation Metrics
```
Detection Quality:
├── Precision: >90% (hard hat detection)
├── Recall: >85% (person detection)
├── F1-Score: >87%
└── False Positive Rate: <5%

Performance Quality:
├── Latency: <100ms per frame
├── Throughput: >15 FPS
├── Availability: >99%
└── Error Rate: <1%
```

### Testing Pipeline
```
Test Categories:
├── Unit Tests (individual functions)
├── Integration Tests (pipeline stages)
├── Performance Tests (load testing)
└── Accuracy Tests (model validation)

Test Data:
├── Synthetic images (controlled scenarios)
├── Real construction site images
├── Edge cases (lighting, angles)
└── Negative samples (no violations)
```

## Configuration Management

### Model Configuration
```python
AI_CONFIG = {
    "model_path": "../ai/models/yolov8n.pt",
    "confidence_threshold": 0.4,
    "iou_threshold": 0.5,
    "input_size": [640, 640],
    "device": "cpu",
    "classes": {
        0: "person",
        1: "hard_hat",
        2: "safety_vest"
    }
}
```

### Pipeline Configuration
```python
PIPELINE_CONFIG = {
    "processing_interval": 0.1,  # seconds
    "max_detections": 50,
    "violation_distance": 50,    # pixels
    "alert_cooldown": 30,        # seconds
    "max_alerts_per_minute": 10,
    "tracking_enabled": True,
    "save_violation_images": True
}
```

## Monitoring & Analytics

### Pipeline Metrics
```
Real-time Metrics:
├── Frames processed per second
├── Detection confidence distribution
├── Violation detection rate
├── Alert generation frequency
├── Processing latency histogram
└── Error rate tracking

Historical Analytics:
├── Violation trends over time
├── Peak violation periods
├── Camera effectiveness analysis
├── Model accuracy drift
└── System performance trends
```

### Logging Strategy
```
Log Levels:
├── DEBUG: Detailed processing info
├── INFO: General pipeline status
├── WARNING: Non-critical issues
├── ERROR: Pipeline failures
└── CRITICAL: System outages

Log Content:
├── Frame metadata (timestamp, source)
├── Detection results (count, confidence)
├── Violation details (type, location)
├── Performance metrics (timing, memory)
└── Error information (type, context)
```

## Future Enhancements

### Advanced Features
```
Planned Improvements:
├── Multi-camera fusion
├── 3D scene understanding
├── Behavior analysis (anomaly detection)
├── Predictive safety analytics
├── Automated model retraining
└── Edge deployment optimization
```

### Technology Upgrades
```
Next-Generation Pipeline:
├── YOLOv9 integration (higher accuracy)
├── Transformer-based detection
├── Real-time video analytics
├── Cloud-based processing
├── Federated learning
└── Explainable AI (XAI)
```

---

**Document Version**: 1.0.0  
**Last Updated**: March 2026  
**Pipeline Version**: Current Implementation
