"""
AI Configuration for AI Construction System
"""
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class AIConfig:
    """AI model and detection configuration"""
    
    # Model Settings
    MODEL_PATH = os.getenv("MODEL_PATH", "../ai/models/yolov8n.pt")
    MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", 0.6))
    MODEL_IOU_THRESHOLD = float(os.getenv("MODEL_IOU_THRESHOLD", 0.5))
    MODEL_INPUT_SIZE = os.getenv("MODEL_INPUT_SIZE", "640x640")
    
    # COCO Dataset Classes (standard YOLOv8n model)
    DETECTION_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush"
    }
    
    # Violation Detection Rules - Maps model outputs to violations
    VIOLATION_RULES = {
        "no_hard_hat": {
            "detection_class": "NO-Hardhat",
            "min_confidence": 0.4,
            "description": "Worker without hard hat detected"
        },
        "no_safety_vest": {
            "detection_class": "NO-Safety Vest", 
            "min_confidence": 0.4,
            "description": "Worker without safety vest detected"
        },
        "no_mask": {
            "detection_class": "NO-Mask",
            "min_confidence": 0.4,
            "description": "Worker without mask detected"
        }
    }
    
    # Alert Settings
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.8))
    ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", 30))  # seconds
    MAX_ALERTS_PER_MINUTE = int(os.getenv("MAX_ALERTS_PER_MINUTE", 10))
    
    # Processing Settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
    MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", 50))
    PROCESSING_INTERVAL = float(os.getenv("PROCESSING_INTERVAL", 0.1))  # seconds
    
    # Hardware Settings
    DEVICE = os.getenv("DEVICE", "cpu")  # cpu, cuda, mps
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
    HALF_PRECISION = os.getenv("HALF_PRECISION", "False").lower() == "true"
    
    # Performance Settings
    ENABLE_TRACKING = os.getenv("ENABLE_TRACKING", "True").lower() == "true"
    TRACKING_MAX_DISAPPEARED = int(os.getenv("TRACKING_MAX_DISAPPEARED", 30))
    TRACKING_MAX_DISTANCE = float(os.getenv("TRACKING_MAX_DISTANCE", 50))
    
    # Quality Settings
    MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", 640))
    MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", 480))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1920))
    
    # Logging Settings
    DETECTION_LOG_FILE = os.getenv("DETECTION_LOG_FILE", "logs/detection.log")
    LOG_DETECTIONS = os.getenv("LOG_DETECTIONS", "True").lower() == "true"
    LOG_CONFIDENCE_THRESHOLD = float(os.getenv("LOG_CONFIDENCE_THRESHOLD", 0.5))
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "model_path": AIConfig.MODEL_PATH,
            "confidence": AIConfig.MODEL_CONFIDENCE,
            "iou_threshold": AIConfig.MODEL_IOU_THRESHOLD,
            "input_size": AIConfig.MODEL_INPUT_SIZE,
            "device": AIConfig.DEVICE,
            "half_precision": AIConfig.HALF_PRECISION
        }
    
    @staticmethod
    def get_detection_config() -> Dict[str, Any]:
        """Get detection configuration"""
        return {
            "classes": AIConfig.DETECTION_CLASSES,
            "violation_rules": AIConfig.VIOLATION_RULES,
            "max_detections": AIConfig.MAX_DETECTIONS,
            "batch_size": AIConfig.BATCH_SIZE,
            "processing_interval": AIConfig.PROCESSING_INTERVAL
        }
    
    @staticmethod
    def get_alert_config() -> Dict[str, Any]:
        """Get alert configuration"""
        return {
            "threshold": AIConfig.ALERT_THRESHOLD,
            "cooldown": AIConfig.ALERT_COOLDOWN,
            "max_alerts_per_minute": AIConfig.MAX_ALERTS_PER_MINUTE
        }
    
    @staticmethod
    def validate_detections(detections: List[Dict]) -> List[Dict]:
        """Validate and filter detections"""
        valid_detections = []
        
        for detection in detections:
            # Check minimum confidence
            if detection.get("confidence", 0) < AIConfig.MODEL_CONFIDENCE:
                continue
            
            # Check valid class
            if detection.get("class_id") not in AIConfig.DETECTION_CLASSES:
                continue
            
            # Check bounding box validity
            bbox = detection.get("bbox", {})
            if not all(key in bbox for key in ["x", "y", "width", "height"]):
                continue
            
            if bbox["width"] <= 0 or bbox["height"] <= 0:
                continue
            
            valid_detections.append(detection)
        
        return valid_detections
    
    @staticmethod
    def generate_mock_detections() -> List[Dict]:
        """Generate mock detections for demo purposes when model is not available"""
        import random
        
        mock_detections = [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": round(random.uniform(0.8, 0.95), 2),
                "bbox": {
                    "x": random.randint(50, 200),
                    "y": random.randint(50, 200),
                    "width": random.randint(80, 150),
                    "height": random.randint(150, 250)
                }
            },
            {
                "class_id": 1,
                "class_name": "hard_hat",
                "confidence": round(random.uniform(0.7, 0.9), 2),
                "bbox": {
                    "x": random.randint(60, 210),
                    "y": random.randint(30, 80),
                    "width": random.randint(40, 60),
                    "height": random.randint(30, 40)
                }
            }
        ]
        
        # Randomly add safety vest detection
        if random.random() > 0.3:  # 70% chance of vest
            mock_detections.append({
                "class_id": 2,
                "class_name": "safety_vest",
                "confidence": round(random.uniform(0.6, 0.85), 2),
                "bbox": {
                    "x": random.randint(70, 180),
                    "y": random.randint(120, 180),
                    "width": random.randint(60, 80),
                    "height": random.randint(70, 90)
                }
            })
        
        return mock_detections
    
    @staticmethod
    def detect_violations(detections: List[Dict]) -> List[Dict]:
        """Detect violations from detections"""
        violations = []
        
        # Group detections by class
        detections_by_class = {}
        for detection in detections:
            class_name = AIConfig.DETECTION_CLASSES.get(detection.get("class_id"))
            if class_name:
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = []
                detections_by_class[class_name].append(detection)
        
        # Check for violations
        persons = detections_by_class.get("person", [])
        hard_hats = detections_by_class.get("hard_hat", [])
        safety_vests = detections_by_class.get("safety_vest", [])
        
        for person in persons:
            person_violations = []
            
            # Check hard hat violation
            if not AIConfig._has_required_equipment(person, hard_hats, "hard_hat"):
                person_violations.append("no_hard_hat")
            
            # Check safety vest violation
            if not AIConfig._has_required_equipment(person, safety_vests, "safety_vest"):
                person_violations.append("no_safety_vest")
            
            # Create violation record
            if person_violations:
                violation = {
                    "person_id": person.get("track_id", len(violations)),
                    "violations": person_violations,
                    "confidence": person.get("confidence", 0),
                    "bbox": person.get("bbox", {}),
                    "timestamp": person.get("timestamp"),
                    "image_path": person.get("image_path")
                }
                
                # Check for multiple violations
                if len(person_violations) >= 2:
                    violation["severity"] = "high"
                    violation["type"] = "multiple_violations"
                else:
                    violation["severity"] = "medium"
                    violation["type"] = person_violations[0]
                
                violations.append(violation)
        
        return violations
    
    @staticmethod
    def _has_required_equipment(person: Dict, equipment: List[Dict], equipment_type: str) -> bool:
        """Check if person has required equipment"""
        if not equipment:
            return False
        
        person_bbox = person.get("bbox", {})
        person_center_x = person_bbox.get("x", 0) + person_bbox.get("width", 0) / 2
        person_center_y = person_bbox.get("y", 0) + person_bbox.get("height", 0) / 2
        
        # Check if any equipment is near the person
        for item in equipment:
            item_bbox = item.get("bbox", {})
            item_center_x = item_bbox.get("x", 0) + item_bbox.get("width", 0) / 2
            item_center_y = item_bbox.get("y", 0) + item_bbox.get("height", 0) / 2
            
            # Calculate distance
            distance = ((person_center_x - item_center_x) ** 2 + 
                       (person_center_y - item_center_y) ** 2) ** 0.5
            
            # Check if equipment is close enough to person
            if distance < AIConfig.TRACKING_MAX_DISTANCE:
                return True
        
        return False
    
    @staticmethod
    def get_performance_metrics() -> Dict[str, Any]:
        """Get performance metrics configuration"""
        return {
            "device": AIConfig.DEVICE,
            "batch_size": AIConfig.BATCH_SIZE,
            "half_precision": AIConfig.HALF_PRECISION,
            "num_workers": AIConfig.NUM_WORKERS,
            "processing_interval": AIConfig.PROCESSING_INTERVAL
        }

# Global AI configuration
ai_config = AIConfig()

if __name__ == "__main__":
    # Test AI configuration
    print("🤖 AI Configuration Test")
    print(f"Model Path: {AIConfig.MODEL_PATH}")
    print(f"Confidence Threshold: {AIConfig.MODEL_CONFIDENCE}")
    print(f"Device: {AIConfig.DEVICE}")
    print(f"Detection Classes: {list(AIConfig.DETECTION_CLASSES.values())}")
    print(f"Violation Rules: {list(AIConfig.VIOLATION_RULES.keys())}")
    
    # Test violation detection with mock data
    mock_detections = [
        {"class_id": 0, "confidence": 0.9, "bbox": {"x": 100, "y": 100, "width": 50, "height": 100}},
        {"class_id": 1, "confidence": 0.8, "bbox": {"x": 95, "y": 80, "width": 20, "height": 20}}
    ]
    
    violations = AIConfig.detect_violations(mock_detections)
    print(f"\n🔍 Test Violations Detected: {len(violations)}")
    for violation in violations:
        print(f"  - {violation.get('type', 'unknown')}: {violation.get('severity', 'medium')}")
