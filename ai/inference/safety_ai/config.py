PPE_MODEL_PATH = "models/safety_best.pt"
SOURCE = 0  # laptop webcam

CONFIDENCE_THRESHOLD = 0.40
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "AI Construction Safety Monitoring System"

SAVE_VIOLATION_IMAGES = True
VIOLATION_IMAGE_DIR = "violations"
LOG_DIR = "logs"
LOG_FILE = "logs/violations_log.csv"

SAVE_INTERVAL_SECONDS = 10
EVENT_CONFIRMATION_FRAMES = 15   # around half second to 1 sec depending on fps
EVENT_COOLDOWN_SECONDS = 10

CAMERA_NAME = "Laptop_Webcam"
CAMERA_IP = "Local_Webcam"