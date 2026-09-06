"""
config.py
================================================================================
Centralized configuration parameters, alert thresholds, Twilio credentials,
and spatiotemporal anomaly classification constants.
================================================================================
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# ==============================================================================
# BASE DIRECTORIES
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
CAPTURES_DIR: Path = BASE_DIR / "captures"
TEST_CLIPS_DIR: Path = BASE_DIR / "test_clips"

# Ensure runtime directories exist
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
TEST_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CAMERA & NODE GEOLOCATION (DYNAMIC LIVE DETECTION)
# ==============================================================================
CAMERA_ID: str = os.getenv("CAMERA_ID", "Surveillance Node 01")

# High-precision verified geographical coordinates for Rajanukunte, Bangalore (560064)
CAMERA_LATITUDE: float = float(os.getenv("CAMERA_LATITUDE", "13.1870"))
CAMERA_LONGITUDE: float = float(os.getenv("CAMERA_LONGITUDE", "77.5502"))
CAMERA_LOCATION_NAME: str = os.getenv("CAMERA_LOCATION_NAME", "Rajanukunte, Karnataka (560064)")
CAMERA_LAT_LON: Tuple[float, float] = (CAMERA_LATITUDE, CAMERA_LONGITUDE)

# Whether to mention the exact location name text in WhatsApp messages (False by default per user request)
WHATSAPP_INCLUDE_EXACT_LOCATION_NAME: bool = os.getenv("WHATSAPP_INCLUDE_EXACT_LOCATION_NAME", "false").lower() in ("true", "1", "yes")

# ==============================================================================
# COMPUTER VISION & TRACKING SETTINGS
# ==============================================================================
# Lightweight YOLO Pose model for real-time inference (17 COCO keypoints)
YOLO_MODEL_NAME: str = os.getenv("YOLO_MODEL_NAME", "yolov8n-pose.pt")

# Detection confidence score threshold for person bounding boxes
DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))

# Tracker configuration (bytetrack.yaml or botsort.yaml)
TRACKER_CONFIG: str = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")

# Minimum IoU overlap between bounding boxes to trigger interaction zone
INTERACTION_IOU_THRESHOLD: float = float(os.getenv("INTERACTION_IOU_THRESHOLD", "0.02"))

# Normalized or pixel proximity threshold between pedestrian centroids
INTERACTION_PROXIMITY_PX: float = float(os.getenv("INTERACTION_PROXIMITY_PX", "380.0"))

# ==============================================================================
# SPATIOTEMPORAL ANOMALY CLASSIFICATION CONSTANTS
# ==============================================================================
# History buffer window size N (frames) for velocity/acceleration analysis (N >= 25)
SPATIOTEMPORAL_WINDOW_N: int = int(os.getenv("SPATIOTEMPORAL_WINDOW_N", "25"))

# Threat confidence score threshold to consider an anomaly verified ([0.0 - 1.0])
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.78"))

# Sustained anomaly time required to fire alert (seconds) - 0.4s enables snappy quick response
SUSTAINED_ANOMALY_DURATION_SEC: float = float(os.getenv("SUSTAINED_ANOMALY_DURATION_SEC", "0.4"))

# Fallback frame count trigger (if FPS is unknown, e.g. 0.6s * 25 fps = 15 frames)
ANOMALY_FRAMES_TRIGGER: int = int(os.getenv("ANOMALY_FRAMES_TRIGGER", "15"))

# Keypoint acceleration threshold (pixels/frame) indicating violent strikes/lunges
KEYPOINT_ACCELERATION_THRESHOLD: float = float(os.getenv("KEYPOINT_ACCELERATION_THRESHOLD", "22.0"))

# Bounding box jitter threshold (ratio of dimension variance to bounding box area)
BBOX_JITTER_THRESHOLD: float = float(os.getenv("BBOX_JITTER_THRESHOLD", "0.15"))

# Benign interaction constraints (Hugs, High-Fives & Handshakes):
# Maximum centroid distance variance over N frames for benign close contact
BENIGN_MAX_DISTANCE_VARIANCE: float = float(os.getenv("BENIGN_MAX_DISTANCE_VARIANCE", "45.0"))

# Maximum approach velocity allowed just prior to contact in benign scenarios (pixels/frame)
BENIGN_MAX_APPROACH_VELOCITY: float = float(os.getenv("BENIGN_MAX_APPROACH_VELOCITY", "18.0"))

# High-Five gesture constraints (raised wrists above shoulders & close hand proximity)
BENIGN_HIGH_FIVE_MAX_WRIST_DIST: float = float(os.getenv("BENIGN_HIGH_FIVE_MAX_WRIST_DIST", "220.0"))
BENIGN_HIGH_FIVE_ELEVATION_PX: float = float(os.getenv("BENIGN_HIGH_FIVE_ELEVATION_PX", "10.0"))

# Hugging gesture constraints (close chest-to-chest proximity & reciprocal arm wrapping)
BENIGN_HUG_PROXIMITY_PX: float = float(os.getenv("BENIGN_HUG_PROXIMITY_PX", "130.0"))
BENIGN_HUG_MIN_DURATION_SEC: float = float(os.getenv("BENIGN_HUG_MIN_DURATION_SEC", "0.3"))

# Machine Learning Gesture Classifier Model Paths (Fast NPZ binary & fallback Joblib)
GESTURE_FAST_MODEL_PATH: Path = BASE_DIR / "gesture_classifier_fast.npz"
GESTURE_MODEL_PATH: Path = GESTURE_FAST_MODEL_PATH if GESTURE_FAST_MODEL_PATH.exists() else (BASE_DIR / "gesture_classifier.joblib")


# ==============================================================================
# ALERT & COOLDOWN SETTINGS
# ==============================================================================
# Strict cooldown period (seconds) to prevent alert spamming during ongoing incidents
COOLDOWN_SECONDS: int = int(os.getenv("COOLDOWN_SECONDS", "30"))

# ==============================================================================
# WHATSAPP DISPATCH SETTINGS (HEADLESS NODE.JS BOT MICROSERVICE - SAME AS UAV)
# ==============================================================================
# Enable WhatsApp notifications
ENABLE_WHATSAPP: bool = os.getenv("ENABLE_WHATSAPP", "true").lower() in ("true", "1", "yes")
USE_FREE_WHATSAPP: bool = ENABLE_WHATSAPP

# The URL of the local Headless WhatsApp bot server (default port 3000)
WHATSAPP_SERVER_URL: str = os.getenv("WHATSAPP_SERVER_URL", "http://localhost:3000/send")
WHATSAPP_STATUS_URL: str = os.getenv("WHATSAPP_STATUS_URL", "http://localhost:3000/status")
WHATSAPP_QR_URL: str = os.getenv("WHATSAPP_QR_URL", "http://localhost:3000/qr")

# Authorized Sender phone number (Source: 8618118952 -> +918618118952)
SENDER_NUMBER: str = os.getenv("SENDER_NUMBER", "+918618118952")

# Target Recipient phone number (format: 919876543210 or 917760489756)
WHATSAPP_TO_NUMBER: str = os.getenv("WHATSAPP_TO_NUMBER", "917760489756")

# Emergency recipient phone number list
_raw_recipients: str = os.getenv("RECIPIENT_NUMBERS", WHATSAPP_TO_NUMBER)
RECIPIENT_NUMBERS: List[str] = [
    num.strip() if num.strip().startswith("+") else (
        f"+91{num.strip()}" if len(num.strip()) == 10 and num.strip().isdigit() else f"+{num.strip()}"
    )
    for num in _raw_recipients.split(",")
    if num.strip()
]

# Optional CallMeBot Free API Key (if user has one for headless silent WhatsApp delivery)
CALLMEBOT_APIKEY: str = os.getenv("CALLMEBOT_APIKEY", "")

# Optional Twilio credentials (only used if explicitly requested, not required)
TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
TWILIO_MEDIA_BASE_URL: str = os.getenv("TWILIO_MEDIA_BASE_URL", "")

# ==============================================================================
# DISPLAY & USER INTERFACE
# ==============================================================================
DISPLAY_WINDOW_NAME: str = "CCTV Threat & Anomaly Surveillance System - Women Safety"
PREVIEW_WIDTH: int = int(os.getenv("PREVIEW_WIDTH", "1280"))
PREVIEW_HEIGHT: int = int(os.getenv("PREVIEW_HEIGHT", "720"))
# Frame rate controls: strictly capped at 40 FPS max per user requirement
MAX_FPS: float = float(os.getenv("MAX_FPS", "40.0"))
MIN_TARGET_FPS: float = float(os.getenv("MIN_TARGET_FPS", "30.0"))
DEFAULT_FPS: float = float(os.getenv("DEFAULT_FPS", "30.0"))


# YOLO Inference Image Size (256 ensures >= 30 FPS on standard CPUs for ultra-responsive gesture tracking)
YOLO_IMGSZ: int = int(os.getenv("YOLO_IMGSZ", "256"))

# Optimal CPU inference threads (default 8 for 12-core system)
YOLO_CPU_THREADS: int = int(os.getenv("YOLO_CPU_THREADS", "8"))


# Video tracking inference stride (default 2 for high-FPS 60-100+ FPS mode; set to 1 for per-frame detection)
YOLO_INFERENCE_STRIDE: int = int(os.getenv("YOLO_INFERENCE_STRIDE", "2"))

