"""
GuardianAI Configuration Module
================================
Central configuration file for all system constants and thresholds.
Modify these values to tune the system sensitivity.
"""

import os

# Camera Configuration
CAMERA_ID: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS: int = 30

# Audio Configuration
# Ensure you create an 'assets' folder with these .wav files
ASSETS_DIR = "assets"
ALARM_FILE = os.path.join(ASSETS_DIR, "alarm.wav")       # Loud alarm for drowsiness
WARNING_FILE = os.path.join(ASSETS_DIR, "warning.wav")   # Softer warning for yawning/microsleep

# Eye Aspect Ratio (EAR) Thresholds
EAR_THRESHOLD: float = 0.21  # Below this value indicates closed eyes
BLINK_CONSEC_FRAMES: int = 15  # Number of consecutive frames to confirm drowsiness
EAR_BUFFER_SIZE: int = 15  # Size of moving average buffer

# Yawning (Mouth Aspect Ratio - MAR) Thresholds
# MAR = Vertical Mouth Distance / Horizontal Mouth Distance
MAR_THRESHOLD: float = 0.5   # Threshold for open mouth (yawning)
YAWN_CONSEC_FRAMES: int = 20 # Consecutive frames to confirm a yawn
# MediaPipe Face Mesh Indices for Mouth [Top, Bottom, Left, Right]
MOUTH_INDICES: list = [13, 14, 78, 308] 

# Microsleep / Dead Man Switch
# If no blink is detected for this many seconds, trigger a warning
MICROSLEEP_THRESHOLD: int = 60  

# Head Pose Thresholds (in degrees)
YAW_THRESHOLD: int = 60  # Maximum allowed head rotation left/right
PITCH_THRESHOLD: int = 18  # Maximum allowed head tilt up/down
ROLL_THRESHOLD: int = 25  # Maximum allowed head tilt side-to-side

# Safety Score Configuration
INITIAL_SAFETY_SCORE: int = 100
DROWSINESS_PENALTY: int = 5
DISTRACTION_PENALTY: int = 3
YAWN_PENALTY: int = 2        # New penalty for yawning
SCORE_REGENERATION: float = 0.5  # Points regenerated per normal frame
MIN_SAFETY_SCORE: int = 0
MAX_SAFETY_SCORE: int = 100
CRITICAL_SCORE_THRESHOLD: int = 30  # Below this triggers emergency protocols

# Logging Configuration
LOG_FILE: str = "logs.csv"
EVIDENCE_FOLDER: str = "evidence"
SNAPSHOT_PREFIX: str = "violation"

# MediaPipe Configuration
MEDIAPIPE_MAX_FACES: int = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.6

# UI Configuration
UI_FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_SCALE: float = 0.6
UI_THICKNESS: int = 2
UI_COLOR_NORMAL: tuple = (0, 255, 0)  # Green
UI_COLOR_WARNING: tuple = (0, 165, 255)  # Orange
UI_COLOR_CRITICAL: tuple = (0, 0, 255)  # Red
UI_COLOR_TEXT: tuple = (255, 255, 255)  # White
UI_COLOR_BG: tuple = (0, 0, 0)  # Black

# 3D Face Model Points (nose tip, chin, left eye, right eye, left mouth, right mouth)
FACE_3D_MODEL = [
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
]

# MediaPipe Face Mesh Landmark Indices
NOSE_TIP: int = 1
CHIN: int = 152
LEFT_EYE_LEFT: int = 33
RIGHT_EYE_RIGHT: int = 263
LEFT_MOUTH: int = 61
RIGHT_MOUTH: int = 291

# Eye Landmark Indices (for EAR calculation)
LEFT_EYE_INDICES: list = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES: list = [33, 160, 158, 133, 153, 144]
ALARM_FILE = os.path.join(ASSETS_DIR, "alarm.wav")
WARNING_FILE = os.path.join(ASSETS_DIR, "warning.wav")