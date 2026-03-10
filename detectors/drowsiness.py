"""
Drowsiness Detection Module
============================
Detects driver drowsiness using Eye Aspect Ratio (EAR), Yawning (MAR),
and Microsleep (Blink Rate) analysis.

Mathematical Background:
-----------------------
1. Eye Aspect Ratio (EAR):
   EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

2. Mouth Aspect Ratio (MAR):
   Used to detect yawning. 
   MAR = Vertical Mouth Height / Horizontal Mouth Width

3. Microsleep / Dead Man Check:
   Monitors time since the last blink. If a driver stares for too long
   without blinking (e.g., 60s), it indicates cognitive disengagement.
"""

import cv2 
import numpy as np
from collections import deque
from typing import Optional, Tuple
import time
import config
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class DrowsinessDetector:
    """
    Detects drowsiness by monitoring Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
    Also tracks blink rate to detect "staring spells" (microsleeps).
    """
    
    def __init__(self) -> None:
        """
        Initialize MediaPipe Face Landmarker, EAR/MAR tracking, and blink timers.
        """
        # Create Face Landmarker with base options
        base_options = python.BaseOptions(
            model_asset_buffer=self._download_model()
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=config.MEDIAPIPE_MAX_FACES,
            min_face_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Buffers for smoothing EAR values (moving average)
        self.left_ear_buffer: deque = deque(maxlen=config.EAR_BUFFER_SIZE)
        self.right_ear_buffer: deque = deque(maxlen=config.EAR_BUFFER_SIZE)
        
        # Counters
        self.drowsy_frames: int = 0
        self.yawn_frames: int = 0
        
        # Microsleep / Blink Tracking
        self.last_blink_time: float = time.time()
        self.is_blinking: bool = False
        
        # Current status
        self.current_status: str = "ALERT"
    
    def _download_model(self) -> bytes:
        """
        Download MediaPipe Face Landmarker model.
        Returns: Model file as bytes
        """
        import urllib.request
        import os
        
        model_path = "face_landmarker.task"
        
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("✓ Model downloaded successfully")
        
        with open(model_path, "rb") as f:
            return f.read()
        
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    def calculate_EAR(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        """
        # Vertical eye distances
        vertical_1 = self._euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = self._euclidean_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye distance
        horizontal = self._euclidean_distance(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def calculate_MAR(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for Yawning detection.
        Formula: MAR = Vertical Distance / Horizontal Distance
        
        Args:
            mouth_landmarks: Array of 4 mouth points [Top, Bottom, Left, Right]
        """
        # Vertical: Top Lip to Bottom Lip
        vertical = self._euclidean_distance(mouth_landmarks[0], mouth_landmarks[1])
        
        # Horizontal: Left Corner to Right Corner
        horizontal = self._euclidean_distance(mouth_landmarks[2], mouth_landmarks[3])
        
        if horizontal == 0:
            return 0.0
            
        return vertical / horizontal
    
    def _extract_landmarks(self, face_landmarks, img_width: int, img_height: int, 
                           indices: list) -> np.ndarray:
        """
        Extract landmark coordinates from MediaPipe results.
        """
        points = []
        for idx in indices:
            landmark = face_landmarks[idx]
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            points.append([x, y])
        
        return np.array(points, dtype=np.float64)
    
    def detect_drowsiness(self, image: np.ndarray) -> Tuple[str, float, float, np.ndarray]:
        """
        Detect drowsiness, yawning, and microsleeps in the input image.
        
        Returns:
            Tuple of (status, average_EAR, average_MAR, annotated_image)
        """
        img_h, img_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face landmarks
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return "NO_FACE", 0.0, 0.0, image
        
        face_landmarks = detection_result.face_landmarks[0]
        
        # 1. EAR Calculation (Drowsiness)
        left_eye = self._extract_landmarks(
            face_landmarks, img_w, img_h, config.LEFT_EYE_INDICES
        )
        right_eye = self._extract_landmarks(
            face_landmarks, img_w, img_h, config.RIGHT_EYE_INDICES
        )
        
        left_ear = self.calculate_EAR(left_eye)
        right_ear = self.calculate_EAR(right_eye)
        
        # Add to buffers
        self.left_ear_buffer.append(left_ear)
        self.right_ear_buffer.append(right_ear)
        
        # Calculate average EAR
        avg_ear = (np.mean(self.left_ear_buffer) + np.mean(self.right_ear_buffer)) / 2.0
        
        # 2. MAR Calculation (Yawning)
        # config.MOUTH_INDICES is [13, 14, 78, 308] -> [Top, Bottom, Left, Right]
        mouth_points = self._extract_landmarks(
            face_landmarks, img_w, img_h, config.MOUTH_INDICES
        )
        mar = self.calculate_MAR(mouth_points)
        
        # 3. Update Blink Tracking (for Microsleep)
        # We consider a blink to occur when EAR drops below threshold
        if avg_ear < config.EAR_THRESHOLD:
            self.is_blinking = True
        else:
            # If we were blinking and now we are not, we just finished a blink
            if self.is_blinking:
                self.last_blink_time = time.time()
                self.is_blinking = False

        # Calculate time since last blink
        time_since_blink = time.time() - self.last_blink_time
        
        # 4. Check All Statuses
        self.current_status = "ALERT"
        
        # Priority 1: Drowsiness (Eyes Closed)
        if avg_ear < config.EAR_THRESHOLD:
            self.drowsy_frames += 1
            if self.drowsy_frames >= config.BLINK_CONSEC_FRAMES:
                self.current_status = "DROWSY"
        else:
            self.drowsy_frames = 0
            
        # Priority 2: Yawning (Only if not already drowsy)
        if self.current_status != "DROWSY":
            if mar > config.MAR_THRESHOLD:
                self.yawn_frames += 1
                if self.yawn_frames >= config.YAWN_CONSEC_FRAMES:
                    self.current_status = "YAWNING"
            else:
                self.yawn_frames = 0
                
        # Priority 3: Microsleep (Staring spell / Dead man switch)
        # Only trigger if eyes are open (ALERT) but haven't blinked in a long time
        if self.current_status == "ALERT" and time_since_blink > config.MICROSLEEP_THRESHOLD:
            self.current_status = "MICROSLEEP"
        
        return self.current_status, avg_ear, mar, image
    
    def reset(self) -> None:
        """
        Reset all buffers and counters.
        """
        self.left_ear_buffer.clear()
        self.right_ear_buffer.clear()
        self.drowsy_frames = 0
        self.yawn_frames = 0
        self.last_blink_time = time.time()
        self.current_status = "ALERT"
    
    def __del__(self) -> None:
        """Clean up MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()