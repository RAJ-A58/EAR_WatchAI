# 🚗 GuardianAI – Train Driver Monitoring System

GuardianAI is a real-time computer vision–based Driver Monitoring System (DMS) designed to enhance safety by continuously analyzing driver behavior. It detects drowsiness, distraction, yawning, and microsleeps, providing instant visual and audio alerts while maintaining a dynamic safety score.

## 🚀 Key Features

* **Real-Time Drowsiness Detection:** Uses Eye Aspect Ratio (EAR) to detect prolonged eye closure.
* **Audio Alerts & Alarms:**
    * **Critical Alarm:** Looping alarm for drowsiness or microsleep.
    * **Warning Tone:** Single warning beep for distraction or yawning.
* **Distraction Detection:** Estimates 3D Head Pose (Pitch, Yaw, Roll) to detect when the driver looks away.
* **Microsleep Detection:** Monitors blink rates to detect staring (e.g., no blinking for 60s).
* **Dynamic Safety Score (0–100):** Penalizes unsafe behavior and regenerates during normal driving.
* **Automated Evidence Collection:** Automatically saves snapshots when violations occur, tagged with the violation type and safety score.

## 🛠️ Tech Stack

* Python
* OpenCV (cv2)
* MediaPipe
* NumPy

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/GuardianAI.git](https://github.com/yourusername/GuardianAI.git)
    cd GuardianAI
    ```

2.  **Install Dependencies**
    *(Refer to project requirements)*

3.  **Generate Audio Assets**
    Run this script to generate the required `.wav` files:
    ```bash
    python create_sounds.py
    ```

## 📁 Project Structure
GuardianAI/
├── main.py                 # Entry point
├── config.py               # System settings
├── create_sounds.py        # Audio generator script
├── logs.csv                # Event history log
├── assets/                 # Audio files
├── evidence/               # Snapshots of violations
├── detectors/
│   ├── drowsiness.py       # EAR logic
│   └── head_pose.py        # PnP & Euler angles logic
└── utils/
├── sound.py            # Audio alert manager
└── logger.py           # Logging & Snapshot handling




## 🖥️ Usage

Run the System:

python main.py
📊 Data & Evidence Output
Evidence Snapshots (evidence/)
Snapshots are saved immediately upon violation. Filename format: violation_{TYPE}_{YYYYMMDD}_{HHMMSS}_score{SCORE}.jpg

Event Logs (logs.csv)
Events are logged periodically or upon state changes.

🎮 Controls
When the window is active:

q : Quit the application

r : Reset safety score and alarms

s : Save a manual snapshot

c : Clear logs

⚙️ Configuration
Tune the system in config.py:

EAR_THRESHOLD: Eye closure sensitivity.

YAW_THRESHOLD: Maximum allowed head rotation.

MICROSLEEP_THRESHOLD: Seconds without blinking.

CRITICAL_SCORE_THRESHOLD: Triggers emergency evidence capture.
