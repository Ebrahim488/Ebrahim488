🛡️ AI Security Surveillance System

An AI-powered video surveillance system that combines Military Uniform Detection, Weapon Detection, Vehicle Detection, and Multi-Object Tracking into one integrated pipeline.

The system processes CCTV or recorded video, sends selected frames to three pretrained AI models, tracks detected objects across frames, and produces an annotated output video containing detection classes, confidence scores, and persistent tracking IDs.

---

📌 Project Overview

Traditional video surveillance systems depend heavily on human monitoring. This can make it difficult to continuously identify important events, especially when multiple people, weapons, and vehicles appear simultaneously.

This project aims to build an integrated AI surveillance pipeline capable of detecting and tracking:

- 👤 Military / civilian personnel
- 🔫 Weapons
- 🚙 Vehicles
- 🪖 Military-related objects


🎯 Objectives

The main objectives of this project are:

1. Detect military uniforms/personnel.
2. Detect weapons.
3. Detect vehicles.
4. Track detected objects across consecutive video frames.
5. Reduce the disappearance and jitter of bounding boxes.
6. Maintain a tracking ID for each detected object.
7. Process the three AI models efficiently.
8. Generate an annotated output video.
9. Create a foundation for a future automated security risk-assessment system.

---

🤖 AI Models

The system uses three pretrained Roboflow models.

1. Military Uniform Model

military-soldier/1

This model is responsible for detecting objects related to the military-soldier dataset.

The model output is used as the Uniform Detection Layer.

Example:

UNIFORM ID:1 soldier 91%

---

2. Weapon Detection Model

security-and-weapon-detection-d4ya2-yyieq-nuyqk-v1n7e/1

This model is responsible for detecting weapons in the video.

Example:

WEAPON ID:3 gun 87%

---

3. Vehicle Detection Model

toofan-llvuy/1

This model is responsible for detecting vehicles.

Example:

VEHICLE ID:2 military truck 94%

---

🧠 System Architecture

The complete system consists of six major stages.

1. Video Input

The user uploads a video to Google Colab.

uploaded = files.upload()

The uploaded video is saved as:

integrated_tracking/input.mp4

---

2. Frame Extraction

OpenCV reads the video frame by frame.

cap = cv2.VideoCapture(
    str(INPUT_VIDEO)
)

Each frame contains the visual information that will be processed by the AI models.

---

⚡ 3. Frame Sampling

Running three AI models on every frame can be computationally expensive and can generate a large number of API requests.

Therefore, the system performs AI detection every two frames.

DETECTION_INTERVAL = 2

The detection condition is:

run_detection = (
    (frame_number - 1)
    % DETECTION_INTERVAL
    == 0
)

This means:

Frame 1 → AI Detection
Frame 2 → Tracking
Frame 3 → AI Detection
Frame 4 → Tracking
Frame 5 → AI Detection
...

The important difference is that tracking continues on every frame.

---

🔄 4. Image Preprocessing

Before sending a frame to Roboflow, the frame can be resized.

MAX_API_WIDTH = 960

The function:

def resize_for_api(frame):

    h, w = frame.shape[:2]

    if w <= MAX_API_WIDTH:
        return frame

    scale = MAX_API_WIDTH / w

    new_w = MAX_API_WIDTH
    new_h = int(h * scale)

    return cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

This reduces the amount of data sent to the API and can improve processing speed.

---

📦 5. Roboflow Detection

The project communicates with the Roboflow detection API using Python "requests".

def run_roboflow(
    image_bytes,
    model_id
):

    endpoint = (
        "https://detect.roboflow.com/"
        f"{model_id}"
    )

    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": CONF_THRESHOLD
    }

    response = requests.post(
        endpoint,
        params=params,
        files={
            "file": (
                "frame.jpg",
                image_bytes,
                "image/jpeg"
            )
        },
        timeout=API_TIMEOUT
    )

    return response.json()

The confidence threshold is:

CONF_THRESHOLD = 0.25

Only detections with confidence equal to or greater than this threshold are accepted.

---

🚀 6. Parallel Model Execution

The three models are executed in parallel using:

ThreadPoolExecutor

Configuration:

MAX_WORKERS = 3

The models are submitted simultaneously:

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(
            run_roboflow,
            image_bytes,
            model_id
        ): name

        for name, model_id
        in model_configs.items()
    }

The three models are therefore processed approximately like:

                 Frame
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     Uniform     Weapon     Vehicle
      Model       Model       Model
        │          │          │
        └──────────┼──────────┘
                   ▼
              Detections

This is more efficient than waiting for one model to finish before starting the next.

---

📐 Bounding Box Conversion

Roboflow returns bounding boxes using:

x
y
width
height

where "x" and "y" represent the center of the bounding box.

The system converts them into:

x1
y1
x2
y2

using:

x1 = (x - w / 2) * scale_x
y1 = (y - h / 2) * scale_y

x2 = (x + w / 2) * scale_x
y2 = (y + h / 2) * scale_y

This format is easier to use for tracking and drawing bounding boxes.

---

🎯 Multi-Object Tracking

One of the main problems with pure object detection is that bounding boxes can appear and disappear between frames.

For example:

Frame 1 → Person detected
Frame 2 → Person detected
Frame 3 → Person NOT detected
Frame 4 → Person detected

Without tracking, the bounding box may disappear.

This project introduces a tracking layer.

---

🧮 Tracking Algorithm

The tracker combines:

- Kalman Filter
- IoU matching
- Center-distance matching
- Persistent object IDs
- Lost-frame management

Each tracked object contains:

Track ID
Bounding Box
Confidence
Class
Age
Hits
Lost Frames
Kalman State

---

🧠 Kalman Filter

The Kalman Filter estimates where an object is expected to move.

The state contains:

cx
cy
width
height
vx
vy
vw
vh

Where:

cx, cy       → object center
width,height → bounding box dimensions
vx,vy        → velocity
vw,vh        → size change

The prediction step estimates the object's new position:

prediction = self.kalman.predict()

This allows the bounding box to continue moving even between AI detection frames.

---

🔗 IoU Matching

Intersection over Union (IoU) is used to determine whether a new detection corresponds to an existing track.

The formula is:

IoU = Intersection Area / Union Area

The implementation:

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union

The current threshold is:

IOU_THRESHOLD = 0.20

---

📍 Center Distance Matching

IoU alone may not always be sufficient, especially when objects move quickly.

Therefore, the tracker also compares the distance between object centers.

MAX_CENTER_DISTANCE = 0.35

This gives the tracker another way to associate a new detection with an existing object.

---

🆔 Persistent Tracking IDs

Every new object receives an ID.

For example:

UNIFORM ID:1
UNIFORM ID:2

WEAPON ID:1

VEHICLE ID:1
VEHICLE ID:2

The same object can maintain the same ID across multiple frames.

Example:

Frame 1 → UNIFORM ID:1
Frame 2 → UNIFORM ID:1
Frame 3 → UNIFORM ID:1
Frame 4 → UNIFORM ID:1

---

🔵🟥🟢 Visualization

The output video uses different colors for each detection type.

Blue  → Uniform
Red   → Weapon
Green → Vehicle

Each bounding box contains:

Model Type
Track ID
Class
Confidence

Example:

UNIFORM ID:1 soldier 92%
WEAPON ID:2 gun 87%
VEHICLE ID:1 military truck 95%

---

⏱️ Lost Object Management

Objects may temporarily disappear because of:

- Occlusion
- Detection errors
- Motion blur
- Poor lighting
- API detection failure

The tracker therefore keeps an object alive for:

MAX_LOST_FRAMES = 12

If the object is not matched for more than this number of frames, the track is removed.

---

📊 System Processing Flow

The complete processing sequence is:

                   INPUT VIDEO
                       │
                       ▼
                Read Video Frame
                       │
                       ▼
                 Resize Frame
                       │
                       ▼
                  JPEG Encode
                       │
                       ▼
              Is Detection Frame?
                 /            \
               YES             NO
                │               │
                ▼               ▼
        Run 3 AI Models      Prediction
          in Parallel          Only
                │               │
        ┌───────┼───────┐       │
        ▼       ▼       ▼       │
     Uniform  Weapon  Vehicle   │
        │       │       │       │
        └───────┼───────┘       │
                ▼               │
           Detections           │
                │               │
                ▼               │
          IoU Matching ◄────────┘
                │
                ▼
          Update Tracks
                │
                ▼
          Draw Bounding Boxes
                │
                ▼
            Save Frame
                │
                ▼
             Next Frame

---

⚙️ Configuration

The most important configuration parameters are:

CONF_THRESHOLD = 0.25

Detection confidence threshold.

DETECTION_INTERVAL = 2

Run AI detection every two frames.

MAX_LOST_FRAMES = 12

Maximum number of frames a track can survive without a new detection.

IOU_THRESHOLD = 0.20

IoU threshold for object association.

MAX_CENTER_DISTANCE = 0.35

Maximum center-distance ratio used for association.

MAX_API_WIDTH = 960

Maximum frame width sent to the API.

---

💻 Technologies Used

Programming Language

Python

Computer Vision

OpenCV

AI Models

Roboflow Hosted Models

HTTP API

Requests

Tracking

OpenCV Kalman Filter
IoU Matching
Center Distance Matching

Execution Environment

Google Colab

---

📦 Required Libraries

Install dependencies using:

pip install requests opencv-python

The project imports:

import cv2
import requests
import time
import os
import base64
import numpy as np

from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

---

▶️ How to Run

Step 1 — Open Google Colab

Create a new Google Colab notebook.

---

Step 2 — Run the Installation Cell

!pip install -q requests opencv-python

---

Step 3 — Configure the Roboflow API Key

Set your API key:

ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

«Security Note: Do not commit your real API key to GitHub. Use environment variables or Google Colab Secrets instead.»

---

Step 4 — Upload a Video

Run the project.

Colab will display an upload dialog.

Select a video file from your computer.

---

Step 5 — Processing

The system will:

1. Read the video.
2. Sample frames.
3. Resize frames when necessary.
4. Send detection frames to the three AI models.
5. Receive predictions.
6. Convert bounding boxes.
7. Match detections with existing tracks.
8. Predict object movement.
9. Draw bounding boxes.
10. Save the output video.

---

📁 Output

The project generates:

integrated_tracking/
│
├── input.mp4
│
├── tracking_raw.mp4
│
└── tracking_final.mp4

The final video contains:

Uniform Detection
Weapon Detection
Vehicle Detection
Tracking IDs
Confidence Scores

---

📈 Performance Strategy

The project uses several techniques to reduce processing time.

1. Frame Sampling

Instead of sending every frame to the AI models:

DETECTION_INTERVAL = 2

Only every second frame is sent for detection.

2. Parallel Requests

The three models run concurrently:

MAX_WORKERS = 3

3. Image Resizing

Large frames are resized before API processing:

MAX_API_WIDTH = 960

4. JPEG Compression

Frames are encoded using:

JPEG_QUALITY = 80

This reduces the amount of data sent to the API.

---

⚠️ Current Limitations

This project is currently a prototype and has several limitations.

1. Internet/API Dependency

The detection models are accessed through the Roboflow API.

Therefore, the system requires an internet connection.

---

2. API Latency

Because the models are hosted remotely, network latency affects the processing speed.

---

3. Tracking Limitations

The current tracker is a lightweight custom tracker based on Kalman prediction and IoU association.

It can lose an ID when:

- An object is heavily occluded.
- Multiple objects overlap.
- The object moves very quickly.
- The detector misses an object for too long.
- The object leaves and re-enters the scene.

---

4. Uniform Model Limitation

The Uniform model currently detects classes provided by the pretrained model.

Therefore, a prediction such as:

soldier

should not automatically be interpreted as definitive proof that a person is wearing a military uniform.

The model's actual training classes and performance determine what the prediction means.

---

🔮 Future Improvements

The project can be extended into a complete automated security system.

1. Advanced Tracking

Replace the lightweight tracker with:

ByteTrack
BoT-SORT
DeepSORT

This can improve object association in crowded scenes.

---

2. Local Model Inference

Instead of sending every frame to a remote API, the models could eventually be deployed locally using:

YOLO
PyTorch
ONNX
TensorRT

This could significantly reduce API latency.

---

3. Object Fusion

A future version can associate:

Person
   +
Weapon
   +
Vehicle

For example:

Person ID 12
      │
      ├── Uniform: Soldier
      │
      ├── Weapon: Detected
      │
      └── Nearby Vehicle: Military Truck

---

4. Risk Assessment Layer

A future rule-based or machine-learning risk engine can combine detections.

Example conceptual structure:

Uniform Detection
        +
Weapon Detection
        +
Vehicle Detection
        +
Tracking
        │
        ▼
   Risk Assessment
        │
        ▼
   Alert Manager

The system could then generate event records such as:

Event:
Weapon detected

Track:
Weapon ID 3

Location:
Camera 01

Time:
2026-XX-XX XX:XX:XX

---

5. Alert System

Future versions could integrate:

Email Alerts
SMS
Telegram
Dashboard Notifications
Database Events

---

6. Database Integration

Detection events could be stored in:

PostgreSQL
MySQL
SQL Server
MongoDB

Example database structure:

DetectionEvent
-------------------------
EventID
CameraID
Timestamp
ObjectType
Class
TrackID
Confidence
X1
Y1
X2
Y2

---

7. Real-Time CCTV

The current project processes recorded videos.

The next stage is connecting it directly to:

RTSP Camera
      │
      ▼
OpenCV
      │
      ▼
AI Detection
      │
      ▼
Tracking
      │
      ▼
Alerts

---

🧩 Complete Project Concept

The final architecture can evolve into:

                    CCTV / RTSP
                         │
                         ▼
                  Video Stream
                         │
                         ▼
                 Frame Processing
                         │
                         ▼
              ┌──────────────────┐
              │   AI Detection   │
              ├──────────────────┤
              │ Uniform Model    │
              │ Weapon Model     │
              │ Vehicle Model    │
              └──────────────────┘
                         │
                         ▼
                Object Association
                         │
                         ▼
                  Object Tracking
                         │
                         ▼
                  Object Fusion
                         │
                         ▼
                 Risk Assessment
                         │
                         ▼
                   Alert Manager
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
          Database               Dashboard
             │                       │
             └───────────┬───────────┘
                         ▼
                   Security Event

---

📜 License

This project is intended for educational, research, and authorized security-monitoring applications.

Users are responsible for ensuring that deployment complies with applicable laws, privacy requirements, camera-use policies, and the terms of the AI model providers.

---

👨‍💻 Author

Ibrahim Mohamed

AI / Data Analysis / Computer Vision Project

---

⭐ Project Summary

This project demonstrates how multiple pretrained computer-vision models can be integrated into a single video surveillance pipeline.

The system combines:

🪖 Military Uniform Detection
+
🔫 Weapon Detection
+
🚙 Vehicle Detection
+
🎯 Multi-Object Tracking
+
⚡ Parallel AI Inference
+
🎥 Video Processing

The current implementation provides the foundation for a larger intelligent surveillance platform capable of real-time detection, object tracking, event fusion, risk assessment, alert generation, and dashboard-based monitoring.
