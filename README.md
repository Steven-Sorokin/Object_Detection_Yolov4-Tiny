# Object Detection using YOLOv4-Tiny

## Overview
This project integrates **YOLO-based object detection** with a **Flask web application** and **Sound alerts**. When a `person`, `cat`, or `dog` is detected, the system plays an **alert sound** and displays detection information. The web interface allows users to **toggle sound** and **select object categories to detect**.

---

#  `object_detection_doorbell.py` – Backend (Flask & YOLO)
##  Features
- Uses **Raspberry Pi Camera V2** for **real-time object detection**.
- Uses **YOLOv4-Tiny**.
- Streams video using **Flask**.
- Plays **MP3 sound alerts** when objects are detected.
- Allows **checkbox filtering** (Enable/Disable detection for specific objects).
- Maintains **last detection info**.

### **Flask Web Server & Configuration**
- Initializes a **Flask app**.
- Loads **YOLO model and class labels**.
- Configures **Pi Camera V2 (1640x1232 resolution)**.

### **Video Streaming**
- Uses OpenCV (`cv2`) and Flask to serve an **MJPEG video stream**.
- **Resizes frames** to `820x616` for smooth browser rendering.

### **Object Detection (YOLOv4-Tiny)**
- **Processes every 2nd frame** to improve performance.
- Resizes frames to `608x608` for YOLO inference.
- **Draws bounding boxes** and overlays detection labels.
- Uses **Non-Maximum Suppression (NMS)** to eliminate duplicate detections.

### **Sound Playback (MP3 Alerts)**
- Uses HTML5 `<audio>` elements.
- Implements **5-second "cooldown" per object** to prevent spam.

### **Flask API Routes**
| Route             | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | Renders the `sound.html` web UI |
| `/video_feed`    | GET    | Streams live detection video |
| `/detection_info` | GET   | Returns latest detections (JSON) |
| `/set_filter`    | POST   | Updates detection filters based on user input |

---

#### **Install Dependencies**
```bash
pip3 install -r requirements.txt
```
#### **Download YOLOv4-Tiny Weights**
```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
```
#### **Run the Flask App**
```bash
python3 object_detection_doorbell.py
```
#### **Open in Browser**
```url
http://localhost:5000/
```

### 📂 Folder Structure
```
├── object_detection_doorbell.py            # Backend
├── templates/
│   ├── detection_web.html       # Web
├── static/
│   ├── background.jpg
│   ├── person-human.mp3
│   ├── cat-meow.mp3
│   ├── dog-bark.mp3
├── yolov4-tiny.weights # YOLO Model
├── coco.names          # Object classes
├── requirements.txt    # All installations
└── README.md           # GitHub Documentation
```