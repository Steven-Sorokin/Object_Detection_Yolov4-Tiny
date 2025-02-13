""" 
    Object Detection backend interface using YOLOv4-Tiny.
    ---------------------------------------------------------------
    - uses a Raspberry-Pi Camera V2
    - using a already trained module "YOLOv4-Tiny" to detect 
      objects in a real-time video
    - using Flask web app to display the stream in the web
    - playing sound when objects are detected
    - provides controll over 4 checkboxes to enable/disable frames 
      and detection info of the selected boxes.
"""
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, render_template, Response, request, jsonify
import threading
from datetime import datetime
import time

app = Flask(__name__)

# load yolov4-tiny model -> .weights and .cfg files
yolo_cfg = 'yolov4-tiny.cfg'
yolo_weights = 'yolov4-tiny.weights'
yolo_net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# load COCO class labels -> coco.names file
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# initializing raspi cam V2 -> 1640x1232 px
camera = Picamera2()
camera_config = camera.create_video_configuration(main={"size": (1640, 1232)})
camera.configure(camera_config)
camera.start()

time.sleep(2) # wait for the camera to load

# we are looking specifically for: [Person|Cat|Dog] and other detections will be [Other]
detection_filter = {'person': True, 'cat': True, 'dog': True, 'other': True}  
detection_info = "No detections yet"
detection_info_lock = threading.Lock()  # using thread for updates

last_detection = []  # last detected objects
last_sound_time = {'person': 0, 'cat': 0, 'dog': 0}  # count to prevent playing sound non stop
sound_delay = 5  # 5 sec delay between sounds

current_detected_objects = []  # currently detected objects

frame_skip = 2  # for better performance i chose to skip every 2nd frame (can be changed)
frame_count = 0  # count processed frames


def process_detection(frame):
    """
    object detection process:
    - detects, draws bounding boxes and stores info and timestamps
    - keeps last detection info on screen if no objects are detected
    """
    global detection_info, last_detection, current_detected_objects

    height, width = frame.shape[:2]  # Get frame dimensions

    # resize image for YOLO input 
    # both 608x608 and 416x416 work well but it detects a bit better on 608x608.
    yolo_input_size = (608, 608)  
    resized_frame = cv2.resize(frame, yolo_input_size)

    # convert frame to a yolo format 
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, yolo_input_size, (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

    class_ids, confidences, boxes = [], [], []
    detected_classes = [] 

    # yolo output
    for out in outs:
        for detection in out:
            #     0         1       2       3          4            5:            ...
            # [center_x, center_y, width, height, confidence, class_1_score, class_2_score, ...]
            scores = detection[5:]  # extract confidence scores
            class_id = np.argmax(scores) # finds the class with the highest probability from scores
            confidence = scores[class_id]

            # threshold -> confidence 0.5 worked the best for me
            if confidence > 0.5:  
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                # x, y as the !top-left! corner
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_classes.append(classes[class_id])

    # removes overlapping detections using "Non-Maximum Suppression"
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_with_time = {}
    current_time = datetime.now().strftime('%H:%M:%S')
    current_detected_objects.clear()

    detection_list = []  # detected objects with timestamps

    # frame bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            class_id = class_ids[i]
            label = str(classes[class_id])

            if label in detection_filter and detection_filter[label]:  # apply checkbox filters
                x, y, w, h = boxes[i]
                color = (0, 255, 0) if label == "person" else (255, 0, 0) if label == "cat" else (0, 0, 255) # color

                # draw frame and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                detection_list.append(f"{label} detected at {current_time}")
                current_detected_objects.append(label)

    # updating the detection info box
    with detection_info_lock:
        if detection_list:
            detection_info = "\n".join(detection_list)
            last_detection = detection_list  # last detection when objects are present
        else:
            detection_info = "\n".join(last_detection)  # last detection
    return frame


def generate_frames():
    """
    streaming with detections:
    - capturing from picam
    - running YOLO object detection every 2nd frame
    - streaming using Flask
    """
    global frame_count
    while True:
        frame = camera.capture_array()  # capturing from picam v2 -> 1640x1232 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert to openCV format

        if frame_count % frame_skip == 0:
            frame = process_detection(frame)

        frame_count += 1
                        # resize for streaming to 820x616
        stream_frame = cv2.resize(frame, (820, 616), interpolation=cv2.INTER_AREA)

        ret, jpeg = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')


# Flask routes -> handles ui & API requests
#-------------------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    active_classes = [key for key, value in detection_filter.items() if value]
    with detection_info_lock:
                            # html file name
        return render_template('detection_web.html', active_classes=active_classes, detection_info=detection_info)

@app.route('/set_filter', methods=['POST'])
def set_filter():
    global detection_filter
    active_classes = request.form.getlist('class')
    detection_filter = {key: key in active_classes for key in detection_filter}
    return ('', 204)

@app.route('/detection_info')
def detection_info_route():
    return jsonify({'last_detection': detection_info, 'current_objects': current_detected_objects})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
