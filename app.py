import logging
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load YOLO model for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]  # Fixed line

# Load COCO labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Serve the HTML file
@app.route('/')
def index():
    app.logger.debug('Serving index.html')
    return send_from_directory('.', 'index.html')

# Detect "person" objects in a frame and draw bounding boxes
def detect_persons(frame):
    app.logger.debug('Detecting persons in frame')
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to store detected persons
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':  # Only detect "person"
                # Bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    detected_persons = 0
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detected_persons += 1

    app.logger.debug(f'Detected persons: {detected_persons}')
    return frame, detected_persons

# Detect head turn direction (left, right, up, down)
def detect_head_turn(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        direction = []
        threshold = 30  # Reduced threshold for better sensitivity

        app.logger.debug(f'Face Center: ({face_center_x}, {face_center_y}), Frame Center: ({frame_center_x}, {frame_center_y})')

        # Detect left or right
        if face_center_x < frame_center_x - threshold:
            direction.append("Left")
        elif face_center_x > frame_center_x + threshold:
            direction.append("Right")

        # Detect up or down
        if face_center_y < frame_center_y - threshold:
            direction.append("Up")
        elif face_center_y > frame_center_y + threshold:
            direction.append("Down")

        if not direction:
            direction.append("Center")

        return " ".join(direction)

    return "No Face"

# Handle video upload and processing
@app.route('/upload', methods=['POST'])
def upload_video():
    app.logger.debug('Handling video upload')
    if 'file' not in request.files:
        app.logger.error('No file uploaded')
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    video_path = 'uploaded_video.mp4'  # Save as .mp4
    file.save(video_path)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.error('Error opening video file')
        return jsonify({'error': 'Error opening video file'}), 500

    # Variables for person detection
    detected_persons = 0
    frames_with_persons = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Stop after 5 seconds
        if time.time() - start_time > 5:
            break

        # Resize frame to decrease size
        frame = cv2.resize(frame, (640, 360))  # Resize to 640x360 (adjust as needed)

        # Detect persons in the frame
        frame_with_persons, persons_count = detect_persons(frame)
        if persons_count > 0:  # Only save frames with detected persons
            # Detect head turn direction
            direction = detect_head_turn(frame_with_persons)
            cv2.putText(frame_with_persons, f'Head Turn: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frames_with_persons.append(frame_with_persons)
            detected_persons += persons_count

    cap.release()

    # Save frames with detected persons
    output_dir = 'detected_frames'
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames_with_persons):
        cv2.imwrite(f'{output_dir}/frame_{i}.jpg', frame)

    # Prepare response
    response = {
        'message': f'Detected {detected_persons} persons in 5 seconds',
        'frames': [f'detected_frames/frame_{i}.jpg' for i in range(len(frames_with_persons))],
    }

    app.logger.debug(f'Detected persons: {detected_persons}')
    return jsonify(response), 200

# Serve detected frames
@app.route('/detected_frames/<filename>')
def serve_frame(filename):
    return send_from_directory('detected_frames', filename)

if __name__ == '__main__':
    app.run(debug=True)
