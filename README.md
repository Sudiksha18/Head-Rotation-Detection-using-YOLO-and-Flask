# Head-Rotation-Detection-using-YOLO-and-Flask
This project detects head rotations (left, right, up, down) in videos using YOLOv3 for person detection and Haar Cascade for face detection. The backend is built with Flask, and the frontend is a simple HTML interface for uploading videos and viewing detected frames.

Features
Upload and process videos.

Detect persons using YOLOv3.

Detect head rotation using Haar Cascade.

Display detected frames with bounding boxes and head direction.

Setup
Clone the repository.

Install dependencies: pip install flask opencv-python.
Yolov3 weights:https://pjreddie.com/media/files/yolov3.weights
Yolov3 configuration:https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
Run the app: python app.py.

Open http://localhost:5000 in your browser.

Usage
Upload a video.

View detected frames with head rotation directions.
