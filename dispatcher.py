import cv2
import zmq
import time

cap = cv2.VideoCapture("rtsp://viewer:passw0rd@localhost:8554/webcam")
if not cap.isOpened():
    print("❌ Failed to open stream")
    exit()

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.setsockopt(zmq.SNDHWM, 10)
socket.bind("tcp://*:5555")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        time.sleep(0.1)
        continue

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    socket.send(buffer.tobytes())
    print("✅ Frame sent")
