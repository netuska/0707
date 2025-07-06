import zmq
import cv2
import numpy as np
import torch
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://dispatcher:5555")

while True:
    try:
        frame_id, jpg_buffer = socket.recv_multipart()
        jpg_array = np.frombuffer(jpg_buffer, dtype=np.uint8)
        frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

        results = model(frame)
        print(f"[{os.getenv('NAME')}] Frame {frame_id.decode()}")
        results.print()
    except Exception as e:
        print(f"Error: {e}")