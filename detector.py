import zmq
import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.eval()

# Set up ZeroMQ subscriber
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://<DISPATCHER_VM_IP>:5555")  # replace with actual IP
socket.setsockopt_string(zmq.SUBSCRIBE, '')

print("[INFO] Detector started. Waiting for frames...")

while True:
    try:
        # Receive frame as bytes
        frame_bytes = socket.recv()
        
        # Decode frame
        jpg_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("❌ Failed to decode frame")
            continue

        # YOLO detection
        results = model(frame)

        # Draw results on the frame
        annotated = np.squeeze(results.render())

        # Display frame
        cv2.imshow("Detections", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"[ERROR] {e}")
        continue

# Clean up
cv2.destroyAllWindows()
