import zmq
import msgpack
import cv2
import numpy as np
import threading

# Config: list of detector ports
DETECTOR_PORTS = [5557, 5558]  # Add more if needed

context = zmq.Context()
frame_buffer = {}
expected_frame_id = 0

# Threaded function to receive from each detector
def receive_frames(port):
    sock = context.socket(zmq.PULL)
    sock.connect(f"tcp://<VM_IP>:${port}")  # Replace <VM_IP> accordingly
    while True:
        try:
            msg = sock.recv()
            data = msgpack.unpackb(msg, raw=False)
            frame_id = data["frame_id"]
            jpg = data["image"]
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer[frame_id] = frame
        except Exception as e:
            print(f"[ERROR Port {port}] {e}")

# Start threads to receive from all detectors
for port in DETECTOR_PORTS:
    threading.Thread(target=receive_frames, args=(port,), daemon=True).start()

# Display loop
cv2.namedWindow("YOLOv5 Reconstructed Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 Reconstructed Video", 960, 540)

while True:
    if expected_frame_id in frame_buffer:
        frame = frame_buffer.pop(expected_frame_id)
        cv2.imshow("YOLOv5 Reconstructed Video", frame)
        expected_frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
