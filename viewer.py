import zmq
import msgpack
import cv2
import numpy as np
import threading

# List of detector VMs and their ports
DETECTORS = [
    ("10.18.6.11", 5557),  # VM1
    ("10.18.6.12", 5558),  # VM2
]

context = zmq.Context()
frame_buffer = {}
expected_frame_id = 0

# Function to receive frames from one detector VM
def receive_from_vm(ip, port):
    sock = context.socket(zmq.PULL)
    sock.connect(f"tcp://{ip}:{port}")
    print(f"[INFO] Connected to detector at {ip}:{port}")
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
            print(f"[ERROR {ip}:{port}] {e}")

# Start one thread per detector
for ip, port in DETECTORS:
    threading.Thread(target=receive_from_vm, args=(ip, port), daemon=True).start()

# Display loop: reconstruct in correct order
cv2.namedWindow("YOLOv5 Reconstructed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 Reconstructed", 960, 540)

while True:
    if expected_frame_id in frame_buffer:
        frame = frame_buffer.pop(expected_frame_id)
        cv2.imshow("YOLOv5 Reconstructed", frame)
        expected_frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
