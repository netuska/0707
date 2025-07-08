import cv2
import zmq
import time

rtsp_url = 'rtsp://viewer:passw0rd@localhost:8554/webcam'
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Failed to open stream.")
    exit()

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5555")
socket.SNDHWM = 1  # Limit high-water mark to avoid buffer explosion

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Frame read failed.")
        time.sleep(0.1)
        continue

    try:
        _, buffer = cv2.imencode('.jpg', frame)
        socket.send(buffer.tobytes(), flags=zmq.NOBLOCK)
        print(f"✅ Sent frame {frame_id}")
        frame_id += 1

    except zmq.Again:
        print("⏳ ZMQ send buffer full. Retrying...")
        time.sleep(0.05)  # Give time for buffer to clear

    except Exception as e:
        print(f"❌ Other ZMQ Error: {e}")
        time.sleep(0.1)


