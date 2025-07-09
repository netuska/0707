cv2.namedWindow("YOLOv5 Live Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 Live Stream", 960, 540)

while True:
    try:
        # Receive and decode frame
        frame_bytes = socket.recv()
        jpg_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ Failed to decode frame.")
            continue

        # Show live video
        cv2.imshow("YOLOv5 Live Stream", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"[ERROR] {e}")
        continue

# Cleanup
cv2.destroyAllWindows()
