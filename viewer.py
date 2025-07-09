msg = socket.recv()
data = msgpack.unpackb(msg, raw=False)

frame_id = data["frame_id"]
jpg_bytes = data["image"]
jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)

frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

_, jpeg = cv2.imencode('.jpg', annotated_frame)
out_msg = msgpack.packb({
    "frame_id": frame_id,
    "image": jpeg.tobytes()
}, use_bin_type=True)
sender.send(out_msg)
