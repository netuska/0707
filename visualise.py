import cv2
import numpy as np

# === Configuration ===
video_paths = [
    "output_1_instance.avi",
    "output_2_instances.avi",
    "output_3_instances.avi"  # You can remove this line if only comparing 2
]

target_duration_sec = 30
target_fps = 30
target_frame_count = target_duration_sec * target_fps
frame_width = 640
frame_height = 360

def load_and_normalize_video(path, target_frame_count):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        frames.append(frame)

    cap.release()
    current = len(frames)

    if current == 0:
        # If video has no frames, fill with black frames
        return [np.zeros((frame_height, frame_width, 3), dtype=np.uint8)] * target_frame_count

    if current < target_frame_count:
        # Repeat last frame to pad
        frames += [frames[-1]] * (target_frame_count - current)
    elif current > target_frame_count:
        # Downsample evenly
        step = current / target_frame_count
        frames = [frames[int(i * step)] for i in range(target_frame_count)]

    return frames

# === Load and Normalize All Videos ===
normalized_videos = [load_and_normalize_video(path, target_frame_count) for path in video_paths]

# === Display Side-by-Side ===
cv2.namedWindow("Side-by-Side Comparison", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Side-by-Side Comparison", frame_width * len(normalized_videos), frame_height)

for i in range(target_frame_count):
    frames = [video[i] for video in normalized_videos]
    combined = np.hstack(frames)
    cv2.imshow("Side-by-Side Comparison", combined)

    if cv2.waitKey(int(1000 / target_fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
