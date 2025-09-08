app.py
import os
import subprocess
from flask import Flask, request, render_template_string

UPLOAD_FOLDER = "/uploads"
RTSP_URL = "rtsp://rtsp-server:8554/webcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
ffmpeg_process = None

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Video Uploader</title></head>
<body>
  <h2>Upload a video to stream via RTSP</h2>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="video" accept="video/*">
    <input type="submit" value="Upload & Stream">
  </form>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def upload_video():
    global ffmpeg_process
    if request.method == "POST":
        if "video" not in request.files:
            return "No file uploaded", 400

        video = request.files["video"]
        if video.filename == "":
            return "No selected file", 400

        filepath = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(filepath)

        # Stop previous ffmpeg if running
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()

        # Launch ffmpeg to stream uploaded video
        cmd = [
            "ffmpeg", "-re", "-stream_loop", "-1",
            "-i", filepath,
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-g", "48", "-keyint_min", "48", "-pix_fmt", "yuv420p",
            "-rtsp_transport", "tcp", "-an",
            "-f", "rtsp", RTSP_URL
        ]
        ffmpeg_process = subprocess.Popen(cmd)
        return f"✅ Streaming {video.filename} to {RTSP_URL}"

    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)






Dockerfile: 
FROM python:3.9-slim

WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY app.py .
RUN pip install flask

EXPOSE 8080
VOLUME ["/uploads"]

CMD ["python", "app.py"]


extend docker-compose

  web-uploader:
    build: ./web-uploader
    container_name: web-uploader
    volumes:
      - ./uploads:/uploads
    ports:
      - "8080:8080"
    depends_on:
      - rtsp-server







import json
import csv
from datetime import datetime

# Step 1: Load JSON from file
with open("input.json", "r") as f:
    data = json.load(f)

# Step 2: Prepare CSV rows
rows = []
for result in data["data"]["result"]:
    for timestamp, value in result["values"]:
        time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
        rows.append({
            "timestamp": time_str,
            "cpu": value
        })

# Step 3: Write to CSV
with open("output.csv", "w", newline="") as csvfile:
    fieldnames = ["timestamp", "cpu"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("CSV file has been saved as 'output.csv'")








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load CSV with your CPU usage data ---
# Your CSV must have at least: timestamp, cpu
df = pd.read_csv("your_cpu_data.csv")  # <<< Change filename

# Convert timestamp to seconds if needed
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
df = df[['seconds', 'cpu']].rename(columns={"seconds": "timestamp"})  # keep consistent column names

# --- 2. Define evaluation function ---
def evaluate_lr_over_windows(data, max_window_size=30, prediction_horizon=180):
    results = []

    for window_size in range(3, max_window_size + 1):  # test sliding windows from 3 to N
        y_true = []
        y_pred = []

        for i in range(window_size, len(data) - prediction_horizon // 20):
            X = np.arange(window_size).reshape(-1, 1) * 20  # time step in seconds (assumes 20s sampling interval)
            y = data['cpu'].iloc[i - window_size:i].values

            model = LinearRegression().fit(X, y)

            # Predict 3 minutes ahead
            future_time = np.array([[X[-1][0] + prediction_horizon]])
            pred = model.predict(future_time)[0]
            actual = data['cpu'].iloc[i + prediction_horizon // 20]

            y_true.append(actual)
            y_pred.append(pred)

        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results.append({
            'window_size': window_size,
            'MAPE': mape,
            'RMSE': rmse,
            'R2': r2
        })

    return pd.DataFrame(results)

# --- 3. Run evaluation ---
results_df = evaluate_lr_over_windows(df, max_window_size=30, prediction_horizon=180)

# --- 4. Plot the evaluation metrics ---
plt.figure(figsize=(12, 6))
plt.plot(results_df['window_size'], results_df['MAPE'], label='MAPE', marker='o')
plt.plot(results_df['window_size'], results_df['RMSE'], label='RMSE', marker='o')
plt.plot(results_df['window_size'], results_df['R2'], label='R2 Score', marker='o')
plt.title("Linear Regression Accuracy vs Window Size (3-minute Prediction)")
plt.xlabel("Window Size (number of past samples)")
plt.ylabel("Error / Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
