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
