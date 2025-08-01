import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# --- 1. Simulate CPU Usage Time Series Data ---
np.random.seed(42)
timestamps = np.arange(0, 3600, 20)  # 1 hour of data sampled every 20 seconds
cpu_usage = 200 + 50 * np.sin(2 * np.pi * timestamps / 600) + 0.1 * timestamps + np.random.normal(0, 10, len(timestamps))

df = pd.DataFrame({
    'timestamp': timestamps,
    'cpu': cpu_usage
})

# --- 2. Define Evaluation Function ---
def evaluate_lr_over_windows(data, max_window_size=30, prediction_horizon=180):
    results = []

    for window_size in range(3, max_window_size + 1):  # Try sliding windows from 3 to N
        y_true = []
        y_pred = []

        for i in range(window_size, len(data) - prediction_horizon // 20):
            # Prepare training data for Linear Regression
            X = np.arange(window_size).reshape(-1, 1) * 20  # time steps in seconds
            y = data['cpu'].iloc[i - window_size:i].values

            model = LinearRegression().fit(X, y)

            # Predict 3 minutes ahead (i.e., 180 seconds)
            future_time = np.array([[X[-1][0] + prediction_horizon]])
            pred = model.predict(future_time)[0]
            actual = data['cpu'].iloc[i + prediction_horizon // 20]

            y_true.append(actual)
            y_pred.append(pred)

        # Compute evaluation metrics
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

# --- 3. Run Evaluation ---
results_df = evaluate_lr_over_windows(df, max_window_size=30, prediction_horizon=180)

# --- 4. Plot Results ---
plt.figure(figsize=(12, 6))
plt.plot(results_df['window_size'], results_df['MAPE'], label='MAPE', marker='o')
plt.plot(results_df['window_size'], results_df['RMSE'], label='RMSE', marker='o')
plt.plot(results_df['window_size'], results_df['R2'], label='R2', marker='o')
plt.title("Linear Regression Accuracy vs Window Size (3-minute Prediction)")
plt.xlabel("Window Size (number of past samples)")
plt.ylabel("Error / Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
