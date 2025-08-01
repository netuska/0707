def add_agent_entry(ip, port, path="agent_instances.json"):
    agents = []
    try:
        with open(path, "r") as f:
            agents = json.load(f)
    except:
        pass

    if not any(d["ip"] == ip and d["port"] == port for d in agents):
        agents.append({"ip": ip, "port": port})
        with open(path, "w") as f:
            json.dump(agents, f, indent=2)
        print(f"[INFO] Agent {ip}:{port} added.")


def remove_agent_entry(ip, port, path="agent_instances.json"):
    try:
        with open(path, "r") as f:
            agents = json.load(f)

        agents = [a for a in agents if not (a["ip"] == ip and a["port"] == port)]

        with open(path, "w") as f:
            json.dump(agents, f, indent=2)

        print(f"[INFO] Agent {ip}:{port} removed.")
    except Exception as e:
        print(f"[ERROR] Could not remove agent: {e}")

import numpy as np
from sklearn.linear_model import LinearRegression
import math

def predict_next_usage(metric_series, interval=20, steps_ahead=1):
    timestamps = np.array([i * interval for i in range(len(metric_series))]).reshape(-1, 1)
    values = np.array(metric_series)
    model = LinearRegression().fit(timestamps, values)
    future_time = np.array([[timestamps[-1][0] + interval * steps_ahead]])
    return model.predict(future_time)[0]





cpu_window = [180, 200, 220, 250, 270, 290]
cpu_limit_per_instance = 100

predicted_cpu = predict_next_usage(cpu_window)
required_instances = math.ceil(predicted_cpu / cpu_limit_per_instance)

# Only trigger if predictive result is higher than current replicas
if required_instances > current_replicas:
    scale_service(required_instances)  # your existing scale function
    print(f"[Predictive] Scaling to {required_instances} based on prediction: {predicted_cpu} mCores")
