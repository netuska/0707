import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pandas as pd

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.model(state)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.training_log = {'loss': [], 'reward': [], 'epsilon': []}

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size=32):
        # Allow training with whatever is available in memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            target_f = self.model(state_tensor).clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.loss_fn(output, target_f)
            loss.backward()
            self.optimizer.step()

            self.training_log['loss'].append(loss.item())
            self.training_log['reward'].append(reward)
            self.training_log['epsilon'].append(self.epsilon)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save_training_log()

    def save_training_log(self, filename='training_log.csv'):
        df = pd.DataFrame(self.training_log)
        df.to_csv(filename, index=False)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)







from flask import Flask, request, jsonify
import requests
import json
import time
import numpy as np
from dqn_agent import DQNAgent

app = Flask(__name__)

prometheus_url = 'http://10.18.6.11:9091'
host_ips = ['10.18.6.11', '10.18.6.19']

# --- Load Machine Data ---
def load_machine_data(file_path='machine_list.json'):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_vm_ips():
    data = load_machine_data()
    return [entry['ip_address'] for entry in data]

vm_ips = get_vm_ips()
state_size = len(vm_ips)
action_size = len(vm_ips)
dqn_agent = DQNAgent(state_size, action_size)

# --- Prometheus Queries ---
def get_container_count():
    vm_list = load_machine_data()
    query = 'container_last_seen{name=~"agent1.*"} > time() - 10'
    query_url = f'{prometheus_url}/api/v1/query'
    try:
        response = requests.get(query_url, params={'query': query})
        response.raise_for_status()
        data = response.json().get('data', {}).get('result', [])
        container_counts = {vm['ip_address']: 0 for vm in vm_list}
        for metric in data:
            ip = metric['metric'].get('instance', '').split(':')[0]
            if ip in container_counts:
                container_counts[ip] += 1
        return [container_counts[vm['ip_address']] for vm in vm_list]
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return [0] * len(vm_list)

def get_cpu_usage_percent(ip_address):
    query = f"100 - (avg by (instance)(rate(node_cpu_seconds_total{{mode='idle', instance='{ip_address}:9100'}}[5m]))) * 100"
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", params={"query": query})
        response.raise_for_status()
        results = response.json()['data']['result']
        return float(results[0]['value'][1]) if results else None
    except Exception as e:
        print(f"CPU query failed: {e}")
        return None

def get_average_power():
    return np.random.uniform(0, 100)  # placeholder

def query_latency():
    return np.random.uniform(0, 300)  # placeholder

def query_fps():
    return np.random.uniform(0, 60)  # placeholder

# --- Helper Functions ---
def build_state(container_counts):
    return [count for count in container_counts]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

def get_non_saturated_vm_indices(threshold=50):
    non_saturated = []
    for idx, ip in enumerate(vm_ips):
        cpu = get_cpu_usage_percent(ip)
        if cpu is not None and cpu < threshold:
            non_saturated.append(idx)
    return non_saturated

# --- Flask Routes ---
@app.route('/selected_vm_to_scale_up', methods=['GET'])
def selected_vm_to_scale_up():
    container_counts = get_container_count()
    state = build_state(container_counts)
    valid_actions = get_non_saturated_vm_indices()

    if not valid_actions:
        return jsonify({"error": "All VMs are saturated"})

    action = dqn_agent.act(state)
    selected_vm_ip = vm_ips[action]

    # Simulate metrics
    power = normalize(get_average_power(), 0, 170)
    latency = normalize(query_latency(), 0, 180)
    fps = normalize(query_fps(), 0, 12)

    reward = -2 * power + 3 * latency + 4 * fps

    # Take action (simulate)
    container_counts[action] += 1
    next_state = build_state(container_counts)

    dqn_agent.remember(state, action, reward, next_state)
    dqn_agent.replay()

    return jsonify({
        "selected_vm_index": action,
        "selected_vm_ip": selected_vm_ip,
        "reward": reward
    })

@app.route('/reset_agent', methods=['POST'])
def reset_agent():
    global dqn_agent
    dqn_agent = DQNAgent(state_size, action_size)
    return jsonify({"status": "Agent reset."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
