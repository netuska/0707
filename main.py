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
