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
