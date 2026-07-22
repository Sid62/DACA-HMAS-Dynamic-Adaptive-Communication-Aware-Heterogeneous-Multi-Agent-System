import sys, json, csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import distance_matrix

ARTIFACT_DIR = Path(r"C:\Users\siddh\.gemini\antigravity-ide\brain\b527b2da-3323-4b8f-afae-838fe0629f21")

print("=== STEP 1: GENERATING TRACES & CSV FILES FOR A5 SEARCH_RESCUE OSCILLATORY SEED 0 ===")

orch = DACAOrchestrator(
    scenario='search_rescue',
    network_profile='oscillatory',
    seed=0,
    config=CONFIGS['A5'],
    max_steps=200,
)

fleet = orch.env.fleet
net = orch.env.network
cqm = orch.cqm
acds = orch.acds

steps = []
q_list = []
loss_list = []
lat_list = []
bw_list = []
cqi_list = []
mode_list = []
switch_list = []

print("Step | Raw Q(t) | Packet Loss | Latency(s) | Bandwidth Util | System CQI | Architecture Mode | Switch Count")
print("-" * 105)

for step in range(200):
    dist_mat = distance_matrix(fleet.agents)
    for node_id in range(fleet.n_agents):
        net_state = net.simulate_message(step)
        cqm.update_from_network(node_id, net_state)
    
    cqi_matrix = cqm.update_pairwise(dist_mat, orch.thresholds.get('C1', 50.0))
    sys_cqi = cqm.system_cqi()
    mode_before = acds.mode
    mode_after = acds.evaluate(sys_cqi)
    
    q_val = net._channel_quality(step)
    loss_val = net_state.packet_loss_rate
    lat_val = net_state.latency
    bw_val = net_state.bandwidth_utilization
    sw_cnt = acds.switch_count
    
    mode_str = "Centralized(0)" if mode_after == 0 else "Decentralized(1)"
    
    steps.append(step)
    q_list.append(q_val)
    loss_list.append(loss_val)
    lat_list.append(lat_val)
    bw_list.append(bw_val)
    cqi_list.append(sys_cqi)
    mode_list.append(mode_str)
    switch_list.append(sw_cnt)
    
    print(f"{step:4d} | {q_val:9.3f} | {loss_val:11.3f} | {lat_val:10.3f}s | {bw_val:14.3f} | {sys_cqi:10.3f} | {mode_str:17s} | {sw_cnt:12d}")

# Save network_trace.csv
net_csv = ARTIFACT_DIR / "network_trace.csv"
with open(net_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'raw_channel_quality', 'packet_loss', 'latency', 'bandwidth'])
    for i in range(200):
        writer.writerow([steps[i], q_list[i], loss_list[i], lat_list[i], bw_list[i]])
print(f"\nSaved {net_csv}")

# Save cqi_trace.csv
cqi_csv = ARTIFACT_DIR / "cqi_trace.csv"
with open(cqi_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'system_cqi', 'theta_down', 'theta_up'])
    for i in range(200):
        writer.writerow([steps[i], cqi_list[i], acds.theta_down, acds.theta_up])
print(f"Saved {cqi_csv}")

# Save switch_trace.csv
sw_csv = ARTIFACT_DIR / "switch_trace.csv"
with open(sw_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'architecture_mode', 'switch_count'])
    for i in range(200):
        writer.writerow([steps[i], mode_list[i], switch_list[i]])
print(f"Saved {sw_csv}")

print("\n=== STEP 2: GENERATING PLOTS ===")

# Plot 1: Raw Channel Quality
plt.figure(figsize=(10, 4))
plt.plot(steps, q_list, color='#1f77b4', linewidth=2, label='Raw Channel Quality Q(t)')
plt.title('Physical Wireless Channel Quality Q(t) Over Time')
plt.xlabel('Simulation Step')
plt.ylabel('Channel Quality Q(t)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
q_plot = ARTIFACT_DIR / "raw_channel_quality.png"
plt.savefig(q_plot, dpi=150)
plt.close()
print(f"Saved {q_plot}")

# Plot 2: Packet Loss
plt.figure(figsize=(10, 4))
plt.plot(steps, loss_list, color='#d62728', linewidth=2, label='Packet Loss Rate')
plt.title('Wireless Packet Loss Rate Over Time')
plt.xlabel('Simulation Step')
plt.ylabel('Packet Loss Rate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
loss_plot = ARTIFACT_DIR / "packet_loss.png"
plt.savefig(loss_plot, dpi=150)
plt.close()
print(f"Saved {loss_plot}")

# Plot 3: Latency
plt.figure(figsize=(10, 4))
plt.plot(steps, lat_list, color='#ff7f0e', linewidth=2, label='Latency (seconds)')
plt.title('Wireless Communication Latency Over Time')
plt.xlabel('Simulation Step')
plt.ylabel('Latency (s)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
lat_plot = ARTIFACT_DIR / "latency.png"
plt.savefig(lat_plot, dpi=150)
plt.close()
print(f"Saved {lat_plot}")

# Plot 4: Bandwidth Utilization
plt.figure(figsize=(10, 4))
plt.plot(steps, bw_list, color='#9467bd', linewidth=2, label='Bandwidth Utilization')
plt.title('Wireless Bandwidth Utilization Over Time')
plt.xlabel('Simulation Step')
plt.ylabel('Bandwidth Utilization')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
bw_plot = ARTIFACT_DIR / "bandwidth.png"
plt.savefig(bw_plot, dpi=150)
plt.close()
print(f"Saved {bw_plot}")

# Plot 5: System CQI with ThetaDown & ThetaUp Overlay
plt.figure(figsize=(10, 4))
plt.plot(steps, cqi_list, color='#2ca02c', linewidth=2, label='System CQI')
plt.axhline(y=acds.theta_down, color='red', linestyle='--', linewidth=1.8, label=f'ThetaDown ({acds.theta_down:.2f})')
plt.axhline(y=acds.theta_up, color='blue', linestyle='--', linewidth=1.8, label=f'ThetaUp ({acds.theta_up:.2f})')
plt.title('Communication Quality Index (CQI) with ACDS Hysteresis Thresholds')
plt.xlabel('Simulation Step')
plt.ylabel('System CQI')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.legend(loc='upper right')
plt.tight_layout()
cqi_plot = ARTIFACT_DIR / "system_cqi.png"
plt.savefig(cqi_plot, dpi=150)
plt.close()
print(f"Saved {cqi_plot}")

print("\n=== STEP 3: RUNNING EXPERIMENTS ACROSS SEEDS 0-4 AND SCENARIOS ===")
from subprocess import run, PIPE

results_all = {}

scenarios = ['search_rescue', 'inspection', 'logistics']
profiles = ['oscillatory']

for scenario in scenarios:
    for seed in range(5):
        cmd = [
            sys.executable, "experiments/run_daca_hmas.py",
            "--config", "A5",
            "--scenario", scenario,
            "--profile", "oscillatory",
            "--seed", str(seed)
        ]
        res = run(cmd, capture_output=True, text=True)
        json_path = ROOT / "experiments" / "results" / f"summary_A5_{scenario}_oscillatory.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            key = f"{scenario}_oscillatory_seed_{seed}"
            results_all[key] = data
            print(f"[{key}] Success={data.get('success_rate')}%, SC={data.get('switch_count')}, PeerMsgs={data.get('peer_messages')}, Cloud={data.get('cloud_planning_calls')}, Device={data.get('device_planning_calls')}, Tokens={data.get('tokens')}, API={data.get('api_calls')}")

print("\n=== STEP 4: VERIFYING STABLE, GRADUAL, SUDDEN PROFILES ===")
for profile in ['stable', 'gradual', 'sudden']:
    cmd = [
        sys.executable, "experiments/run_daca_hmas.py",
        "--config", "A5",
        "--scenario", "search_rescue",
        "--profile", profile,
        "--seed", "0"
    ]
    res = run(cmd, capture_output=True, text=True)
    json_path = ROOT / "experiments" / "results" / f"summary_A5_search_rescue_{profile}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        key = f"search_rescue_{profile}_seed_0"
        results_all[key] = data
        print(f"[{key}] Success={data.get('success_rate')}%, SC={data.get('switch_count')}, PeerMsgs={data.get('peer_messages')}, Cloud={data.get('cloud_planning_calls')}, Device={data.get('device_planning_calls')}")

with open(ARTIFACT_DIR / "all_execution_results.json", 'w') as f:
    json.dump(results_all, f, indent=2)

print("\nALL RUNS COMPLETED SUCCESSFULLY!")
