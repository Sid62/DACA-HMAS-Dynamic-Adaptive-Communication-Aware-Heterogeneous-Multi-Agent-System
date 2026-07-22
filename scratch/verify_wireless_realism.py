import sys, json
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import distance_matrix

print("=== VERIFYING REALISTIC WIRELESS NETWORK MODEL ===")

for scenario in ['search_rescue', 'inspection', 'logistics']:
    for profile in ['oscillatory', 'stable', 'gradual', 'sudden']:
        orch = DACAOrchestrator(
            scenario=scenario,
            network_profile=profile,
            seed=0,
            config=CONFIGS['A5'],
            max_steps=200,
        )
        fleet = orch.env.fleet
        net = orch.env.network
        cqm = orch.cqm
        acds = orch.acds
        
        cqi_history = []
        loss_history = []
        lat_history = []
        bw_history = []
        q_history = []
        
        for step in range(200):
            dist_mat = distance_matrix(fleet.agents)
            for node_id in range(fleet.n_agents):
                net_state = net.simulate_message(step)
                cqm.update_from_network(node_id, net_state)
            
            cqi_matrix = cqm.update_pairwise(dist_mat, orch.thresholds.get('C1', 50.0))
            sys_cqi = cqm.system_cqi()
            acds.evaluate(sys_cqi)
            
            loss_history.append(net_state.packet_loss_rate)
            lat_history.append(net_state.latency)
            bw_history.append(net_state.bandwidth_utilization)
            q_history.append(net._channel_quality(step))
            cqi_history.append(sys_cqi)
            
        print(f"Scenario: {scenario:13s} | Profile: {profile:11s} | Switches: {acds.switch_count:2d} | CQI Range: [{min(cqi_history):.3f}, {max(cqi_history):.3f}] | Loss Range: [{min(loss_history):.3f}, {max(loss_history):.3f}] | Latency Range: [{min(lat_history):.3f}s, {max(lat_history):.3f}s]")
