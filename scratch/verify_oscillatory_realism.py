import sys, json
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import distance_matrix

print("=== REALISTIC WIRELESS NETWORK MODEL VERIFICATION ===")

for scenario in ['search_rescue', 'inspection', 'logistics']:
    print(f"\n--- SCENARIO: {scenario.upper()} (OSCILLATORY PROFILE) ---")
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile='oscillatory',
        seed=0,
        config=CONFIGS['A5'],
        max_steps=200,
    )
    fleet = orch.env.fleet
    net = orch.env.network
    cqm = orch.cqm
    acds = orch.acds
    
    for step in range(200):
        dist_mat = distance_matrix(fleet.agents)
        for node_id in range(fleet.n_agents):
            net_state = net.simulate_message(step)
            cqm.update_from_network(node_id, net_state)
        
        cqi_matrix = cqm.update_pairwise(dist_mat, orch.thresholds.get('C1', 50.0))
        sys_cqi = cqm.system_cqi()
        mode_before = acds.mode
        mode_after = acds.evaluate(sys_cqi)
        switched = (mode_before != mode_after)
        
        if step < 15 or step in [20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 120, 140, 160, 180, 199] or switched:
            print(f"Step {step:3d} | Loss={net_state.packet_loss_rate:.3f} | Latency={net_state.latency:.3f}s | BW={net_state.bandwidth_utilization:.3f} | Q(t)={net._channel_quality(step):.3f} | sys_cqi={sys_cqi:.3f} | Mode:{mode_before}->{mode_after} | SC={acds.switch_count}")
