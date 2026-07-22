import sys, json
from pathlib import Path
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import distance_matrix

orch = DACAOrchestrator(
    scenario='search_rescue',
    network_profile='oscillatory',
    seed=0,
    config=CONFIGS['A5'],
    max_steps=200,
)

fleet = orch.env.fleet
acds = orch.acds
cqm = orch.cqm
net = orch.env.network

print("=== PHASE 1-7 BASELINE RUNTIME TRACE (A5 search_rescue oscillatory s0) ===")
print(f"Initial ACDS: theta_down={acds.theta_down:.3f}, theta_up={acds.theta_up:.3f}, persistence_window={acds.persistence_window}")

step_records = []

for step in range(200):
    dist_mat = distance_matrix(fleet.agents)
    
    for node_id in range(fleet.n_agents):
        net_state = net.simulate_message(step)
        cqm.update_from_network(node_id, net_state)
    
    cqi_matrix = cqm.update_pairwise(dist_mat, orch.thresholds.get('C1', 50.0))
    sys_cqi = cqm.system_cqi()
    
    q_t = net._channel_quality(step)
    loss = net.loss_rate_at(step)
    lat = net.latency_at(step)
    bw = net.bandwidth_at(step)
    
    recent_before = list(acds.cqi_history)[-acds.persistence_window:] if len(acds.cqi_history) >= acds.persistence_window else list(acds.cqi_history)
    c_less_down = all(c < acds.theta_down for c in recent_before) if len(recent_before) == acds.persistence_window else False
    c_greater_up = all(c > acds.theta_up for c in recent_before) if len(recent_before) == acds.persistence_window else False
    
    mode_before = acds.mode
    mode_after = acds.evaluate(sys_cqi)
    switched = (mode_before != mode_after)
    
    rec = {
        'step': step,
        'mode_before': mode_before,
        'mode_after': mode_after,
        'sys_cqi': sys_cqi,
        'q_t': q_t,
        'loss': loss,
        'latency': lat,
        'bw': bw,
        'theta_down': acds.theta_down,
        'theta_up': acds.theta_up,
        'c_less_down': c_less_down,
        'c_greater_up': c_greater_up,
        'switches': acds.switch_count,
    }
    step_records.append(rec)
    
    mode_str = 'Centralized(0)' if mode_after == 0 else 'Decentralized(1)'
    decision_str = 'SWITCH (0->1)' if (switched and mode_after==1) else ('SWITCH (1->0)' if (switched and mode_after==0) else 'HOLD')
    
    if step < 25 or step % 5 == 0 or switched:
        print(f"Step {step:3d} | Q(t)={q_t:.3f} Loss={loss:.3f} Lat={lat:.3f}s BW={bw:.3f} | sys_cqi={sys_cqi:.3f} | Theta=[{acds.theta_down:.2f}, {acds.theta_up:.2f}] | Mode: {mode_before}->{mode_after} ({mode_str}) | c<Down:{str(c_less_down):5s} c>Up:{str(c_greater_up):5s} | Decision: {decision_str:14s} | TotalSwitches={acds.switch_count}")

print("\n=== SUMMARY OF ALL SWITCH EVENTS ===")
switches = [r for r in step_records if r['mode_before'] != r['mode_after']]
if not switches:
    print("No mode switches occurred.")
else:
    for s in switches:
        print(f"Step {s['step']}: Mode {s['mode_before']} -> {s['mode_after']} | sys_cqi={s['sys_cqi']:.3f} | ThetaDown={s['theta_down']:.2f} ThetaUp={s['theta_up']:.2f}")

print(f"\nMax CQI achieved across 200 steps: {max(r['sys_cqi'] for r in step_records):.3f}")
print(f"Min CQI achieved across 200 steps: {min(r['sys_cqi'] for r in step_records):.3f}")
