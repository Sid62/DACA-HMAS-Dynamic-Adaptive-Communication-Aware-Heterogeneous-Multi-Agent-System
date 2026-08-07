#!/usr/bin/env python3
"""Fast diagnostic: NO LLM calls. Only traces coalition structure + CQI + ACDS."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.env.agents import distance_matrix, create_fleet_from_scenario
from src.env.scenarios import get_scenario
from src.env.network_conditions import NetworkConditionGenerator
from src.communication.models import discover_agent_type_domains, domains_in_coalition, dominant_domain_for_coalition
from src.cqm.monitor import CommunicationQualityMonitor
from src.acds.switch_engine import ACDSSwitchEngine
from src.config import get_thresholds

thresholds = get_thresholds()

for scenario_name in ["logistics", "search_rescue"]:
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    scenario = get_scenario(scenario_name, thresholds, seed=1)
    fleet = create_fleet_from_scenario(
        scenario.agent_config,
        thresholds.get("kinematics", {}),
        c1=thresholds.get("C1", 50.0),
        c2=thresholds.get("C2", 5.0),
        seed=1,
    )
    
    print(f"Fleet size: {fleet.n_agents}")
    domains = discover_agent_type_domains(fleet)
    print(f"Agent-type domains:")
    for domain, agents in domains.items():
        print(f"  {domain}: {agents}")
    
    print(f"\nSubtasks: {len(scenario.subtasks)}")
    
    # --- Simulate what CoalitionFormation._merge_singleton_coalitions does ---
    # The mock LLM typically assigns one agent per coalition (singletons)
    # Then _merge_singleton_coalitions groups same-type singletons together
    print(f"\n--- Simulating singleton merge (what mock LLM produces) ---")
    
    # Mock LLM output: each agent gets its own coalition
    mock_coalitions = [
        {"coalition_id": i, "members": [a.agent_id]}
        for i, a in enumerate(fleet.agents)
    ]
    
    # Group by domain (what _merge_singleton_coalitions does)
    domain_of = {a.agent_id: a.agent_type.value for a in fleet.agents}
    domain_groups = {}
    for c in mock_coalitions:
        mid = c["members"][0]
        d = domain_of[mid]
        domain_groups.setdefault(d, []).append(mid)
    
    print(f"After singleton merge, expected coalitions:")
    for domain, members in domain_groups.items():
        c_domains = domains_in_coalition(members, fleet)
        leader = dominant_domain_for_coalition(members, fleet)
        peer_domains = [d for d in c_domains if d != leader]
        print(f"  domain={domain} members={members} "
              f"coalition_domains={c_domains} leader={leader} "
              f"peer_domains={peer_domains}")
        if len(peer_domains) == 0:
            print(f"    >>> ISSUE 1 ROOT CAUSE: peer_domains is EMPTY!")
            print(f"    >>> All {len(members)} members are same type '{domain}'")
            print(f"    >>> broadcast() in _run_distributed_coalition_planning sends to ALL registered peers")
            print(f"    >>> BUT the peer REVIEW loop (line 283) iterates only over coalition domains")
            print(f"    >>> Since there's only 1 domain in this coalition, the review loop body NEVER executes")
            print(f"    >>> => no send_message() calls from peers => peer_messages stays 0")
    
    # --- CQI / ACDS trace ---
    print(f"\n--- CQI / ACDS trace (seed=1, 100 steps, oscillatory) ---")
    net = NetworkConditionGenerator.from_scenario(
        scenario_name, "oscillatory", thresholds, seed=1, total_steps=100
    )
    net.fleet = fleet
    cqm = CommunicationQualityMonitor.from_config(thresholds, fleet.n_agents)
    acds = ACDSSwitchEngine.from_config(thresholds, use_hysteresis=True)
    print(f"theta_down={acds.theta_down:.4f} theta_up={acds.theta_up:.4f} "
          f"persistence_window={acds.persistence_window}")
    
    cqi_values = []
    switches = []
    for t in range(100):
        for node_id in range(fleet.n_agents):
            net_state = net.simulate_message(t)
            cqm.update_from_network(node_id, net_state)
        cqi = cqm.system_cqi()
        cqi_values.append(cqi)
        mode_before = acds.mode
        mode_after = acds.evaluate(cqi)
        if mode_before != mode_after:
            recent = list(acds.cqi_history)[-acds.persistence_window:]
            switches.append({
                "step": t, "cqi": cqi, "mode": f"{mode_before}->{mode_after}",
                "switch_count": acds.switch_count,
                "recent_cqi": [f"{v:.4f}" for v in recent],
            })
    
    print(f"CQI range: [{min(cqi_values):.4f}, {max(cqi_values):.4f}]")
    print(f"CQI mean: {np.mean(cqi_values):.4f}")
    print(f"Final switch_count: {acds.switch_count}")
    
    print(f"\nSwitch events:")
    for sw in switches:
        print(f"  step={sw['step']} CQI={sw['cqi']:.4f} {sw['mode']} "
              f"SC={sw['switch_count']}")
        print(f"    recent_cqi={sw['recent_cqi']}")
        threshold = acds.theta_down if "0->1" in sw["mode"] else acds.theta_up
        label = "theta_down" if "0->1" in sw["mode"] else "theta_up"
        all_below = all(float(v) < acds.theta_down for v in sw["recent_cqi"])
        all_above = all(float(v) > acds.theta_up for v in sw["recent_cqi"])
        if "0->1" in sw["mode"]:
            print(f"    Valid switch? All recent < theta_down({acds.theta_down:.4f})? {all_below}")
        else:
            print(f"    Valid switch? All recent > theta_up({acds.theta_up:.4f})? {all_above}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
