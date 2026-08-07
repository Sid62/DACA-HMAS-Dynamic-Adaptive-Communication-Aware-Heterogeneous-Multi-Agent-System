"""Empirical probe for Optimization B: Coalition retry loop analysis.
Measures attempt0, attempt1, attempt2 call counts, coalition feasibility scores (CFR),
and mission success rates across logistics, inspection, and search_rescue scenarios.
"""
import time
import numpy as np
from src.coordination.orchestrator import DACAOrchestrator, CONFIGS
from src.config import get_llm_config

def run_probe_b():
    scenarios = ["logistics", "inspection", "search_rescue"]
    seeds = [0, 1, 2]
    
    print("=" * 70)
    print("PHASE 1 — OPTIMIZATION B EMPIRICAL PROBE")
    print("=" * 70)
    
    results = {}
    
    for sc in scenarios:
        results[sc] = {
            "attempts": {0: 0, 1: 0, 2: 0},
            "attempt_details": [],
            "success_rates": [],
            "cloud_calls": [],
        }
        for seed in seeds:
            orchestrator = DACAOrchestrator(sc, "oscillatory", seed, CONFIGS["A5"], max_steps=50)
            orchestrator.cloud_llm.config["use_mock"] = True
            for d_client in orchestrator.device_llms.values():
                d_client.config["use_mock"] = True
            
            attempt_counts = {0: 0, 1: 0, 2: 0}
            attempt_cfrs = []
            
            def form_interceptor(fleet, subtasks, distance_matrix, cqi_matrix):
                from src.coalition.feasibility import build_psi_matrix, validate_coalition_members
                psi = build_psi_matrix(distance_matrix, cqi_matrix, orchestrator.coalition_formation.c1)
                id_to_idx = {a.agent_id: i for i, a in enumerate(fleet.agents)}
                agents_ctx = fleet.to_dict_list()
                subtasks_ctx = [{"id": s.subtask_id, "skills": s.required_skills} for s in subtasks]
                
                coalitions = []
                for attempt in range(orchestrator.coalition_formation.max_retries):
                    attempt_counts[attempt] = attempt_counts.get(attempt, 0) + 1
                    raw = orchestrator.cloud_llm.form_coalitions(
                        subtasks_ctx, agents_ctx, distance_matrix.tolist(), cqi_matrix.tolist()
                    )
                    coalitions = []
                    infeasible = []
                    for c in raw:
                        members = c.get("members", [])
                        if validate_coalition_members(members, id_to_idx, psi, orchestrator.coalition_formation.gamma_min):
                            coalitions.append(c)
                        else:
                            infeasible.append(c)
                    
                    cfr_before_repair = orchestrator.coalition_formation.compute_cfr(coalitions, fleet, distance_matrix, cqi_matrix)
                    
                    if not infeasible:
                        attempt_cfrs.append((attempt, cfr_before_repair, cfr_before_repair))
                        break
                    
                    repaired = orchestrator.coalition_formation._repair_infeasible(infeasible, fleet, psi, id_to_idx)
                    coalitions.extend(repaired)
                    cfr_after_repair = orchestrator.coalition_formation.compute_cfr(coalitions, fleet, distance_matrix, cqi_matrix)
                    attempt_cfrs.append((attempt, cfr_before_repair, cfr_after_repair))
                
                coalitions = orchestrator.coalition_formation._merge_singleton_coalitions(coalitions, fleet, psi, id_to_idx)
                if not coalitions:
                    coalitions = [{"coalition_id": i, "members": [a.agent_id]} for i, a in enumerate(fleet.agents)]
                coalitions = orchestrator.coalition_formation._stabilize_coalition_ids(coalitions)
                for c in coalitions:
                    for mid in c.get("members", []):
                        if mid in id_to_idx:
                            fleet.agents[id_to_idx[mid]].coalition_id = c.get("coalition_id")
                return coalitions
            
            orchestrator.coalition_formation.form = form_interceptor
            metrics = orchestrator.run()
            
            results[sc]["success_rates"].append(metrics.mission_success if hasattr(metrics, "mission_success") else metrics.success_rate)
            results[sc]["cloud_calls"].append(metrics.cloud_api_calls)
            for k, v in attempt_counts.items():
                results[sc]["attempts"][k] += v
            results[sc]["attempt_details"].extend(attempt_cfrs)
            
    print("\n--- PROBE B SUMMARY RESULTS ---")
    for sc, data in results.items():
        print(f"\nScenario: {sc.upper()}")
        print(f"  Attempt Counts: {data['attempts']}")
        print(f"  Mean Cloud API Calls: {np.mean(data['cloud_calls']):.2f}")
        print(f"  Mean Success Rate: {np.mean(data['success_rates']) * 100:.1f}%")
        
        # Analyze feasibility delta
        cfr_rep = [post for (att, pre, post) in data["attempt_details"] if pre != post or att > 0]
        print(f"  Total Re-query/Repair events: {len(cfr_rep)}")
        if data["attempt_details"]:
            pre_avg = np.mean([pre for (att, pre, post) in data["attempt_details"]])
            post_avg = np.mean([post for (att, pre, post) in data["attempt_details"]])
            print(f"  Avg Pre-Repair CFR: {pre_avg:.4f} | Avg Post-Repair CFR: {post_avg:.4f}")

if __name__ == "__main__":
    run_probe_b()
