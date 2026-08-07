"""Comprehensive BEFORE vs AFTER verification suite across logistics, inspection, and search_rescue.
Measures:
- Cloud API Calls
- Success Rate (%)
- Communication Steps (and paper_communication_steps)
- Computation Time (s)
- Memory (MB)
- Switch Count
- Coalition Quality (CFR)
- Planning Latency (s)
- Peer Messages
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import numpy as np

from src.coordination.orchestrator import DACAOrchestrator, CONFIGS

def run_verification():
    scenarios = ["logistics", "inspection", "search_rescue"]
    seeds = [0, 1, 2]
    
    print("=" * 80)
    print("PHASE 3 & 4 — FULL EXPERIMENTAL VERIFICATION SUITE")
    print("=" * 80)
    
    results = {}
    
    for sc in scenarios:
        results[sc] = {
            "success_rates": [],
            "cloud_calls": [],
            "comm_steps": [],
            "paper_comm_steps": [],
            "comp_time": [],
            "memory": [],
            "switches": [],
            "cfr": [],
            "planning_latency": [],
            "peer_messages": [],
            "breakdown": {
                "initial": 0,
                "completion": 0,
                "cqi": 0,
                "packet_loss": 0,
                "latency": 0,
                "switch": 0,
                "coalition_retry": 0,
                "hallucination_retry": 0,
                "cache_hits": 0,
            }
        }
        for seed in seeds:
            orchestrator = DACAOrchestrator(sc, "oscillatory", seed, CONFIGS["A5"], max_steps=50)
            orchestrator.cloud_llm.config["use_mock"] = True
            orchestrator.cloud_llm.config["cache_responses"] = False
            orchestrator.cloud_llm.semantic_cache.enabled = False
            for d_client in orchestrator.device_llms.values():
                d_client.config["use_mock"] = True
                d_client.config["cache_responses"] = False
                
            metrics = orchestrator.run()
            
            # Phase 2 Instrumentation assertion
            usage = orchestrator.cloud_llm.usage
            init_calls = getattr(usage, "initial_planning_calls", 0)
            comp_calls = getattr(usage, "completion_replan_calls", 0)
            cqi_calls = getattr(usage, "cqi_replan_calls", 0)
            pl_calls = getattr(usage, "packet_loss_replan_calls", 0)
            lat_calls = getattr(usage, "latency_replan_calls", 0)
            sw_calls = getattr(usage, "switch_replan_calls", 0)
            coal_calls = getattr(usage, "coalition_retry_calls", 0)
            hal_calls = getattr(usage, "hallucination_retry_calls", 0)
            
            cat_sum = (
                init_calls + comp_calls + cqi_calls + pl_calls +
                lat_calls + sw_calls + coal_calls + hal_calls
            )
            print(f"[VERIFY] Scenario={sc} Seed={seed}: Category Sum ({cat_sum}) == Cloud API Calls ({usage.cloud_api_calls})")
            assert cat_sum == usage.cloud_api_calls, f"Category sum mismatch: {cat_sum} != {usage.cloud_api_calls}"
            
            results[sc]["success_rates"].append(metrics.success_rate)
            results[sc]["cloud_calls"].append(metrics.cloud_api_calls)
            results[sc]["comm_steps"].append(metrics.communication_steps)
            results[sc]["paper_comm_steps"].append(metrics.paper_communication_steps)
            results[sc]["comp_time"].append(metrics.computation_s)
            results[sc]["memory"].append(metrics.device_memory_mb)
            results[sc]["switches"].append(metrics.switch_count)
            results[sc]["cfr"].append(metrics.cfr)
            results[sc]["planning_latency"].append(metrics.avg_planning_latency)
            results[sc]["peer_messages"].append(metrics.peer_messages)
            
            results[sc]["breakdown"]["initial"] += init_calls
            results[sc]["breakdown"]["completion"] += comp_calls
            results[sc]["breakdown"]["cqi"] += cqi_calls
            results[sc]["breakdown"]["packet_loss"] += pl_calls
            results[sc]["breakdown"]["latency"] += lat_calls
            results[sc]["breakdown"]["switch"] += sw_calls
            results[sc]["breakdown"]["coalition_retry"] += coal_calls
            results[sc]["breakdown"]["hallucination_retry"] += hal_calls
            results[sc]["breakdown"]["cache_hits"] += getattr(usage, "cache_hits", 0)

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON REPORT")
    print("=" * 80)
    
    for sc, data in results.items():
        print(f"\n### Scenario: {sc.upper()}")
        print(f"  - Mean Cloud API Calls : {np.mean(data['cloud_calls']):.2f}")
        print(f"  - Mean Success Rate    : {np.mean(data['success_rates']) * 100:.2f}%")
        print(f"  - Mean Comm Steps      : {np.mean(data['comm_steps']):.1f} (Paper Comm Steps: {np.mean(data['paper_comm_steps']):.1f})")
        print(f"  - Mean Comp Time (s)   : {np.mean(data['comp_time']):.4f}s")
        print(f"  - Mean Memory (MB)     : {np.mean(data['memory']):.1f} MB")
        print(f"  - Mean Switch Count    : {np.mean(data['switches']):.1f}")
        print(f"  - Mean Coalition CFR   : {np.mean(data['cfr']):.4f}")
        print(f"  - Mean Plan Latency (s): {np.mean(data['planning_latency']):.4f}s")
        print(f"  - Mean Peer Messages   : {np.mean(data['peer_messages']):.1f}")
        print(f"  - API Breakdown        : {data['breakdown']}")

if __name__ == "__main__":
    run_verification()
