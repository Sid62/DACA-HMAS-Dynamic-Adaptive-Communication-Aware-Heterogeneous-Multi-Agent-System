#!/usr/bin/env python3
"""Test fix for domain link CQI resolution in PeerCommunicationManager."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator

for scenario in ["logistics", "search_rescue"]:
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile="oscillatory",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=100,
    )
    metrics = orch.run()
    print(f"\nResults for {scenario}:")
    print(f"  success_rate: {metrics.success_rate:.2f}%")
    print(f"  steps: {metrics.steps}")
    print(f"  switch_count: {metrics.switch_count}")
    print(f"  peer_messages: {metrics.peer_messages}")
    print(f"  broadcast_count: {metrics.broadcast_count}")
    print(f"  consensus_rounds: {metrics.consensus_rounds}")
    print(f"  distributed_replanning_count: {metrics.distributed_replanning_count}")
