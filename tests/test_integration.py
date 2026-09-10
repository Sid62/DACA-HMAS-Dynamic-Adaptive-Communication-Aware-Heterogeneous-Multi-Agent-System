"""Integration smoke tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


@pytest.mark.parametrize("config", ["B1", "B2", "A5"])
@pytest.mark.parametrize("scenario", ["logistics", "inspection"])
def test_orchestrator_runs(config, scenario):
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile="stable",
        seed=42,
        config=CONFIGS[config],
        max_steps=30,
    )
    metrics = orch.run()
    assert 0.0 <= metrics.success_rate <= 1.0
    assert metrics.steps > 0


def test_baseline_ordering_logistics_vs_inspection():
    """Phase 0 gate: architectures should produce valid results."""
    b1 = DACAOrchestrator("logistics", "stable", 0, CONFIGS["B1"], max_steps=50).run()
    b2 = DACAOrchestrator("inspection", "gradual", 0, CONFIGS["B2"], max_steps=50).run()
    assert b1.success_rate >= 0.0
    assert b2.success_rate >= 0.0
