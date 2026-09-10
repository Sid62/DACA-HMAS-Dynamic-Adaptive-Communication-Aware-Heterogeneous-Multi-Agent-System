"""Determinism Audit Unit Test for DACA-HMAS Recovery Pipeline."""

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def test_end_to_end_determinism_same_seed(tmp_path):
    """Verify that running DACAOrchestrator twice with the exact same seed produces bit-for-bit identical metrics."""
    orch1 = DACAOrchestrator(
        scenario="logistics",
        network_profile="oscillatory",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=50,
    )
    orch1.cloud_llm.config["cache_dir"] = str(tmp_path / "c1")
    for dc in orch1.device_llms.values():
        dc.config["cache_dir"] = str(tmp_path / "c1")
    metrics1 = orch1.run().to_dict()

    orch2 = DACAOrchestrator(
        scenario="logistics",
        network_profile="oscillatory",
        seed=1,
        config=CONFIGS["A5"],
        max_steps=50,
    )
    orch2.cloud_llm.config["cache_dir"] = str(tmp_path / "c2")
    for dc in orch2.device_llms.values():
        dc.config["cache_dir"] = str(tmp_path / "c2")
    metrics2 = orch2.run().to_dict()

    # Compare key metrics for 100% identity
    assert metrics1["success_rate"] == metrics2["success_rate"]
    assert metrics1["steps"] == metrics2["steps"]
    assert metrics1["tokens"] == metrics2["tokens"]
    assert metrics1["api_calls"] == metrics2["api_calls"]
    assert metrics1["switch_count"] == metrics2["switch_count"]
    assert metrics1["communication_steps"] == metrics2["communication_steps"]
    assert metrics1["communication_step_breakdown"] == metrics2["communication_step_breakdown"]
    assert metrics1["hallucination_stats"] == metrics2["hallucination_stats"]
