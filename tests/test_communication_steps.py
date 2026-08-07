"""Unit tests for CommunicationStepCounter and communication_steps metric integration."""

import numpy as np
from src.metrics.communication_counter import CommunicationStepCounter
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def test_communication_step_counter():
    """Verify CommunicationStepCounter functionality."""
    counter = CommunicationStepCounter()
    assert counter.value == 0
    assert counter.paper_value == 0

    counter.increment("global_planning", 1, "test_event")
    assert counter.value == 1
    assert counter.paper_value == 1

    counter.increment("dispatch", 2, "multi_event")
    assert counter.value == 3
    assert counter.paper_value == 3

    counter.increment("local_coordination", 5, "local_event")
    assert counter.value == 8
    assert counter.paper_value == 3  # paper_value tracks only global_planning + dispatch

    counter.reset()
    assert counter.value == 0
    assert counter.paper_value == 0


def test_experiment_metrics_dict_export():
    """Verify that communication_steps and paper_communication_steps are serialized in to_dict()."""
    metrics = ExperimentMetrics(
        config_name="A5",
        scenario="logistics",
        network_profile="stable",
        seed=42,
        success_rate=1.0,
        steps=15,
        cloud_tokens=500,
        device_tokens=300,
        total_tokens=800,
        cloud_api_calls=2,
        device_api_calls=4,
        total_api_calls=6,
        device_memory_mb=128.5,
        computation_s=1.234,
        switch_count=3,
        communication_steps=12,
        paper_communication_steps=4,
        communication_step_breakdown={"global_planning": 2, "dispatch": 2, "local_coordination": 8},
    )

    d = metrics.to_dict()
    assert "communication_steps" in d
    assert d["communication_steps"] == 12
    assert "paper_communication_steps" in d
    assert d["paper_communication_steps"] == 4

    # Check that existing fields maintain exact definitions and formats
    assert d["config"] == "A5"
    assert d["scenario"] == "logistics"
    assert d["profile"] == "stable"
    assert d["seed"] == 42
    assert d["success_rate"] == 100.0
    assert d["steps"] == 15
    assert d["tokens"] == 800
    assert d["api_calls"] == 6
    assert d["memory_mb"] == 128.5
    assert d["computation_s"] == 1.234
    assert d["switch_count"] == 3


def test_orchestrator_communication_steps_recording():
    """Verify that DACAOrchestrator tracks communication_steps and paper_communication_steps."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=0,
        config=CONFIGS["A5"],
        max_steps=20,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()

    assert "communication_steps" in d
    assert isinstance(d["communication_steps"], int)
    assert d["communication_steps"] > 0, "communication_steps must be non-zero after mission run"
    assert "paper_communication_steps" in d
    assert isinstance(d["paper_communication_steps"], int)
    assert "communication_step_breakdown" in d
    assert isinstance(d["communication_step_breakdown"], dict)

    breakdown = d["communication_step_breakdown"]
    expected_paper_steps = breakdown.get("global_planning", 0) + breakdown.get("dispatch", 0)
    assert d["paper_communication_steps"] == expected_paper_steps


