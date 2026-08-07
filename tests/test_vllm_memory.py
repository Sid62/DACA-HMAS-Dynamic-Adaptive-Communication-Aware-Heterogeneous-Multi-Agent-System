"""Unit tests for Python heap delta and device token load proxy memory metrics."""

import pytest
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector


def test_vllm_memory_fields_serialization():
    """Verify that python heap delta and token load proxy fields serialize cleanly in to_dict()."""
    metrics = ExperimentMetrics(
        config_name="A5",
        scenario="logistics",
        network_profile="stable",
        seed=0,
        success_rate=1.0,
        steps=10,
        cloud_tokens=100,
        device_tokens=50,
        total_tokens=150,
        cloud_api_calls=2,
        device_api_calls=5,
        total_api_calls=7,
        device_memory_mb=45.5,
        computation_s=0.5,
        device_llm_python_heap_delta_mb=1.25,
        device_llm_python_heap_delta_by_device={"uav": 1.25, "robot": 0.8},
        device_llm_tokens_processed_by_device={"uav": 30, "robot": 20},
    )

    d = metrics.to_dict()
    assert "device_llm_python_heap_delta_mb" in d
    assert "device_llm_python_heap_delta_by_device" in d
    assert "device_llm_tokens_processed_by_device" in d

    assert d["device_llm_python_heap_delta_mb"] == 1.25
    assert d["device_llm_python_heap_delta_by_device"] == {"uav": 1.25, "robot": 0.8}
    assert d["device_llm_tokens_processed_by_device"] == {"uav": 30, "robot": 20}
    assert d["memory_mb"] == 45.5


def test_orchestrator_vllm_memory_recording():
    """Run a short DACAOrchestrator mission in mock mode and verify metrics recording."""
    orch = DACAOrchestrator(
        scenario="inspection",
        network_profile="oscillatory",
        seed=0,
        config=CONFIGS["A5"],
        max_steps=5,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d in orch.device_llms.values():
        d.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()

    assert "device_llm_python_heap_delta_mb" in d
    assert "device_llm_python_heap_delta_by_device" in d
    assert "device_llm_tokens_processed_by_device" in d

    assert d["device_llm_python_heap_delta_mb"] >= 0.0
    assert isinstance(d["device_llm_python_heap_delta_by_device"], dict)
    assert isinstance(d["device_llm_tokens_processed_by_device"], dict)
    for key in orch.device_llms.keys():
        assert key in metrics.device_llm_python_heap_delta_by_device
        assert metrics.device_llm_python_heap_delta_by_device[key] >= 0.0
        assert key in metrics.device_llm_tokens_processed_by_device
        assert metrics.device_llm_tokens_processed_by_device[key] >= 0
