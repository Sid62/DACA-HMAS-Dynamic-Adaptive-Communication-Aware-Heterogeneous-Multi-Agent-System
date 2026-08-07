"""Backward compatibility wrapper for memory metric tests."""

from tests.test_vllm_memory import (
    test_vllm_memory_fields_serialization as test_experiment_metrics_to_dict_isolated_memory,
    test_orchestrator_vllm_memory_recording as test_orchestrator_isolated_memory_recording,
)

__all__ = [
    "test_experiment_metrics_to_dict_isolated_memory",
    "test_orchestrator_isolated_memory_recording",
]
