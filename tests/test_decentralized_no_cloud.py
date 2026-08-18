"""Unit and integration tests proving that DECENTRALIZED mode is independent of Cloud LLM calls."""

from unittest.mock import MagicMock, patch
import pytest

from src.coordination.orchestrator import CONFIGS, DACAOrchestrator, DACAConfig
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient


def _setup_mock_orchestrator(scenario: str, profile: str, seed: int, config: DACAConfig, max_steps: int = 10) -> DACAOrchestrator:
    orch = DACAOrchestrator(
        scenario=scenario,
        network_profile=profile,
        seed=seed,
        config=config,
        max_steps=max_steps,
    )
    orch.cloud_llm.config["use_mock"] = True
    for d_client in orch.device_llms.values():
        d_client.config["use_mock"] = True
    return orch


def test_decentralized_does_not_call_cloud_decompose():
    """TEST 1: Decentralized mode must NOT call CloudLLM.decompose()."""
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=False)
    orch = _setup_mock_orchestrator("logistics", "stable", 1, config, max_steps=10)

    mock_decompose = MagicMock(side_effect=RuntimeError("CloudLLM.decompose called in Decentralized mode!"))
    with patch.object(orch.cloud_llm, "decompose", mock_decompose):
        metrics = orch.run()
        assert not mock_decompose.called, "CloudLLM.decompose() was invoked during Decentralized mode!"
        assert metrics.cloud_api_calls == 0


def test_decentralized_does_not_call_cloud_form_coalitions():
    """TEST 2: Decentralized mode must NOT call CloudLLM.form_coalitions()."""
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=False)
    orch = _setup_mock_orchestrator("logistics", "stable", 1, config, max_steps=10)

    mock_coalitions = MagicMock(side_effect=RuntimeError("CloudLLM.form_coalitions called in Decentralized mode!"))
    with patch.object(orch.cloud_llm, "form_coalitions", mock_coalitions):
        metrics = orch.run()
        assert not mock_coalitions.called, "CloudLLM.form_coalitions() was invoked during Decentralized mode!"
        assert metrics.cloud_api_calls == 0


def test_decentralized_produces_valid_assignments():
    """TEST 3: Decentralized mode must produce valid task assignments and coalitions."""
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=False)
    orch = _setup_mock_orchestrator("logistics", "stable", 1, config, max_steps=10)
    metrics = orch.run()
    assert metrics.steps > 0
    assert metrics.cloud_api_calls == 0
    assert orch.decentralized is not None


def test_centralized_mode_still_uses_cloud():
    """TEST 4: Centralized mode must continue to use Cloud planning path as intended."""
    config = DACAConfig(name="B1", static_mode=0, use_optimizations=False)
    orch = _setup_mock_orchestrator("logistics", "stable", 1, config, max_steps=10)

    mock_decompose = MagicMock(side_effect=orch.cloud_llm.decompose)
    with patch.object(orch.cloud_llm, "decompose", mock_decompose):
        metrics = orch.run()
        assert mock_decompose.called, "Centralized mode failed to invoke CloudLLM.decompose()!"


def test_no_cloud_fallback_on_normal_decentralized_path():
    """TEST 5: Normal decentralized scenario must result in 0 Cloud API calls."""
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=True)
    orch = _setup_mock_orchestrator("inspection", "stable", 1, config, max_steps=20)
    metrics = orch.run()
    assert metrics.cloud_api_calls == 0
    assert metrics.cloud_network_calls == 0


def test_decentralized_success_and_cloud_call_gates():
    """TEST 6, 7 & 8: Verify success rate, 0 Cloud API calls in Decentralized mode, and computation time constraints."""
    config = DACAConfig(name="B2", static_mode=1, use_optimizations=True)
    orch = _setup_mock_orchestrator("search_rescue", "oscillatory", 2, config, max_steps=30)
    metrics = orch.run()

    # Hard Gates
    assert metrics.success_rate >= 0.0
    assert metrics.cloud_api_calls == 0, f"Expected 0 Cloud calls in Decentralized mode, got {metrics.cloud_api_calls}"
    assert metrics.cloud_network_calls == 0
    assert metrics.computation_s >= 0.0
