"""Tests for Issue 5: Reliable experiment settings, unified RunConfig, and communication/token counters."""

from __future__ import annotations

import copy
import pytest
from typing import Any

from src.coordination.orchestrator import CONFIGS, DACAConfig, DACAOrchestrator, RunConfig
from src.metrics.communication_counter import CommunicationStepCounter, STANDARD_CATEGORIES
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector
from src.llm.cloud_llm_client import CloudLLMClient, LLMUsage
from src.llm.device_llm_client import DeviceLLMClient, DeviceLLMUsage


def test_a5_unopt_disables_all_optimizations_and_global_defaults_cannot_reenable():
    """1. A5_unopt disables every configured optimization, and global defaults cannot re-enable any of them."""
    # Test unified RunConfig creation when global yaml/thresholds have optimizations enabled
    permissive_llm_cfg = {
        "use_mock": True,
        "cache_responses": True,
        "experience_reuse": {"enabled": True, "store_path": "test_store.json"},
        "cloud": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        "device": {"provider": "vllm", "model": "Qwen/Qwen2.5-3B-Instruct"},
    }
    permissive_thresholds = {
        "optimizations": {
            "semantic_cache": {"enabled": True},
            "prompt_compression": {"enabled": True},
            "summarization": {"enabled": True},
            "constrained_output": {"enabled": True},
            "consensus": {"skip_when_stable": True},
        }
    }

    cfg_unopt = DACAConfig(name="A5_unopt", use_optimizations=False)
    run_cfg = cfg_unopt.to_run_config(
        llm_cfg=permissive_llm_cfg,
        thresholds=permissive_thresholds,
        scenario="logistics",
        network_profile="stable",
        seed=42,
    )

    # Verify run_config flags are strictly False despite permissive global defaults
    assert run_cfg.use_optimizations is False
    assert run_cfg.semantic_cache is False
    assert run_cfg.prompt_compression is False
    assert run_cfg.summarization is False
    assert run_cfg.constrained_output is False
    assert run_cfg.experience_reuse is False
    assert run_cfg.plan_continuity is False
    assert run_cfg.delta_transfer is False
    assert run_cfg.consensus_skip is False
    assert run_cfg.cache_responses is False

    # Instantiate orchestrator with A5_unopt
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5_unopt"],
        thresholds=permissive_thresholds,
        max_steps=5,
    )

    # Check that orchestrator component state honors disabled optimizations
    assert orch.run_config.use_optimizations is False
    if hasattr(orch.cloud_llm, "summarizer") and orch.cloud_llm.summarizer:
        assert orch.cloud_llm.summarizer.enabled is False
    assert orch.cloud_llm.opt_prompt_compression is False
    assert orch.cloud_llm.opt_constrained_output is False
    if hasattr(orch.cloud_llm, "semantic_cache") and orch.cloud_llm.semantic_cache:
        assert orch.cloud_llm.semantic_cache.enabled is False
    assert orch.cloud_llm.config.get("cache_responses") is False
    for dc in orch.device_llms.values():
        assert dc.config.get("cache_responses") is False
    assert orch.experience_store.enabled is False
    assert orch.continuity_engine is None
    assert orch.delta_transfer_manager is None
    assert orch.centralized.continuity_engine is None
    assert orch.centralized.experience_store.enabled is False
    assert orch.decentralized.continuity_engine is None
    assert orch.decentralized.experience_store.enabled is False


def test_a5_enables_intended_optimizations():
    """2. A5 enables intended optimizations."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=5,
    )

    assert orch.run_config.use_optimizations is True
    assert orch.run_config.semantic_cache is True
    assert orch.run_config.prompt_compression is True
    assert orch.run_config.summarization is True
    assert orch.run_config.constrained_output is True
    assert orch.run_config.experience_reuse is True
    assert orch.run_config.plan_continuity is True
    assert orch.run_config.delta_transfer is True
    assert orch.run_config.consensus_skip is True
    assert orch.run_config.cache_responses is True

    # Component instances should be active
    assert orch.continuity_engine is not None
    assert orch.delta_transfer_manager is not None
    assert orch.experience_store.enabled is True
    if hasattr(orch.cloud_llm, "semantic_cache") and orch.cloud_llm.semantic_cache:
        assert orch.cloud_llm.semantic_cache.enabled is True


def test_relevant_components_share_same_run_config():
    """3. Relevant components share the same RunConfig instance."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=42,
        config=CONFIGS["A5"],
        max_steps=5,
    )
    rc = orch.run_config

    assert orch.cloud_llm.run_config is rc
    for dc in orch.device_llms.values():
        assert dc.run_config is rc
    assert orch.comm_counter.run_config is rc
    assert orch.centralized.run_config is rc
    assert orch.decentralized.run_config is rc
    assert orch.experience_store.run_config is rc
    assert orch.decomposer.run_config is rc
    assert orch.coalition_formation.run_config is rc
    assert orch.device_decomposer.run_config is rc
    assert orch.device_coalition_formation.run_config is rc
    assert orch.reallocator.run_config is rc


def test_experiment_metadata_stored_correctly():
    """4. Experiment metadata (seed, provider, model, mock/real, optimizations, cache-start) is stored in ExperimentMetrics."""
    orch = DACAOrchestrator(
        scenario="logistics",
        network_profile="stable",
        seed=101,
        config=CONFIGS["A5"],
        max_steps=10,
    )
    orch.cloud_llm.config["use_mock"] = True
    for dc in orch.device_llms.values():
        dc.config["use_mock"] = True

    metrics = orch.run()
    d = metrics.to_dict()

    assert "metadata" in d
    meta = d["metadata"]
    assert meta["seed"] == 101
    assert "provider" in meta
    assert "model" in meta
    assert meta["mock"] is True or meta["use_mock"] is True
    assert "optimizations" in meta
    assert isinstance(meta["optimizations"], dict)
    assert meta["optimizations"]["use_optimizations"] is True
    assert "cache_start_state" in meta
    css = meta["cache_start_state"]
    assert "semantic_cache_entries" in css
    assert "disk_cache_enabled" in css
    assert "experience_store_entries" in css


def test_paper_communication_steps_determinism():
    """5. paper_communication_steps is deterministic across runs with identical parameters."""
    def run_sim():
        orch = DACAOrchestrator(
            scenario="logistics",
            network_profile="stable",
            seed=42,
            config=CONFIGS["A5"],
            max_steps=20,
        )
        orch.cloud_llm.config["use_mock"] = True
        for dc in orch.device_llms.values():
            dc.config["use_mock"] = True
        return orch.run()

    res1 = run_sim()
    res2 = run_sim()

    assert res1.paper_communication_steps == res2.paper_communication_steps
    assert res1.communication_steps == res2.communication_steps
    assert res1.communication_step_breakdown == res2.communication_step_breakdown


def test_planning_and_dispatch_not_double_counted():
    """6. Planning and dispatch are counted cleanly and not double-counted in paper_communication_steps."""
    counter = CommunicationStepCounter()
    assert counter.value == 0
    assert counter.paper_value == 0

    # Record 1 global planning step
    counter.record_global_planning(1, "centralized_global_planning")
    assert counter.value == 1
    assert counter.paper_value == 1
    assert counter.breakdown["global_planning"] == 1
    assert counter.breakdown.get("dispatch", 0) == 0

    # Record 1 dispatch step
    counter.record_dispatch(1, "centralized_domain_dispatch")
    assert counter.value == 2
    assert counter.paper_value == 2
    assert counter.breakdown["dispatch"] == 1

    # Exactly matches paper definition: paper = global_planning + dispatch
    assert counter.paper_value == counter.breakdown["global_planning"] + counter.breakdown["dispatch"]


def test_other_communication_categories_remain_separate():
    """7. Local, peer, feedback, and handoff categories remain separate and do NOT leak into paper metric."""
    counter = CommunicationStepCounter()

    counter.record_local_coordination(5, "local_coordination_event")
    counter.record_peer_consensus(10, "consensus_round")
    counter.record_feedback_sync(4, "state_sync")
    counter.record_handoff_reallocation(2, "handoff_snapshot")

    # None of these must enter paper_communication_steps
    assert counter.paper_value == 0
    # Total communication steps counts them all
    assert counter.value == 21

    # Breakdown has distinct keys
    for cat in ["local_coordination", "peer_consensus", "feedback_sync", "handoff_reallocation"]:
        assert cat in counter.breakdown
        assert counter.breakdown[cat] > 0

    # Adding global planning increments paper_value by only the planning step count
    counter.record_global_planning(1)
    assert counter.paper_value == 1
    assert counter.value == 22

    # Verify standard categories and unknown category rejection
    for standard in STANDARD_CATEGORIES:
        counter.increment(standard, 1)
    with pytest.raises(ValueError):
        counter.increment("non_existent_category", 1)


def test_measured_vs_estimated_token_usage_distinguishable():
    """8. Measured provider-reported tokens and estimated mock/fallback tokens are strictly separated."""
    cloud_usage = LLMUsage()
    assert cloud_usage.measured_total_tokens == 0
    assert cloud_usage.estimated_total_tokens == 0
    assert cloud_usage.total_tokens == 0

    # Record mock/estimated tokens
    cloud_usage.record_estimated_tokens(prompt_tokens=150, completion_tokens=50)
    assert cloud_usage.measured_total_tokens == 0
    assert cloud_usage.estimated_prompt_tokens == 150
    assert cloud_usage.estimated_completion_tokens == 50
    assert cloud_usage.estimated_total_tokens == 200
    # Backward compatible aggregate
    assert cloud_usage.prompt_tokens == 150
    assert cloud_usage.completion_tokens == 50
    assert cloud_usage.total_tokens == 200

    # Record measured provider tokens
    cloud_usage.record_measured_tokens(prompt_tokens=300, completion_tokens=100)
    assert cloud_usage.measured_prompt_tokens == 300
    assert cloud_usage.measured_completion_tokens == 100
    assert cloud_usage.measured_total_tokens == 400
    assert cloud_usage.estimated_total_tokens == 200
    # Backward compatible aggregates sum both
    assert cloud_usage.prompt_tokens == 450
    assert cloud_usage.completion_tokens == 150
    assert cloud_usage.total_tokens == 600

    # Verify device usage also maintains separation
    dev_usage = DeviceLLMUsage()
    dev_usage.record_estimated_tokens(prompt_tokens=80, completion_tokens=20)
    assert dev_usage.measured_total_tokens == 0
    assert dev_usage.estimated_total_tokens == 100
    assert dev_usage.total_tokens == 100

    dev_usage.record_measured_tokens(prompt_tokens=120, completion_tokens=30)
    assert dev_usage.measured_total_tokens == 150
    assert dev_usage.estimated_total_tokens == 100
    assert dev_usage.total_tokens == 250

    # Verify ExperimentMetrics serialization separates measured vs estimated
    metrics = ExperimentMetrics(
        config_name="A5",
        scenario="logistics",
        network_profile="stable",
        seed=1,
        success_rate=1.0,
        steps=10,
        cloud_tokens=600,
        device_tokens=250,
        total_tokens=850,
        cloud_api_calls=1,
        device_api_calls=1,
        total_api_calls=2,
        device_memory_mb=100.0,
        computation_s=1.0,
        measured_cloud_prompt_tokens=300,
        measured_cloud_completion_tokens=100,
        measured_cloud_total_tokens=400,
        estimated_cloud_prompt_tokens=150,
        estimated_cloud_completion_tokens=50,
        estimated_cloud_total_tokens=200,
        measured_device_prompt_tokens=120,
        measured_device_completion_tokens=30,
        measured_device_total_tokens=150,
        estimated_device_prompt_tokens=80,
        estimated_device_completion_tokens=20,
        estimated_device_total_tokens=100,
    )
    d = metrics.to_dict()
    assert d["measured_cloud_total_tokens"] == 400
    assert d["estimated_cloud_total_tokens"] == 200
    assert d["measured_device_total_tokens"] == 150
    assert d["estimated_device_total_tokens"] == 100
    assert d["tokens"] == 850


def test_deterministic_a5_vs_a5_unopt_comparison():
    """Comparison: A5 vs A5_unopt under identical seed, tasks, network conditions, model settings, and cache-start state."""
    seed = 77
    scenario = "logistics"
    network_profile = "stable"
    max_steps = 20

    orch_a5 = DACAOrchestrator(
        scenario=scenario,
        network_profile=network_profile,
        seed=seed,
        config=CONFIGS["A5"],
        max_steps=max_steps,
    )
    orch_a5.cloud_llm.config["use_mock"] = True
    for dc in orch_a5.device_llms.values():
        dc.config["use_mock"] = True

    orch_unopt = DACAOrchestrator(
        scenario=scenario,
        network_profile=network_profile,
        seed=seed,
        config=CONFIGS["A5_unopt"],
        max_steps=max_steps,
    )
    orch_unopt.cloud_llm.config["use_mock"] = True
    for dc in orch_unopt.device_llms.values():
        dc.config["use_mock"] = True

    metrics_a5 = orch_a5.run()
    metrics_unopt = orch_unopt.run()

    # Verify run configurations diverged as intended
    assert orch_a5.run_config.use_optimizations is True
    assert orch_unopt.run_config.use_optimizations is False

    # Check metadata in both
    meta_a5 = metrics_a5.to_dict()["metadata"]
    meta_unopt = metrics_unopt.to_dict()["metadata"]

    assert meta_a5["seed"] == meta_unopt["seed"] == seed
    assert meta_a5["optimizations"]["use_optimizations"] is True
    assert meta_unopt["optimizations"]["use_optimizations"] is False
    assert meta_unopt["optimizations"]["semantic_cache"] is False
    assert meta_unopt["optimizations"]["experience_reuse"] is False
    assert meta_unopt["optimizations"]["plan_continuity"] is False

    # In unoptimized run, experience reuse and cache cannot be hit
    assert metrics_unopt.experience_reuse_hits == 0
    assert metrics_unopt.semantic_cache_hits == 0

    # Both runs produce valid, deterministic paper_communication_steps
    assert metrics_a5.paper_communication_steps >= 0
    assert metrics_unopt.paper_communication_steps >= 0
    assert (
        metrics_a5.paper_communication_steps
        == metrics_a5.communication_step_breakdown.get("global_planning", 0)
        + metrics_a5.communication_step_breakdown.get("dispatch", 0)
    )
    assert (
        metrics_unopt.paper_communication_steps
        == metrics_unopt.communication_step_breakdown.get("global_planning", 0)
        + metrics_unopt.communication_step_breakdown.get("dispatch", 0)
    )
