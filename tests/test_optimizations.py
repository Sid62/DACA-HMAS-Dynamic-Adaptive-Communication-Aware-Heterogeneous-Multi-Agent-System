"""Unit and integration tests for Cloud LLM Optimizations 1-8."""

import pytest
import numpy as np
from src.llm.state_summarizer import CompactStateSummarizer
from src.llm.semantic_cache import SemanticPlanCache
from src.llm.cloud_llm_client import CloudLLMClient
from src.communication.peer_manager import PeerCommunicationManager
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector


def test_compact_state_summarizer():
    summarizer = CompactStateSummarizer(enabled=True)
    agents = [
        {"id": "uav_1", "type": "uav", "skills": ["recon"], "pos": [10.0, 20.0, 5.0]},
        {"id": "uav_2", "type": "uav", "skills": ["transport"], "pos": [15.0, 25.0, 5.0]},
    ]
    subtasks = [
        {"id": "T_0", "skills": ["recon"], "target": [100.0, 200.0]},
    ]
    res = summarizer.summarize_decomposition_context(
        instruction="Perform reconnaissance on zone A",
        agents=agents,
        subtasks=subtasks,
        dist_mat=[[0.0, 5.0], [5.0, 0.0]],
    )
    assert res.original_chars > res.summarized_chars
    assert res.prompt_reduction_percent > 0.0
    assert "uav_1" in res.summary_text


def test_semantic_plan_cache():
    cache = SemanticPlanCache(enabled=True, similarity_threshold=0.90, max_cache_age=15)
    state = {
        "active_agents": [{"id": "uav_1"}],
        "subtasks": [{"id": "T_0", "target": [10.0, 10.0]}],
        "cqi": 0.85,
    }
    plan = {"assignments": {"T_0": ["uav_1"]}}

    # Initial lookup is miss
    res1 = cache.lookup(state, current_step=1)
    assert res1 is None
    assert cache.cache_misses == 1

    # Put plan into cache
    cache.put(state, plan, tokens=200, latency=0.05, current_step=1)

    # Subsequent lookup is hit
    res2 = cache.lookup(state, current_step=2)
    assert res2 == plan
    assert cache.cache_hits == 1
    assert cache.saved_cloud_calls == 1
    assert cache.saved_tokens == 200


def test_cloud_llm_optimizations_integration():
    client = CloudLLMClient(config={"use_mock": True, "cache_responses": False, "cloud": {"provider": "openai"}})
    assert client.summarizer is not None
    assert client.semantic_cache is not None

    agents = [{"id": "uav_1", "type": "uav", "skills": ["recon"]}]
    subtasks = [{"id": "T_0", "skills": ["recon"], "target": [10.0, 10.0]}]

    # First call primes cache
    res1 = client.decompose("Execute mission", agents, subtasks)
    assert "T_0" in res1

    # Second identical call hits semantic cache
    res2 = client.decompose("Execute mission", agents, subtasks)
    assert res2 == res1
    assert client.semantic_cache.cache_hits >= 1


def test_consensus_skipped_metric():
    pm = PeerCommunicationManager()
    assert pm.consensus_skipped == 0
    pm.record_consensus_skipped()
    assert pm.consensus_skipped == 1
    snap = pm.metrics_snapshot()
    assert snap["consensus_skipped"] == 1


def test_experiment_metrics_export():
    collector = MetricsCollector()
    metrics = collector.finalize(
        success_rate=1.0,
        steps=10,
        cloud_tokens=1000,
        cloud_api_calls=5,
        device_tokens=500,
        device_api_calls=2,
        device_memory_mb=128.0,
        computation_s=0.5,
        total_wall_clock_s=1.0,
        tfr_history=[1.0],
        cfr_history=[1.0],
        switch_count=0,
        config_name="hybrid",
        scenario="logistics",
        network_profile="ideal",
        seed=42,
        prompt_reduction_percent=45.5,
        cache_misses=2,
        cache_hit_rate=0.6,
        saved_cloud_calls=3,
        saved_tokens=600,
        saved_latency=0.15,
        consensus_skipped=4,
    )
    d = metrics.to_dict()
    assert d["prompt_reduction_percent"] == 45.5
    assert d["saved_cloud_calls"] == 3
    assert d["saved_tokens"] == 600
    assert d["consensus_skipped"] == 4
