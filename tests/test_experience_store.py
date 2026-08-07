"""Unit tests for SubtaskExperienceStore and experience reuse functionality."""

import os
from pathlib import Path
import pytest

from src.memory.experience_store import (
    SubtaskExperienceStore,
    compute_signature,
    _distance_bucket,
)
from src.metrics.evaluation import ExperimentMetrics
from src.coordination.orchestrator import CONFIGS, DACAOrchestrator


def test_distance_bucketing():
    """Verify distance bucketing logic."""
    assert _distance_bucket(0.0) == 0
    assert _distance_bucket(15.0) == 0
    assert _distance_bucket(25.0) == 1
    assert _distance_bucket(195.0) == 9
    assert _distance_bucket(300.0) == 10


def test_signature_invariance_and_sensitivity():
    """Verify that compute_signature is invariant to agent IDs/coords order, but sensitive to skills/scenario."""
    sig1 = compute_signature("logistics", ["transport", "navigate"], ["uav", "robot"], 25.0)
    sig2 = compute_signature("logistics", ["navigate", "transport"], ["robot", "uav"], 28.0)
    # Same scenario, same skills (different order), same agent types (different order), same dist bucket -> SAME signature
    assert sig1 == sig2

    # Different scenario -> different signature
    sig3 = compute_signature("inspection", ["transport", "navigate"], ["uav", "robot"], 25.0)
    assert sig1 != sig3

    # Different distance bucket -> different signature
    sig4 = compute_signature("logistics", ["transport", "navigate"], ["uav", "robot"], 150.0)
    assert sig1 != sig4


def test_store_record_lookup_and_persistence(tmp_path):
    """Verify record, lookup, and JSON persistence of SubtaskExperienceStore."""
    store_file = tmp_path / "test_experience_store.json"
    store = SubtaskExperienceStore(store_path=str(store_file), enabled=True)

    sig = compute_signature("logistics", ["transport"], ["uav"], 10.0)
    assert store.lookup(sig) is None

    # Record a failure — lookup should still return None
    store.record(sig, {"T_0": ["uav_0"]}, success=False)
    assert store.lookup(sig) is None

    # Record a success — lookup should return the plan
    store.record(sig, {"T_0": ["uav_0"]}, success=True)
    plan = store.lookup(sig)
    assert plan == {"T_0": ["uav_0"]}

    # Verify persistence by instantiating a new store pointing to the same file
    store2 = SubtaskExperienceStore(store_path=str(store_file), enabled=True)
    assert store2.lookup(sig) == {"T_0": ["uav_0"]}


def test_store_disabled_by_default():
    """Verify that disabled store performs no-ops and returns None for lookup."""
    store = SubtaskExperienceStore(enabled=False)
    sig = compute_signature("logistics", ["transport"], ["uav"], 10.0)
    store.record(sig, {"T_0": ["uav_0"]}, success=True)
    assert store.lookup(sig) is None
    assert store.reuse_attempts == 0
    assert store.reuse_hits == 0


def test_experiment_metrics_dict_export_experience_reuse():
    """Verify that experience_reuse fields serialize cleanly in ExperimentMetrics."""
    m = ExperimentMetrics(
        config_name="A5",
        scenario="logistics",
        network_profile="stable",
        seed=42,
        success_rate=1.0,
        steps=10,
        cloud_tokens=100,
        device_tokens=50,
        total_tokens=150,
        cloud_api_calls=1,
        device_api_calls=1,
        total_api_calls=2,
        device_memory_mb=512.0,
        computation_s=0.5,
        total_wall_clock_s=1.0,
        experience_reuse_attempts=5,
        experience_reuse_hits=3,
    )
    d = m.to_dict()
    assert "experience_reuse_attempts" in d
    assert d["experience_reuse_attempts"] == 5
    assert "experience_reuse_hits" in d
    assert d["experience_reuse_hits"] == 3
