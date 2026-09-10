"""Experiment metrics collection and statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExperimentMetrics:
    config_name: str
    scenario: str
    network_profile: str
    seed: int
    success_rate: float
    steps: int
    cloud_tokens: int
    device_tokens: int
    total_tokens: int
    cloud_api_calls: int
    device_api_calls: int
    total_api_calls: int
    device_memory_mb: float
    computation_s: float
    total_wall_clock_s: float = 0.0
    tfr: float = 1.0
    cfr: float = 1.0
    switch_count: int = 0
    peer_messages: int = 0
    broadcast_count: int = 0
    consensus_rounds: int = 0
    consensus_latency: float = 0.0
    plan_merge_count: int = 0
    distributed_replanning_count: int = 0
    replanning_count: int = 0
    local_reallocation_count: int = 0
    reallocation_trigger_count: int = 0
    reallocation_skip_count: int = 0
    reallocation_reasons: dict[str, int] = field(default_factory=dict)

    # Upgraded token accounting
    cloud_prompt_tokens: int = 0
    cloud_completion_tokens: int = 0
    cloud_total_tokens: int = 0
    device_prompt_tokens: int = 0
    device_completion_tokens: int = 0
    device_total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cloud_retry_tokens: int = 0
    device_retry_tokens: int = 0

    # Upgraded API call instrumentation
    successful_calls: int = 0
    failed_calls: int = 0
    retried_calls: int = 0
    cache_hits: int = 0
    local_non_llm_operations: int = 0
    cloud_network_calls: int = 0
    cloud_disk_cache_hits: int = 0
    cloud_failed_attempts: int = 0
    semantic_cache_hits: int = 0
    cloud_call_attribution: dict = field(default_factory=dict)
    logical_llm_requests: int = 0
    device_inference_calls: int = 0

    # Upgraded memory instrumentation
    process_peak_rss_mb: float = 0.0
    process_mean_rss_mb: float = 0.0
    gpu_peak_memory_mb: float = 0.0
    gpu_mean_memory_mb: float = 0.0
    device_llm_python_heap_delta_mb: float = 0.0
    device_llm_python_heap_delta_by_device: dict[str, float] = field(default_factory=dict)
    device_llm_tokens_processed_by_device: dict[str, int] = field(default_factory=dict)
    device_llm_memory_mb: dict[str, float] = field(default_factory=dict)
    device_llm_memory_peak_mb: dict[str, float] = field(default_factory=dict)
    device_llm_heap_delta_mb: dict[str, float] = field(default_factory=dict)
    inference_backend_memory_mb: float | None = None

    # Upgraded computation time decomposition (seconds)
    cloud_inference_time_s: float = 0.0
    device_inference_time_s: float = 0.0
    cqi_evaluation_time_s: float = 0.0
    coalition_computation_time_s: float = 0.0
    architecture_switching_time_s: float = 0.0
    snapshot_capture_time_s: float = 0.0
    state_restore_time_s: float = 0.0
    state_verification_time_s: float = 0.0
    coalition_transfer_time_s: float = 0.0
    reallocation_time_s: float = 0.0
    state_handoff_time_s: float = 0.0
    coalition_repair_time_s: float = 0.0
    consensus_time_s: float = 0.0
    planning_time_s: float = 0.0
    network_waiting_time_s: float = 0.0
    simulation_computation_time_s: float = 0.0

    # Upgraded planning latency distribution
    avg_planning_latency: float = 0.0
    planning_latency_p50: float = 0.0
    planning_latency_p95: float = 0.0
    planning_latency_p99: float = 0.0
    planning_latency_min: float = 0.0
    planning_latency_max: float = 0.0
    planning_latency_std: float = 0.0

    # Upgraded communication accounting
    cloud_to_device_messages: int = 0
    device_to_cloud_messages: int = 0
    handoff_messages: int = 0
    coalition_messages: int = 0
    repair_messages: int = 0
    cloud_bytes: int = 0
    peer_bytes: int = 0
    broadcast_bytes: int = 0
    total_bytes: int = 0

    coalition_change_count: int = 0
    cached_plan_reuse_count: int = 0
    merged_singleton_count: int = 0
    communication_steps: int = 0
    paper_communication_steps: int = 0
    dispatch_skipped_rounds: int = 0
    communication_step_breakdown: dict[str, int] = field(default_factory=dict)
    hallucination_stats: dict[str, Any] = field(default_factory=dict)
    experience_reuse_attempts: int = 0
    experience_reuse_hits: int = 0

    # Modular Optimizations Metrics (Optimizations 1-8)
    prompt_reduction_percent: float = 0.0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    saved_cloud_calls: int = 0
    saved_tokens: int = 0
    saved_latency: float = 0.0
    local_reasoning_count: int = 0
    cloud_reasoning_count: int = 0
    confidence_distribution: dict[str, float] = field(default_factory=dict)
    consensus_skipped: int = 0
    consensus_duration: float = 0.0
    planner_latency: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        sem_hits = self.semantic_cache_hits
        tot_cache = sem_hits + self.cache_misses
        hit_rate = round(sem_hits / tot_cache, 4) if tot_cache > 0 else self.cache_hit_rate
        return {
            "config": self.config_name,
            "scenario": self.scenario,
            "profile": self.network_profile,
            "seed": self.seed,
            "success_rate": round(self.success_rate * 100, 2),
            "steps": self.steps,
            "tokens": self.total_tokens,
            "cloud_prompt_tokens": self.cloud_prompt_tokens,
            "cloud_completion_tokens": self.cloud_completion_tokens,
            "cloud_total_tokens": self.cloud_total_tokens,
            "device_prompt_tokens": self.device_prompt_tokens,
            "device_completion_tokens": self.device_completion_tokens,
            "device_total_tokens": self.device_total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "cloud_retry_tokens": self.cloud_retry_tokens,
            "device_retry_tokens": self.device_retry_tokens,
            "api_calls": self.total_api_calls,
            "cloud_planning_calls": self.cloud_api_calls,
            "device_planning_calls": self.device_api_calls,
            "logical_llm_requests": self.logical_llm_requests,
            "logical_requests": self.logical_llm_requests,
            "device_inference_calls": self.device_inference_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "cache_hits": self.cache_hits,
            "local_non_llm_operations": self.local_non_llm_operations,
            "cloud_network_calls": self.cloud_network_calls,
            "cloud_disk_cache_hits": self.cloud_disk_cache_hits,
            "cloud_failed_attempts": self.cloud_failed_attempts,
            "semantic_cache_hits": self.semantic_cache_hits,
            "disk_cache_hits": self.cache_hits,
            "cloud_call_attribution": dict(self.cloud_call_attribution),
            "memory_mb": round(self.device_memory_mb, 1),
            "device_llm_memory_mb": {k: round(v, 4) for k, v in self.device_llm_memory_mb.items()},
            "device_llm_memory_peak_mb": {k: round(v, 4) for k, v in self.device_llm_memory_peak_mb.items()},
            # device_llm_memory_mb_max is a reconstructed operational definition summarizing peak memory across device domains
            "device_llm_memory_mb_max": round(max(self.device_llm_memory_peak_mb.values(), default=0.0), 4),
            "device_llm_heap_delta_mb": {k: round(v, 4) for k, v in self.device_llm_heap_delta_mb.items()},
            "inference_backend_memory_mb": self.inference_backend_memory_mb,
            "device_llm_python_heap_delta_mb": round(self.device_llm_python_heap_delta_mb, 4),
            "device_llm_python_heap_delta_by_device": dict(self.device_llm_python_heap_delta_by_device),
            "device_llm_tokens_processed_by_device": dict(self.device_llm_tokens_processed_by_device),
            "process_peak_rss_mb": round(self.process_peak_rss_mb, 1),
            "process_mean_rss_mb": round(self.process_mean_rss_mb, 1),
            "gpu_peak_memory_mb": round(self.gpu_peak_memory_mb, 1),
            "gpu_mean_memory_mb": round(self.gpu_mean_memory_mb, 1),
            "computation_s": round(self.computation_s, 3),
            "cloud_inference_time_s": round(self.cloud_inference_time_s, 4),
            "device_inference_time_s": round(self.device_inference_time_s, 4),
            "cqi_evaluation_time_s": round(self.cqi_evaluation_time_s, 4),
            "coalition_computation_time_s": round(self.coalition_computation_time_s, 4),
            "architecture_switching_time_s": round(self.architecture_switching_time_s, 4),
            "snapshot_capture_time_s": round(self.snapshot_capture_time_s, 4),
            "state_restore_time_s": round(self.state_restore_time_s, 4),
            "state_verification_time_s": round(self.state_verification_time_s, 4),
            "coalition_transfer_time_s": round(self.coalition_transfer_time_s, 4),
            "reallocation_time_s": round(self.reallocation_time_s, 4),
            "state_handoff_time_s": round(self.state_handoff_time_s, 4),
            "coalition_repair_time_s": round(self.coalition_repair_time_s, 4),
            "consensus_time_s": round(self.consensus_time_s, 4),
            "planning_time_s": round(self.planning_time_s, 4),
            "network_waiting_time_s": round(self.network_waiting_time_s, 4),
            "simulation_computation_time_s": round(self.simulation_computation_time_s, 4),
            "total_wall_clock_s": round(self.total_wall_clock_s, 3),
            "tfr": round(self.tfr, 4),
            "cfr": round(self.cfr, 4),
            "switch_count": self.switch_count,
            "peer_messages": self.peer_messages,
            "broadcast_count": self.broadcast_count,
            "consensus_rounds": self.consensus_rounds,
            "consensus_latency": round(self.consensus_latency, 4),
            "plan_merge_count": self.plan_merge_count,
            "distributed_replanning_count": self.distributed_replanning_count,
            "replanning_count": self.replanning_count,
            "local_reallocation_count": self.local_reallocation_count,
            "reallocation_trigger_count": self.reallocation_trigger_count,
            "reallocation_skip_count": self.reallocation_skip_count,
            "reallocation_reasons": dict(self.reallocation_reasons),
            "cached_plan_reuse_count": self.cached_plan_reuse_count,
            "merged_singleton_count": self.merged_singleton_count,
            "avg_planning_latency": round(self.avg_planning_latency, 4),
            "planning_latency_p50": round(self.planning_latency_p50, 4),
            "planning_latency_p95": round(self.planning_latency_p95, 4),
            "planning_latency_p99": round(self.planning_latency_p99, 4),
            "planning_latency_min": round(self.planning_latency_min, 4),
            "planning_latency_max": round(self.planning_latency_max, 4),
            "planning_latency_std": round(self.planning_latency_std, 4),
            "cloud_to_device_messages": self.cloud_to_device_messages,
            "device_to_cloud_messages": self.device_to_cloud_messages,
            "handoff_messages": self.handoff_messages,
            "coalition_messages": self.coalition_messages,
            "repair_messages": self.repair_messages,
            "cloud_bytes": self.cloud_bytes,
            "peer_bytes": self.peer_bytes,
            "broadcast_bytes": self.broadcast_bytes,
            "total_bytes": self.total_bytes,
            "coalition_change_count": self.coalition_change_count,
            "communication_steps": self.communication_steps,
            "paper_communication_steps": self.paper_communication_steps,
            "dispatch_skipped_rounds": self.dispatch_skipped_rounds,
            "communication_step_breakdown": dict(self.communication_step_breakdown),
            "hallucination_stats": dict(self.hallucination_stats),
            "experience_reuse_attempts": self.experience_reuse_attempts,
            "experience_reuse_hits": self.experience_reuse_hits,
            # Optimizations Metrics
            "prompt_reduction_percent": round(self.prompt_reduction_percent, 2),
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "saved_cloud_calls": self.saved_cloud_calls,
            "saved_tokens": self.saved_tokens,
            "saved_latency": round(self.saved_latency, 4),
            "local_reasoning_count": self.local_reasoning_count,
            "cloud_reasoning_count": self.cloud_reasoning_count,
            "confidence_distribution": dict(self.confidence_distribution),
            "consensus_skipped": self.consensus_skipped,
            "consensus_duration": round(self.consensus_duration, 4),
            "planner_latency": round(self.planner_latency, 4),
        }


@dataclass
class MetricsCollector:
    records: list[ExperimentMetrics] = field(default_factory=list)

    def finalize(
        self,
        success_rate: float,
        steps: int,
        cloud_tokens: int,
        cloud_api_calls: int,
        device_tokens: int,
        device_api_calls: int,
        device_memory_mb: float,
        computation_s: float,
        total_wall_clock_s: float,
        tfr_history: list[float],
        cfr_history: list[float],
        switch_count: int,
        config_name: str,
        scenario: str,
        network_profile: str,
        seed: int,
        peer_messages: int = 0,
        broadcast_count: int = 0,
        consensus_rounds: int = 0,
        consensus_latency: float = 0.0,
        plan_merge_count: int = 0,
        distributed_replanning_count: int = 0,
        replanning_count: int = 0,
        local_reallocation_count: int = 0,
        reallocation_trigger_count: int = 0,
        reallocation_skip_count: int = 0,
        reallocation_reasons: dict[str, int] | None = None,
        cached_plan_reuse_count: int = 0,
        merged_singleton_count: int = 0,
        avg_planning_latency: float = 0.0,
        coalition_change_count: int = 0,
        communication_steps: int = 0,
        paper_communication_steps: int = 0,
        communication_step_breakdown: dict[str, int] | None = None,
        hallucination_stats: dict[str, Any] | None = None,
        experience_reuse_attempts: int = 0,
        experience_reuse_hits: int = 0,
        dispatch_skipped_rounds: int = 0,
        # Upgraded keyword arguments with safe defaults
        cloud_prompt_tokens: int = 0,
        cloud_completion_tokens: int = 0,
        cloud_total_tokens: int = 0,
        device_prompt_tokens: int = 0,
        device_completion_tokens: int = 0,
        device_total_tokens: int = 0,
        cloud_retry_tokens: int = 0,
        device_retry_tokens: int = 0,
        successful_calls: int = 0,
        failed_calls: int = 0,
        retried_calls: int = 0,
        cache_hits: int = 0,
        local_non_llm_operations: int = 0,
        cloud_network_calls: int = 0,
        cloud_disk_cache_hits: int = 0,
        cloud_failed_attempts: int = 0,
        semantic_cache_hits: int = 0,
        cloud_call_attribution: dict | None = None,
        logical_llm_requests: int = 0,
        device_inference_calls: int = 0,
        process_peak_rss_mb: float = 0.0,
        process_mean_rss_mb: float = 0.0,
        gpu_peak_memory_mb: float = 0.0,
        gpu_mean_memory_mb: float = 0.0,
        device_llm_memory_mb: dict[str, float] | None = None,
        device_llm_memory_peak_mb: dict[str, float] | None = None,
        device_llm_heap_delta_mb: dict[str, float] | None = None,
        inference_backend_memory_mb: float | None = None,
        device_llm_python_heap_delta_mb: float = 0.0,
        device_llm_python_heap_delta_by_device: dict[str, float] | None = None,
        device_llm_tokens_processed_by_device: dict[str, int] | None = None,
        cloud_inference_time_s: float = 0.0,
        device_inference_time_s: float = 0.0,
        cqi_evaluation_time_s: float = 0.0,
        coalition_computation_time_s: float = 0.0,
        architecture_switching_time_s: float = 0.0,
        snapshot_capture_time_s: float = 0.0,
        state_restore_time_s: float = 0.0,
        state_verification_time_s: float = 0.0,
        coalition_transfer_time_s: float = 0.0,
        reallocation_time_s: float = 0.0,
        state_handoff_time_s: float = 0.0,
        coalition_repair_time_s: float = 0.0,
        consensus_time_s: float = 0.0,
        planning_time_s: float = 0.0,
        network_waiting_time_s: float = 0.0,
        simulation_computation_time_s: float = 0.0,
        planning_latency_p50: float = 0.0,
        planning_latency_p95: float = 0.0,
        planning_latency_p99: float = 0.0,
        planning_latency_min: float = 0.0,
        planning_latency_max: float = 0.0,
        planning_latency_std: float = 0.0,
        cloud_to_device_messages: int = 0,
        device_to_cloud_messages: int = 0,
        handoff_messages: int = 0,
        coalition_messages: int = 0,
        repair_messages: int = 0,
        cloud_bytes: int = 0,
        peer_bytes: int = 0,
        broadcast_bytes: int = 0,
        total_bytes: int = 0,
        prompt_reduction_percent: float = 0.0,
        cache_misses: int = 0,
        cache_hit_rate: float = 0.0,
        saved_cloud_calls: int = 0,
        saved_tokens: int = 0,
        saved_latency: float = 0.0,
        local_reasoning_count: int = 0,
        cloud_reasoning_count: int = 0,
        confidence_distribution: dict[str, float] | None = None,
        consensus_skipped: int = 0,
        consensus_duration: float = 0.0,
        planner_latency: float = 0.0,
    ) -> ExperimentMetrics:
        breakdown_dict = dict(communication_step_breakdown or {})
        paper_comm_steps = (
            paper_communication_steps
            if paper_communication_steps > 0
            else (breakdown_dict.get("global_planning", 0) + breakdown_dict.get("dispatch", 0))
        )
        c_tot = cloud_total_tokens if cloud_total_tokens > 0 else cloud_tokens
        d_tot = device_total_tokens if device_total_tokens > 0 else device_tokens
        t_tot = c_tot + d_tot
        m = ExperimentMetrics(
            config_name=config_name,
            scenario=scenario,
            network_profile=network_profile,
            seed=seed,
            success_rate=success_rate,
            steps=steps,
            cloud_tokens=c_tot,
            device_tokens=d_tot,
            total_tokens=t_tot,
            cloud_api_calls=cloud_api_calls,
            device_api_calls=device_api_calls,
            total_api_calls=cloud_api_calls + device_api_calls,
            device_memory_mb=device_memory_mb if device_memory_mb > 0.0 else process_peak_rss_mb,
            computation_s=computation_s,
            total_wall_clock_s=total_wall_clock_s,
            tfr=float(np.mean(tfr_history)) if tfr_history else 1.0,
            cfr=float(np.mean(cfr_history)) if cfr_history else 1.0,
            switch_count=switch_count,
            peer_messages=peer_messages,
            broadcast_count=broadcast_count,
            consensus_rounds=consensus_rounds,
            consensus_latency=consensus_latency,
            plan_merge_count=plan_merge_count,
            distributed_replanning_count=distributed_replanning_count,
            replanning_count=replanning_count,
            local_reallocation_count=local_reallocation_count,
            reallocation_trigger_count=reallocation_trigger_count,
            reallocation_skip_count=reallocation_skip_count,
            reallocation_reasons=dict(reallocation_reasons or {}),
            cached_plan_reuse_count=cached_plan_reuse_count,
            merged_singleton_count=merged_singleton_count,
            avg_planning_latency=avg_planning_latency,
            coalition_change_count=coalition_change_count,
            communication_steps=communication_steps,
            paper_communication_steps=paper_comm_steps,
            communication_step_breakdown=breakdown_dict,
            hallucination_stats=dict(hallucination_stats or {}),
            experience_reuse_attempts=experience_reuse_attempts,
            experience_reuse_hits=experience_reuse_hits,
            dispatch_skipped_rounds=dispatch_skipped_rounds,
            cloud_prompt_tokens=cloud_prompt_tokens,
            cloud_completion_tokens=cloud_completion_tokens,
            cloud_total_tokens=c_tot,
            device_prompt_tokens=device_prompt_tokens,
            device_completion_tokens=device_completion_tokens,
            device_total_tokens=d_tot,
            total_prompt_tokens=cloud_prompt_tokens + device_prompt_tokens,
            total_completion_tokens=cloud_completion_tokens + device_completion_tokens,
            cloud_retry_tokens=cloud_retry_tokens,
            device_retry_tokens=device_retry_tokens,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            retried_calls=retried_calls,
            cache_hits=cache_hits,
            local_non_llm_operations=local_non_llm_operations,
            cloud_network_calls=cloud_network_calls,
            cloud_disk_cache_hits=cloud_disk_cache_hits,
            cloud_failed_attempts=cloud_failed_attempts,
            semantic_cache_hits=semantic_cache_hits,
            cloud_call_attribution=dict(cloud_call_attribution or {}),
            logical_llm_requests=logical_llm_requests,
            device_inference_calls=device_inference_calls,
            process_peak_rss_mb=process_peak_rss_mb if process_peak_rss_mb > 0.0 else device_memory_mb,
            process_mean_rss_mb=process_mean_rss_mb,
            gpu_peak_memory_mb=gpu_peak_memory_mb,
            gpu_mean_memory_mb=gpu_mean_memory_mb,
            device_llm_memory_mb=dict(device_llm_memory_mb or {}),
            device_llm_memory_peak_mb=dict(device_llm_memory_peak_mb or {}),
            device_llm_heap_delta_mb=dict(device_llm_heap_delta_mb or {}),
            inference_backend_memory_mb=inference_backend_memory_mb,
            device_llm_python_heap_delta_mb=device_llm_python_heap_delta_mb,
            device_llm_python_heap_delta_by_device=dict(device_llm_python_heap_delta_by_device or {}),
            device_llm_tokens_processed_by_device=dict(device_llm_tokens_processed_by_device or {}),
            cloud_inference_time_s=cloud_inference_time_s,
            device_inference_time_s=device_inference_time_s,
            cqi_evaluation_time_s=cqi_evaluation_time_s,
            coalition_computation_time_s=coalition_computation_time_s,
            architecture_switching_time_s=architecture_switching_time_s,
            snapshot_capture_time_s=snapshot_capture_time_s,
            state_restore_time_s=state_restore_time_s,
            state_verification_time_s=state_verification_time_s,
            coalition_transfer_time_s=coalition_transfer_time_s,
            reallocation_time_s=reallocation_time_s,
            state_handoff_time_s=state_handoff_time_s,
            coalition_repair_time_s=coalition_repair_time_s,
            consensus_time_s=consensus_time_s,
            planning_time_s=planning_time_s,
            network_waiting_time_s=network_waiting_time_s,
            simulation_computation_time_s=simulation_computation_time_s,
            planning_latency_p50=planning_latency_p50,
            planning_latency_p95=planning_latency_p95,
            planning_latency_p99=planning_latency_p99,
            planning_latency_min=planning_latency_min,
            planning_latency_max=planning_latency_max,
            planning_latency_std=planning_latency_std,
            cloud_to_device_messages=cloud_to_device_messages,
            device_to_cloud_messages=device_to_cloud_messages,
            handoff_messages=handoff_messages,
            coalition_messages=coalition_messages,
            repair_messages=repair_messages,
            cloud_bytes=cloud_bytes,
            peer_bytes=peer_bytes,
            broadcast_bytes=broadcast_bytes,
            total_bytes=total_bytes if total_bytes > 0 else (cloud_bytes + peer_bytes + broadcast_bytes),
            prompt_reduction_percent=prompt_reduction_percent,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hit_rate,
            saved_cloud_calls=saved_cloud_calls,
            saved_tokens=saved_tokens,
            saved_latency=saved_latency,
            local_reasoning_count=local_reasoning_count,
            cloud_reasoning_count=cloud_reasoning_count,
            confidence_distribution=dict(confidence_distribution or {}),
            consensus_skipped=consensus_skipped,
            consensus_duration=consensus_duration,
            planner_latency=planner_latency if planner_latency > 0.0 else avg_planning_latency,
        )
        self.records.append(m)
        return m

    def summary_table(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    @staticmethod
    def aggregate_by_config(records: list[ExperimentMetrics]) -> dict[str, dict]:
        groups: dict[str, list[ExperimentMetrics]] = {}
        for r in records:
            key = f"{r.config_name}_{r.scenario}_{r.network_profile}"
            groups.setdefault(key, []).append(r)
        result = {}
        for key, group in groups.items():
            result[key] = {
                "success_mean": float(np.mean([g.success_rate for g in group])),
                "success_std": float(np.std([g.success_rate for g in group])),
                "steps_mean": float(np.mean([g.steps for g in group])),
                "tokens_mean": float(np.mean([g.total_tokens for g in group])),
                "api_calls_mean": float(np.mean([g.total_api_calls for g in group])),
                "tfr_mean": float(np.mean([g.tfr for g in group])),
                "cfr_mean": float(np.mean([g.cfr for g in group])),
                "sc_mean": float(np.mean([g.switch_count for g in group])),
                "peer_messages_mean": float(np.mean([g.peer_messages for g in group])),
                "consensus_rounds_mean": float(np.mean([g.consensus_rounds for g in group])),
                "comm_steps_mean": float(np.mean([g.communication_steps for g in group])),
                "n_seeds": len(group),
            }
        return result

    @staticmethod
    def significance_test(
        group_a: list[float], group_b: list[float]
    ) -> dict[str, float]:
        from scipy import stats

        if len(group_a) < 2 or len(group_b) < 2:
            return {"t_stat": 0.0, "p_value": 1.0}
        t_stat, p_value = stats.ttest_ind(group_a, group_b)
        return {"t_stat": float(t_stat), "p_value": float(p_value)}
