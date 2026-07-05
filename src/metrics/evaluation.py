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
    tfr: float = 1.0
    cfr: float = 1.0
    switch_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config_name,
            "scenario": self.scenario,
            "profile": self.network_profile,
            "seed": self.seed,
            "success_rate": round(self.success_rate * 100, 2),
            "steps": self.steps,
            "tokens": self.total_tokens,
            "api_calls": self.total_api_calls,
            "memory_mb": round(self.device_memory_mb, 1),
            "computation_s": round(self.computation_s, 3),
            "tfr": round(self.tfr, 4),
            "cfr": round(self.cfr, 4),
            "switch_count": self.switch_count,
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
        tfr_history: list[float],
        cfr_history: list[float],
        switch_count: int,
        config_name: str,
        scenario: str,
        network_profile: str,
        seed: int,
    ) -> ExperimentMetrics:
        m = ExperimentMetrics(
            config_name=config_name,
            scenario=scenario,
            network_profile=network_profile,
            seed=seed,
            success_rate=success_rate,
            steps=steps,
            cloud_tokens=cloud_tokens,
            device_tokens=device_tokens,
            total_tokens=cloud_tokens + device_tokens,
            cloud_api_calls=cloud_api_calls,
            device_api_calls=device_api_calls,
            total_api_calls=cloud_api_calls + device_api_calls,
            device_memory_mb=device_memory_mb,
            computation_s=computation_s,
            tfr=float(np.mean(tfr_history)) if tfr_history else 1.0,
            cfr=float(np.mean(cfr_history)) if cfr_history else 1.0,
            switch_count=switch_count,
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
