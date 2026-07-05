"""Main orchestrator wiring all DACA-HMAS modules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.acds.switch_engine import ACDSSwitchEngine
from src.coalition.formation import CoalitionFormation
from src.config import get_llm_config, get_thresholds
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.decentralized_hybrid import DecentralizedHybridCoordinator
from src.cqm.monitor import CommunicationQualityMonitor
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer
from src.env.agents import distance_matrix
from src.env.daca_env import DACAEnv
from src.handoff.ca_transfer import CATransferManager
from src.handoff.snapshot import capture_snapshot, restore_snapshot, verify_task_preservation
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector
from src.reallocation.post_switch import PostSwitchReallocator


@dataclass
class DACAConfig:
    """Experiment configuration flags."""
    name: str = "A5"
    use_distance_decomp: bool = True
    use_coalition_feasibility: bool = True
    use_cqm: bool = True
    use_acds: bool = True
    use_handoff: bool = True
    use_reallocation: bool = True
    use_hysteresis: bool = True
    static_mode: int | None = None


CONFIGS: dict[str, DACAConfig] = {
    "B1": DACAConfig(name="B1", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0),
    "B2": DACAConfig(name="B2", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=1),
    "A1": DACAConfig(name="A1", use_distance_decomp=True, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0),
    "A2": DACAConfig(name="A2", use_distance_decomp=False, use_coalition_feasibility=True,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0),
    "A3": DACAConfig(name="A3", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=True, use_acds=True, use_handoff=False, use_reallocation=False),
    "A4": DACAConfig(name="A4", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=True, use_acds=True, use_hysteresis=False, use_handoff=False, use_reallocation=False),
    "A5": DACAConfig(name="A5"),
}


@dataclass
class DACAOrchestrator:
    scenario: str
    network_profile: str
    seed: int
    config: DACAConfig = field(default_factory=lambda: CONFIGS["A5"])
    thresholds: dict[str, Any] = field(default_factory=get_thresholds)
    max_steps: int = 200
    replan_interval: int = 20

    def __post_init__(self) -> None:
        self.env = DACAEnv(
            self.scenario, self.thresholds, self.network_profile, self.seed, self.max_steps
        )
        llm_cfg = get_llm_config()
        self.cloud_llm = CloudLLMClient(llm_cfg)
        self.device_llm = DeviceLLMClient(llm_cfg)
        n = self.env.fleet.n_agents
        self.cqm = CommunicationQualityMonitor.from_config(self.thresholds, n)
        self.acds = ACDSSwitchEngine.from_config(
            self.thresholds, use_hysteresis=self.config.use_hysteresis
        )
        if self.config.static_mode is not None:
            self.acds.mode = self.config.static_mode
        self.decomposer = DistanceFeasibleDecomposer(
            self.cloud_llm,
            c_task=self.thresholds.get("C_task", 30.0),
            r_reach=self.thresholds.get("R_reach", 100.0),
        )
        self.coalition_formation = CoalitionFormation(
            self.cloud_llm,
            c1=self.thresholds.get("C1", 50.0),
            gamma_min=self.thresholds.get("gamma_min", 0.3),
        )
        self.centralized = CentralizedHybridCoordinator(
            self.cloud_llm, self.device_llm, self.decomposer, self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
        )
        self.decentralized = DecentralizedHybridCoordinator(
            self.cloud_llm, self.device_llm, self.decomposer, self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
        )
        self.ca_transfer = CATransferManager(
            overlap_delta=self.thresholds.get("ca_overlap_delta", 3)
        )
        self.reallocator = PostSwitchReallocator(self.device_llm, self.coalition_formation)
        self.metrics = MetricsCollector()

    def run(self) -> ExperimentMetrics:
        start = time.perf_counter()
        self.env.reset()
        assignments: dict = {}
        coalitions: list = []
        tfr_history: list[float] = []
        cfr_history: list[float] = []
        prev_mode = self.acds.mode

        for step in range(self.max_steps):
            fleet = self.env.fleet
            dist_mat = distance_matrix(fleet.agents)

            for node_id in range(fleet.n_agents):
                net_state = self.env.network.simulate_message(step)
                if self.config.use_cqm:
                    self.cqm.update_from_network(node_id, net_state)

            cqi_matrix = self.cqm.update_pairwise(
                dist_mat, self.thresholds.get("C1", 50.0)
            )
            sys_cqi = self.cqm.system_cqi() if self.config.use_cqm else 1.0

            if self.config.use_acds and self.config.static_mode is None:
                mode = self.acds.evaluate(sys_cqi)
            else:
                mode = self.acds.mode

            if mode != prev_mode and self.config.use_handoff:
                snap = capture_snapshot(
                    fleet, self.env.subtask_list, coalitions,
                    step, prev_mode, mode,
                )
                restore_snapshot(fleet, snap)
                self.ca_transfer.on_mode_change(mode)
                if self.config.use_reallocation and self.reallocator.should_trigger(
                    True, coalitions, fleet, dist_mat, cqi_matrix
                ):
                    coalitions = self.reallocator.reallocate(
                        fleet, self.env.subtask_list, coalitions, dist_mat, cqi_matrix
                    )
                verify_task_preservation(snap, snap)
                prev_mode = mode

            if step % self.replan_interval == 0 or not assignments:
                if mode == 0:
                    assignments, coalitions = self.centralized.plan(self.env, cqi_matrix)
                else:
                    assignments, coalitions = self.decentralized.plan(self.env, cqi_matrix)

            if self.config.use_distance_decomp:
                from src.decomposition.distance_feasible_decomp import compute_tfr
                tfr = compute_tfr(
                    assignments, self.env.subtask_list, fleet,
                    self.thresholds.get("C_task", 30.0),
                    self.thresholds.get("R_reach", 100.0),
                )
                tfr_history.append(tfr)

            if self.config.use_coalition_feasibility:
                cfr = self.coalition_formation.compute_cfr(
                    coalitions, fleet, dist_mat, cqi_matrix
                )
                cfr_history.append(cfr)

            targets = {s.subtask_id: s.target for s in self.env.subtask_list}
            agent_assignments = {}
            for sid, agents in assignments.items():
                if agents:
                    agent_assignments[agents[0]] = sid

            self.ca_transfer.step(self.env.fleet, mode, agent_assignments, targets)

            for sid, agent_list in assignments.items():
                if not agent_list:
                    continue
                agent = fleet.get_agent(agent_list[0])
                subtask = next(
                    (s for s in self.env.subtask_list if s.subtask_id == sid), None
                )
                if subtask:
                    from src.env.agents import dist
                    if dist(agent.position, subtask.target) < 8.0:
                        self.env.mark_subtask_complete(sid)

            self.env.advance()
            if self.env.state.mission_complete:
                break

        elapsed = time.perf_counter() - start
        return self.metrics.finalize(
            success_rate=self.env.success_rate(),
            steps=self.env.state.timestep,
            cloud_tokens=self.cloud_llm.usage.tokens,
            cloud_api_calls=self.cloud_llm.usage.api_calls,
            device_tokens=self.device_llm.usage.tokens,
            device_api_calls=self.device_llm.usage.api_calls,
            device_memory_mb=self.device_llm.usage.memory_mb,
            computation_s=elapsed,
            tfr_history=tfr_history,
            cfr_history=cfr_history,
            switch_count=self.acds.switch_count_metric(),
            config_name=self.config.name,
            scenario=self.scenario,
            network_profile=self.network_profile,
            seed=self.seed,
        )
