"""Main orchestrator wiring all DACA-HMAS modules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.acds.switch_engine import ACDSSwitchEngine
from src.coalition.formation import CoalitionFormation
from src.communication.models import discover_agent_type_domains
from src.communication.peer_manager import PeerCommunicationManager
from src.config import get_llm_config, get_thresholds
from src.coordination.centralized_hybrid import CentralizedHybridCoordinator
from src.coordination.decentralized_hybrid import DecentralizedHybridCoordinator
from src.cqm.monitor import CommunicationQualityMonitor
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer
from src.env.agents import AgentFleet, distance_matrix, dist
from src.env.daca_env import DACAEnv
from src.handoff.ca_transfer import CATransferManager
from src.handoff.snapshot import (
    capture_snapshot,
    restore_distributed_state,
    restore_snapshot,
    verify_task_preservation,
)
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient, aggregate_device_usage
from src.metrics.evaluation import ExperimentMetrics, MetricsCollector
from src.reallocation.post_switch import PostSwitchReallocator
from src.llm.exceptions import ExperimentFailed
from src.coordination.replan_trigger import PlanState, should_replan, update_plan_state



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


def _build_device_llms_by_type(
    fleet: AgentFleet, llm_cfg: dict[str, Any]
) -> dict[str, DeviceLLMClient]:
    """Create one DeviceLLMClient per agent-type domain (centralized + decentralized)."""
    domains = discover_agent_type_domains(fleet)
    return {
        domain: DeviceLLMClient.for_domain(domain, agent_ids, llm_cfg)
        for domain, agent_ids in domains.items()
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
        self.cloud_llm.configure_experiment_context(
            scenario=self.scenario,
            architecture=self.config.name,
            network_profile=self.network_profile,
            seed=self.seed,
        )

        self.device_llms: dict[str, DeviceLLMClient] = _build_device_llms_by_type(
            self.env.fleet, llm_cfg
        )
        self.cloud_llm.device_fallback_decompose = self._device_fallback_decompose
        self.cloud_llm.device_fallback_coalitions = self._device_fallback_coalitions
        self.peer_manager = PeerCommunicationManager(rng=np.random.default_rng(self.seed))
        self.peer_manager.register_domain_peers(list(self.device_llms.keys()))

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
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            decomposer=self.decomposer,
            coalition_formation=self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
        )
        self.decentralized = DecentralizedHybridCoordinator(
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            peer_manager=self.peer_manager,
            decomposer=self.decomposer,
            coalition_formation=self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
        )
        self.ca_transfer = CATransferManager(
            overlap_delta=self.thresholds.get("ca_overlap_delta", 3)
        )
        self.reallocator = PostSwitchReallocator(
            device_llms=self.device_llms,
            coalition_formation=self.coalition_formation,
            peer_manager=self.peer_manager,
        )
        self.metrics = MetricsCollector()
        self._plan_state = PlanState()

    @property
    def device_llm(self) -> DeviceLLMClient:
        """Backward-compatible accessor — first domain Device LLM."""
        if not self.device_llms:
            return DeviceLLMClient()
        return next(iter(self.device_llms.values()))
    
    def _device_fallback_decompose(self, instruction, agents, subtasks):
        client = self.device_llm
        n = len(agents)
        dist_mat = [[0.0] * n for _ in range(n)]
        cqi_mat = [[1.0] * n for _ in range(n)]
        coalitions = client.reallocate_remaining(
            subtasks, agents, dist_mat, cqi_mat, scope_to_managed=False
        )
        assignments: dict[str, list[str]] = {}
        for i, c in enumerate(coalitions):
            members = c.get("members", [])
            if not members or i >= len(subtasks):
                continue
            sid = subtasks[i].get("id", subtasks[i].get("subtask_id", f"T_{i}"))
            assignments[sid] = [members[0]]
        return assignments

    def _device_fallback_coalitions(self, subtasks, agents, distance_matrix, cqi_matrix):
        client = self.device_llm
        dmat = distance_matrix if distance_matrix is not None else [[0.0] * len(agents)] * len(agents)
        qmat = cqi_matrix if cqi_matrix is not None else [[1.0] * len(agents)] * len(agents)
        return client.reallocate_remaining(subtasks, agents, dmat, qmat, scope_to_managed=False)

    def run(self) -> ExperimentMetrics:
        import inspect
        print("========== RUN STARTED ==========")
        print(inspect.getfile(self.__class__))
        start = time.perf_counter()
        self.env.reset()
        self._plan_state = PlanState()
        self._replanning_count = 0
        self._coalition_change_count = 0
        self._planning_latency_total = 0.0
        self._planning_latency_count = 0
        assignments: dict = {}
        coalitions: list = []
        tfr_history: list[float] = []
        cfr_history: list[float] = []
        prev_mode = self.acds.mode

        for step in range(self.max_steps):
            self.cloud_llm.set_step(step)
            fleet = self.env.fleet
            dist_mat = distance_matrix(fleet.agents)

            for node_id in range(fleet.n_agents):
                net_state = self.env.network.simulate_message(step)
                if self.config.use_cqm:
                    self.cqm.update_from_network(node_id, net_state)
                if node_id == 0 and step % 20 == 0:
                    print(
                   f"[NETWORK] Step={step} "
                   f"Loss={net_state.packet_loss_rate:.3f} "
                   f"Latency={net_state.latency:.3f} "
                   f"BW_Util={net_state.bandwidth_utilization:.3f}"
                   )
            cqi_matrix = self.cqm.update_pairwise(
                dist_mat, self.thresholds.get("C1", 50.0)
            )
            sys_cqi = self.cqm.system_cqi() if self.config.use_cqm else 1.0
            if self.config.use_cqm and fleet.n_agents:
                avg_packet_loss = float(
                    np.mean([self.cqm.packet_loss_rate(n) for n in range(fleet.n_agents)])
                )
                avg_latency = float(
                    np.mean([self.cqm.normalized_latency(n) for n in range(fleet.n_agents)])
                )
            else:
                avg_packet_loss, avg_latency = 0.0, 0.0

            if self.config.use_acds and self.config.static_mode is None:
                mode = self.acds.evaluate(sys_cqi)
                print(f"[MODE] step={step} mode={mode} "f"centralized={mode==0} decentralized={mode==1}")
                if step % 20 == 0:
                   print(
                           f"[ACDS] Step={step} "
                           f"CQI={sys_cqi:.3f} "
                           f"Mode={mode} "
                           f"Switches={self.acds.switch_count}"
               )
            else:
                mode = self.acds.mode
            if step % 20 == 0:
                  print(
                        f"ThetaDown={self.acds.theta_down:.3f} "
                        f"ThetaUp={self.acds.theta_up:.3f}"
             )
            if mode != prev_mode and self.config.use_handoff:
                snap = capture_snapshot(
                    fleet, self.env.subtask_list, coalitions,
                    step, prev_mode, mode,
                    shared_plans=self.decentralized.shared_plans,
                    device_llms=self.device_llms,
                    pending_messages=self.peer_manager.pending_messages_all(),
                )
                restore_snapshot(fleet, snap)
                restore_distributed_state(
                    snap, self.device_llms, self.peer_manager, self.decentralized
                )
                self.ca_transfer.on_mode_change(mode)
                if self.config.use_reallocation and self.reallocator.should_trigger(
                    True, coalitions, fleet, dist_mat, cqi_matrix
                ):
                    coalitions = self.reallocator.reallocate(
                        fleet, self.env.subtask_list, coalitions, dist_mat, cqi_matrix
                    )
                verify_task_preservation(snap, snap)
                prev_mode = mode

            replan_now, replan_reason = should_replan(
                self._plan_state,
                self.env.subtask_list,
                fleet,
                coalitions,
                mode=mode,
                sys_cqi=sys_cqi,
                packet_loss=avg_packet_loss,
                latency=avg_latency,
                cqi_delta_threshold=self.thresholds.get("acds", {}).get("delta", 0.08),
            )
        
            if replan_now:
                print(f"[REPLAN] step={step} reason={replan_reason}")
                prev_membership = dict(self._plan_state.coalition_members)

                t_plan = time.perf_counter()
                if mode == 0:
                    assignments, coalitions = self.centralized.plan(self.env, cqi_matrix)
                    print(">>>> USING CENTRALIZED")
                else:
                    assignments, coalitions = self.decentralized.plan(self.env, cqi_matrix)
                    print(">>>> USING DECENTRALIZED")
                self._planning_latency_total += time.perf_counter() - t_plan
                self._planning_latency_count += 1
                self._replanning_count += 1

                new_membership = {
                    c.get("coalition_id"): frozenset(c.get("members", [])) for c in coalitions
                }
                if new_membership != prev_membership:
                    self._coalition_change_count += 1

                update_plan_state(
                    self._plan_state,
                    self.env.subtask_list,
                    fleet,
                    coalitions,
                    assignments,
                    mode=mode,
                    sys_cqi=sys_cqi,
                    packet_loss=avg_packet_loss,
                    latency=avg_latency,
                )
            else:
                print(f"[REPLAN] step={step} skipped -- reusing existing plan")

            print(f"\n[ASSIGN] Step={step}")
            for sid, agents in assignments.items():
                print(f"{sid} -> {agents}")

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
                print(
                    f"{sid} -> {agent_list}"
                )
                if not agent_list:
                    continue
                agent = fleet.get_agent(agent_list[0])
                subtask = next(
                    (s for s in self.env.subtask_list if s.subtask_id == sid), None
                )
                print("Found subtask:", subtask)
                if subtask:
                    if step % 50 == 0:
                       print(
                           f"[DIST] Step={step} "
                           f"Task={sid} "
                           f"Agent={agent.agent_id} "
                           f"Distance={dist(agent.position, subtask.target):.2f}"
                       )
                    
                    if dist(agent.position, subtask.target) < 8.0:
                        self.env.mark_subtask_complete(sid)

            self.env.advance()
            if step % 20 == 0:
                print(
                    f"[MISSION] Step={step} "
                    f"Completed={self.env.success_rate():.2f}% "
                    f"MissionDone={self.env.state.mission_complete}"
       )
            if self.env.state.mission_complete:
                break
        
        elapsed = time.perf_counter() - start
        device_usage = aggregate_device_usage(self.device_llms)
        peer_metrics = self.peer_manager.metrics_snapshot()

        return self.metrics.finalize(
            success_rate=self.env.success_rate(),
            steps=self.env.state.timestep,
            cloud_tokens=self.cloud_llm.usage.tokens,
            cloud_api_calls=self.cloud_llm.usage.api_calls,
            device_tokens=device_usage.tokens,
            device_api_calls=device_usage.api_calls,
            device_memory_mb=device_usage.memory_mb,
            computation_s=elapsed,
            tfr_history=tfr_history,
            cfr_history=cfr_history,
            switch_count=self.acds.switch_count_metric(),
            config_name=self.config.name,
            scenario=self.scenario,
            network_profile=self.network_profile,
            seed=self.seed,
            peer_messages=int(peer_metrics["peer_messages"]),
            broadcast_count=int(peer_metrics["broadcast_count"]),
            consensus_rounds=int(peer_metrics["consensus_rounds"]),
            consensus_latency=float(peer_metrics["consensus_latency"]),
            plan_merge_count=int(peer_metrics["plan_merge_count"]),
            distributed_replanning_count=int(peer_metrics["distributed_replanning_count"]),
            replanning_count=self._replanning_count,
            local_reallocation_count=getattr(self.decentralized, "local_reallocation_count", 0),
            cached_plan_reuse_count=self.cloud_llm.usage.cache_hits + device_usage.cache_hits,
            avg_planning_latency=(
                self._planning_latency_total / self._planning_latency_count
                if self._planning_latency_count else 0.0
            ),
            coalition_change_count=self._coalition_change_count,
        )
