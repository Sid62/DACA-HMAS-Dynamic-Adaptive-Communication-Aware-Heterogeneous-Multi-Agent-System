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
from src.metrics.communication_counter import CommunicationStepCounter
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
    use_optimizations: bool = True


CONFIGS: dict[str, DACAConfig] = {
    "B1": DACAConfig(name="B1", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0, use_optimizations=False),
    "B2": DACAConfig(name="B2", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=1, use_optimizations=False),
    "A1": DACAConfig(name="A1", use_distance_decomp=True, use_coalition_feasibility=False,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0),
    "A2": DACAConfig(name="A2", use_distance_decomp=False, use_coalition_feasibility=True,
                     use_cqm=False, use_acds=False, use_handoff=False, use_reallocation=False, static_mode=0),
    "A3": DACAConfig(name="A3", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=True, use_acds=True, use_handoff=False, use_reallocation=False),
    "A4": DACAConfig(name="A4", use_distance_decomp=False, use_coalition_feasibility=False,
                     use_cqm=True, use_acds=True, use_hysteresis=False, use_handoff=False, use_reallocation=False),
    "A5": DACAConfig(name="A5", use_optimizations=True),
    "A5_unopt": DACAConfig(name="A5_unopt", use_optimizations=False),
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


from src.coordination.plan_continuity import PlanContinuityEngine
from src.coordination.plan_repair import PlanRepairer
from src.handoff.delta_transfer import DeltaStateTransferManager


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

        if self.config.use_optimizations:
            self.continuity_engine = PlanContinuityEngine(
                validity_threshold=self.thresholds.get("plan_validity_threshold", 0.75),
                r_reach=self.thresholds.get("R_reach", 100.0),
                c_task=self.thresholds.get("C_task", 30.0),
            )
            self.plan_repairer = PlanRepairer(r_reach=self.thresholds.get("R_reach", 100.0))
            self.delta_transfer_manager = DeltaStateTransferManager()
        else:
            self.cloud_llm.config["cache_responses"] = False
            for d_client in self.device_llms.values():
                d_client.config["cache_responses"] = False
            self.continuity_engine = None
            self.plan_repairer = None
            self.delta_transfer_manager = None


        from src.memory.experience_store import SubtaskExperienceStore
        exp_cfg = llm_cfg.get("experience_reuse", {})
        self.experience_store = SubtaskExperienceStore(
            store_path=exp_cfg.get("store_path", "experience_store.json"),
            enabled=exp_cfg.get("enabled", False),
        )

        self.centralized = CentralizedHybridCoordinator(
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            decomposer=self.decomposer,
            coalition_formation=self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
            continuity_engine=self.continuity_engine,
            plan_repairer=self.plan_repairer,
            experience_store=self.experience_store,
        )
        self.decentralized = DecentralizedHybridCoordinator(
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            peer_manager=self.peer_manager,
            decomposer=self.decomposer,
            coalition_formation=self.coalition_formation,
            use_distance_decomp=self.config.use_distance_decomp,
            use_coalition_feasibility=self.config.use_coalition_feasibility,
            continuity_engine=self.continuity_engine,
            plan_repairer=self.plan_repairer,
            experience_store=self.experience_store,
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
        self.comm_counter = CommunicationStepCounter()
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
        import psutil
        import os
        print("========== RUN STARTED ==========")
        print(inspect.getfile(self.__class__))
        start = time.perf_counter()
        self.env.reset()
        self.comm_counter.reset()
        self.cloud_llm.usage.reset()
        for d_client in self.device_llms.values():
            d_client.usage.reset()
        self.peer_manager.reset_metrics()

        if hasattr(self, "experience_store") and self.experience_store is not None:
            self.experience_store.reset_run_metrics()
        self._plan_state = PlanState()
        self.decentralized.plan_reuse_count = 0
        self.coalition_formation.merged_singleton_count = 0
        self._replanning_count = 0
        self._coalition_change_count = 0
        self._planning_latency_total = 0.0
        self._planning_latency_count = 0

        planning_latencies: list[float] = []
        process_rss_samples: list[float] = []

        cqi_evaluation_time_s: float = 0.0
        coalition_computation_time_s: float = 0.0
        architecture_switching_time_s: float = 0.0
        state_handoff_time_s: float = 0.0
        coalition_repair_time_s: float = 0.0
        consensus_time_s: float = 0.0
        planning_time_s: float = 0.0
        network_waiting_time_s: float = 0.0
        simulation_computation_time_s: float = 0.0

        assignments: dict = {}
        coalitions: list = []
        tfr_history: list[float] = []
        cfr_history: list[float] = []
        prev_mode = self.acds.mode

        for step in range(self.max_steps):
            t_sim_step_start = time.perf_counter()
            process_rss_samples.append(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
            self.cloud_llm.set_step(step)
            self.peer_manager.set_step(step)
            for d_client in self.device_llms.values():
                d_client.set_step(step)
            fleet = self.env.fleet
            dist_mat = distance_matrix(fleet.agents)

            t_cqi_start = time.perf_counter()
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
            cqi_evaluation_time_s += (time.perf_counter() - t_cqi_start)
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

            t_acds_start = time.perf_counter()
            if self.config.use_acds and self.config.static_mode is None:
                mode = self.acds.evaluate(sys_cqi, step)

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
            architecture_switching_time_s += (time.perf_counter() - t_acds_start)
            if step % 20 == 0:
                  print(
                        f"ThetaDown={self.acds.theta_down:.3f} "
                        f"ThetaUp={self.acds.theta_up:.3f}"
             )
            if mode != prev_mode:
                prev_mode_name = "Centralized" if prev_mode == 0 else "Decentralized"
                mode_name = "Centralized" if mode == 0 else "Decentralized"
                print(f"\nStep {step}")
                print(f"{prev_mode_name} -> {mode_name}\n")

            if mode != prev_mode and self.config.use_handoff:
                t_handoff_start = time.perf_counter()
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
                self.comm_counter.increment("handoff_reallocation", 1, "mode_handoff_snapshot_transfer")
                if self.config.use_reallocation and self.reallocator.should_trigger(
                    True, coalitions, fleet, dist_mat, cqi_matrix
                ):
                    t_realloc_start = time.perf_counter()
                    coalitions = self.reallocator.reallocate(
                        fleet, self.env.subtask_list, coalitions, dist_mat, cqi_matrix
                    )
                    coalition_computation_time_s += (time.perf_counter() - t_realloc_start)
                    self.comm_counter.increment("handoff_reallocation", 1, "post_switch_coalition_reallocation")
                verify_task_preservation(snap, snap)
                state_handoff_time_s += (time.perf_counter() - t_handoff_start)
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
                cqi_delta_threshold=self.thresholds.get(
                    "communication_change_threshold", 0.08
                ),
                current_step=step,
                minimum_replanning_interval=self.thresholds.get(
                    "minimum_replanning_interval", 0
                ),
                continuity_engine=self.continuity_engine,
                cqi_matrix=cqi_matrix,
            )
        
            if replan_now:
                print(f"[REPLAN] step={step} reason={replan_reason}")
                self.cloud_llm.active_replan_reason = replan_reason
                prev_membership = dict(self._plan_state.coalition_members)

                t_plan = time.perf_counter()
                if mode == 0:
                    assignments, coalitions = self.centralized.plan(self.env, cqi_matrix)
                    self.comm_counter.increment("global_planning", 1, "centralized_global_planning")
                    self.comm_counter.increment("dispatch", 1, "centralized_domain_dispatch")
                    print(">>>> USING CENTRALIZED")
                else:
                    print("\nEntering decentralized planner\n")
                    assignments, coalitions = self.decentralized.plan(self.env, cqi_matrix)
                    self.comm_counter.increment("local_coordination", 1, "decentralized_leader_planning")
                    self.comm_counter.increment("peer_consensus", 1, "decentralized_peer_review_consensus")
                    self.comm_counter.increment("feedback_sync", 1, "decentralized_state_sync")
                    print(">>>> USING DECENTRALIZED")
                plan_lat = time.perf_counter() - t_plan
                planning_time_s += plan_lat
                planning_latencies.append(plan_lat)
                self._planning_latency_total += plan_lat
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
                    current_step=step,
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
                t_cfr_start = time.perf_counter()
                cfr = self.coalition_formation.compute_cfr(
                    coalitions, fleet, dist_mat, cqi_matrix
                )
                coalition_computation_time_s += (time.perf_counter() - t_cfr_start)
                cfr_history.append(cfr)

            targets = {s.subtask_id: s.target for s in self.env.subtask_list}
            agent_assignments = {}
            for sid, agents in assignments.items():
                if agents:
                    agent_assignments[agents[0]] = sid

            t_sim_body = time.perf_counter()
            self.ca_transfer.step(self.env.fleet, mode, agent_assignments, targets)

            for sid, agent_list in assignments.items():
                if not agent_list:
                    continue
                agent = fleet.get_agent(agent_list[0])
                subtask = next(
                    (s for s in self.env.subtask_list if s.subtask_id == sid), None
                )
                if subtask:
                    if step % 50 == 0:
                       print(
                           f"[DIST] Step={step} "
                           f"Task={sid} "
                           f"Agent={agent.agent_id} "
                           f"Distance={dist(agent.position, subtask.target):.2f}"
                       )
                    v_agent = getattr(agent, "speed", 3.0)
                    effective_radius = 8.0 + max(0.0, float(avg_latency)) * v_agent * 2.0
                    if dist(agent.position, subtask.target) < effective_radius:
                        was_completed = subtask.completed
                        self.env.mark_subtask_complete(sid)
                        if not was_completed and hasattr(self, "experience_store") and self.experience_store is not None and self.experience_store.enabled:
                            from src.memory.experience_store import compute_signature
                            agent_types = [a.agent_type.value for a in fleet.agents]
                            d_lead = dist(agent.position, subtask.target)
                            sig = compute_signature(self.scenario, subtask.required_skills, agent_types, d_lead)
                            self.experience_store.record(
                                signature=sig,
                                plan={sid: assignments.get(sid, [agent.agent_id])},
                                success=True,
                                scenario=self.scenario,
                                skills=subtask.required_skills,
                                agent_types=agent_types,
                            )

            self.env.advance()
            simulation_computation_time_s += (time.perf_counter() - t_sim_body)
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
        total_llm_wait_s = self.cloud_llm.usage.llm_wait_s + device_usage.llm_wait_s
        local_computation_s = max(0.0, elapsed - total_llm_wait_s)

        # Planning latency statistics
        if planning_latencies:
            plat_p50 = float(np.percentile(planning_latencies, 50))
            plat_p95 = float(np.percentile(planning_latencies, 95))
            plat_p99 = float(np.percentile(planning_latencies, 99))
            plat_min = float(np.min(planning_latencies))
            plat_max = float(np.max(planning_latencies))
            plat_std = float(np.std(planning_latencies))
        else:
            plat_p50 = plat_p95 = plat_p99 = plat_min = plat_max = plat_std = 0.0

        # Memory statistics
        peak_rss = float(np.max(process_rss_samples)) if process_rss_samples else device_usage.memory_mb
        mean_rss = float(np.mean(process_rss_samples)) if process_rss_samples else device_usage.memory_mb

        gpu_peak = 0.0
        gpu_mean = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_peak = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                gpu_mean = float(torch.cuda.memory_allocated() / (1024 * 1024))
        except ImportError:
            pass

        return self.metrics.finalize(
            success_rate=self.env.success_rate(),
            steps=self.env.state.timestep,
            cloud_tokens=self.cloud_llm.usage.total_tokens,
            cloud_api_calls=self.cloud_llm.usage.cloud_api_calls,
            device_tokens=device_usage.total_tokens,
            device_api_calls=device_usage.device_api_calls,
            device_memory_mb=peak_rss,
            computation_s=local_computation_s,
            total_wall_clock_s=elapsed,
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
            cached_plan_reuse_count=self.decentralized.plan_reuse_count,
            merged_singleton_count=self.coalition_formation.merged_singleton_count,
            avg_planning_latency=(
                self._planning_latency_total / self._planning_latency_count
                if self._planning_latency_count else 0.0
            ),
            coalition_change_count=self._coalition_change_count,
            communication_steps=self.comm_counter.value,
            paper_communication_steps=self.comm_counter.paper_value,
            communication_step_breakdown=self.comm_counter.breakdown,
            hallucination_stats=self.cloud_llm.hallucination_stats,
            experience_reuse_attempts=self.experience_store.reuse_attempts if hasattr(self, "experience_store") and self.experience_store else 0,
            experience_reuse_hits=self.experience_store.reuse_hits if hasattr(self, "experience_store") and self.experience_store else 0,
            # Upgraded fine-grained metrics
            cloud_prompt_tokens=self.cloud_llm.usage.prompt_tokens,
            cloud_completion_tokens=self.cloud_llm.usage.completion_tokens,
            cloud_total_tokens=self.cloud_llm.usage.total_tokens,
            device_prompt_tokens=device_usage.prompt_tokens,
            device_completion_tokens=device_usage.completion_tokens,
            device_total_tokens=device_usage.total_tokens,
            cloud_retry_tokens=self.cloud_llm.usage.retry_tokens,
            device_retry_tokens=device_usage.retry_tokens,
            successful_calls=self.cloud_llm.usage.successful_calls + device_usage.successful_calls,
            failed_calls=self.cloud_llm.usage.failed_calls + device_usage.failed_calls,
            retried_calls=self.cloud_llm.usage.retried_calls + device_usage.retried_calls,
            cache_hits=self.cloud_llm.usage.cache_hits + device_usage.cache_hits,
            local_non_llm_operations=0,
            cloud_network_calls=self.cloud_llm.usage.cloud_network_calls,
            cloud_disk_cache_hits=self.cloud_llm.usage.cloud_disk_cache_hits,
            cloud_failed_attempts=self.cloud_llm.usage.cloud_failed_attempts,
            process_peak_rss_mb=peak_rss,
            process_mean_rss_mb=mean_rss,
            gpu_peak_memory_mb=gpu_peak,
            gpu_mean_memory_mb=gpu_mean,
            device_llm_python_heap_delta_mb=getattr(device_usage, "python_heap_delta_mb", 0.0),
            device_llm_python_heap_delta_by_device=getattr(device_usage, "python_heap_delta_by_device", {}),
            device_llm_tokens_processed_by_device=getattr(device_usage, "tokens_processed_by_device", {}),
            cloud_inference_time_s=self.cloud_llm.usage.cloud_inference_time_s,
            device_inference_time_s=device_usage.device_inference_time_s,
            cqi_evaluation_time_s=cqi_evaluation_time_s,
            coalition_computation_time_s=coalition_computation_time_s,
            architecture_switching_time_s=architecture_switching_time_s,
            state_handoff_time_s=state_handoff_time_s,
            coalition_repair_time_s=0.0,
            consensus_time_s=float(peer_metrics["consensus_latency"]),
            planning_time_s=planning_time_s,
            network_waiting_time_s=0.0,
            simulation_computation_time_s=simulation_computation_time_s,
            planning_latency_p50=plat_p50,
            planning_latency_p95=plat_p95,
            planning_latency_p99=plat_p99,
            planning_latency_min=plat_min,
            planning_latency_max=plat_max,
            planning_latency_std=plat_std,
            cloud_to_device_messages=self.comm_counter.breakdown.get("dispatch", 0),
            device_to_cloud_messages=self.comm_counter.breakdown.get("feedback_sync", 0),
            handoff_messages=int(peer_metrics["handoff_messages"]),
            coalition_messages=int(peer_metrics["coalition_messages"]),
            repair_messages=int(peer_metrics["repair_messages"]),
            cloud_bytes=self.cloud_llm.usage.cloud_bytes,
            peer_bytes=int(peer_metrics["peer_bytes"]),
            broadcast_bytes=int(peer_metrics["broadcast_bytes"]),
            total_bytes=self.cloud_llm.usage.cloud_bytes + int(peer_metrics["peer_bytes"]) + int(peer_metrics["broadcast_bytes"]),
            # Optimization Metrics (Optimizations 1-8)
            prompt_reduction_percent=getattr(self.cloud_llm, "prompt_reduction_percent", 0.0),
            cache_misses=getattr(getattr(self.cloud_llm, "semantic_cache", None), "cache_misses", 0),
            cache_hit_rate=getattr(getattr(self.cloud_llm, "semantic_cache", None), "cache_hit_rate", 0.0),
            saved_cloud_calls=getattr(getattr(self.cloud_llm, "semantic_cache", None), "saved_cloud_calls", 0),
            saved_tokens=getattr(getattr(self.cloud_llm, "semantic_cache", None), "saved_tokens", 0),
            saved_latency=getattr(getattr(self.cloud_llm, "semantic_cache", None), "saved_latency", 0.0),
            local_reasoning_count=self.decentralized.plan_reuse_count,
            cloud_reasoning_count=self._replanning_count,
            consensus_skipped=int(peer_metrics.get("consensus_skipped", 0)),
        )
