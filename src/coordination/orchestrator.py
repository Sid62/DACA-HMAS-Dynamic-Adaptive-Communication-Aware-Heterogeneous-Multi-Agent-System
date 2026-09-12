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
class RunConfig:
    """Unified run-level experiment configuration passed to all components."""
    name: str = "A5"
    use_optimizations: bool = True

    # Granular optimization switches
    semantic_cache: bool = True
    prompt_compression: bool = True
    summarization: bool = True
    constrained_output: bool = True
    max_completion_tokens: int = 256
    experience_reuse: bool = True
    plan_continuity: bool = True
    delta_transfer: bool = True
    consensus_skip: bool = True
    cache_responses: bool = True

    # Algorithmic and coordination switches
    use_distance_decomp: bool = True
    use_coalition_feasibility: bool = True
    use_cqm: bool = True
    use_acds: bool = True
    use_handoff: bool = True
    use_reallocation: bool = True
    use_hysteresis: bool = True
    static_mode: int | None = None

    # Routing and model options
    use_mock: bool = True
    cloud_provider: str = "groq"
    cloud_model: str = "llama-3.3-70b-versatile"
    device_provider: str = "vllm"
    device_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Context metadata
    scenario: str = ""
    network_profile: str = ""
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_optimizations": self.use_optimizations,
            "optimizations": {
                "use_optimizations": self.use_optimizations,
                "semantic_cache": self.semantic_cache,
                "prompt_compression": self.prompt_compression,
                "summarization": self.summarization,
                "constrained_output": self.constrained_output,
                "max_completion_tokens": self.max_completion_tokens,
                "experience_reuse": self.experience_reuse,
                "plan_continuity": self.plan_continuity,
                "delta_transfer": self.delta_transfer,
                "consensus_skip": self.consensus_skip,
                "cache_responses": self.cache_responses,
            },
            "routing": {
                "use_mock": self.use_mock,
                "cloud_provider": self.cloud_provider,
                "cloud_model": self.cloud_model,
                "device_provider": self.device_provider,
                "device_model": self.device_model,
            },
            "algorithm": {
                "use_distance_decomp": self.use_distance_decomp,
                "use_coalition_feasibility": self.use_coalition_feasibility,
                "use_cqm": self.use_cqm,
                "use_acds": self.use_acds,
                "use_handoff": self.use_handoff,
                "use_reallocation": self.use_reallocation,
                "use_hysteresis": self.use_hysteresis,
                "static_mode": self.static_mode,
            },
            "context": {
                "scenario": self.scenario,
                "network_profile": self.network_profile,
                "seed": self.seed,
            },
        }


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

    # Granular optimization overrides (None = derive from use_optimizations)
    semantic_cache: bool | None = None
    prompt_compression: bool | None = None
    summarization: bool | None = None
    constrained_output: bool | None = None
    max_completion_tokens: int | None = None
    experience_reuse: bool | None = None
    plan_continuity: bool | None = None
    delta_transfer: bool | None = None
    consensus_skip: bool | None = None
    cache_responses: bool | None = None

    # Routing overrides
    use_mock: bool | None = None
    cloud_provider: str | None = None
    cloud_model: str | None = None
    device_provider: str | None = None
    device_model: str | None = None

    def to_run_config(
        self,
        llm_cfg: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        scenario: str = "",
        network_profile: str = "",
        seed: int = 0,
    ) -> RunConfig:
        llm_cfg = llm_cfg or {}
        thresholds = thresholds or {}
        cloud_cfg = llm_cfg.get("cloud", {})
        dev_cfg = llm_cfg.get("device", {})
        th_opts = thresholds.get("optimizations", {})

        opt_master = self.use_optimizations

        def _resolve_opt(override_val: bool | None, yaml_key: str, default: bool = True) -> bool:
            if not opt_master:
                return False
            if override_val is not None:
                return bool(override_val)
            yaml_sec = th_opts.get(yaml_key, {})
            if isinstance(yaml_sec, dict):
                return bool(yaml_sec.get("enabled", default))
            return default

        sem_cache = _resolve_opt(self.semantic_cache, "semantic_cache", True)
        prompt_comp = _resolve_opt(self.prompt_compression, "prompt_compression", True)
        summ = _resolve_opt(self.summarization, "summarization", True)
        const_out = _resolve_opt(self.constrained_output, "constrained_output", True)

        c_skip = (
            False if not opt_master
            else (self.consensus_skip if self.consensus_skip is not None
                  else bool(th_opts.get("consensus", {}).get("skip_when_stable", True)))
        )
        exp_reuse = (
            False if not opt_master
            else (self.experience_reuse if self.experience_reuse is not None
                  else bool(llm_cfg.get("experience_reuse", {}).get("enabled", True)))
        )
        plan_cont = False if not opt_master else (self.plan_continuity if self.plan_continuity is not None else True)
        delta_trans = False if not opt_master else (self.delta_transfer if self.delta_transfer is not None else True)
        disk_cache = False if not opt_master else (self.cache_responses if self.cache_responses is not None else bool(llm_cfg.get("cache_responses", True)))

        mock_flag = self.use_mock if self.use_mock is not None else bool(llm_cfg.get("use_mock", True))
        c_prov = self.cloud_provider or cloud_cfg.get("provider", "groq")
        c_mod = self.cloud_model or cloud_cfg.get("model", "llama-3.3-70b-versatile")
        d_prov = self.device_provider or dev_cfg.get("provider", "vllm")
        d_mod = self.device_model or dev_cfg.get("model", "Qwen/Qwen2.5-3B-Instruct")
        max_tokens = self.max_completion_tokens or th_opts.get("constrained_output", {}).get("max_completion_tokens", 256)

        return RunConfig(
            name=self.name,
            use_optimizations=opt_master,
            semantic_cache=sem_cache,
            prompt_compression=prompt_comp,
            summarization=summ,
            constrained_output=const_out,
            max_completion_tokens=int(max_tokens),
            experience_reuse=exp_reuse,
            plan_continuity=plan_cont,
            delta_transfer=delta_trans,
            consensus_skip=c_skip,
            cache_responses=disk_cache,
            use_distance_decomp=self.use_distance_decomp,
            use_coalition_feasibility=self.use_coalition_feasibility,
            use_cqm=self.use_cqm,
            use_acds=self.use_acds,
            use_handoff=self.use_handoff,
            use_reallocation=self.use_reallocation,
            use_hysteresis=self.use_hysteresis,
            static_mode=self.static_mode,
            use_mock=mock_flag,
            cloud_provider=c_prov,
            cloud_model=c_mod,
            device_provider=d_prov,
            device_model=d_mod,
            scenario=scenario,
            network_profile=network_profile,
            seed=seed,
        )


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
    "A5": DACAConfig(name="A5", use_optimizations=True, experience_reuse=True, cache_responses=True),
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
        from src.env.scenarios import print_scenario_agent_config
        print_scenario_agent_config(self.env.scenario, self.env.fleet)

        llm_cfg = get_llm_config()

        # Build one unified run configuration instance for the experiment
        if hasattr(self.config, "to_run_config"):
            self.run_config = self.config.to_run_config(
                llm_cfg=llm_cfg,
                thresholds=self.thresholds,
                scenario=self.scenario,
                network_profile=self.network_profile,
                seed=self.seed,
            )
        else:
            self.run_config = RunConfig(
                name=getattr(self.config, "name", "A5"),
                use_optimizations=getattr(self.config, "use_optimizations", True),
                scenario=self.scenario,
                network_profile=self.network_profile,
                seed=self.seed,
            )

        self.cloud_llm = CloudLLMClient(llm_cfg, run_config=self.run_config)
        self.cloud_llm.configure_experiment_context(
            scenario=self.scenario,
            architecture=self.config.name,
            network_profile=self.network_profile,
            seed=self.seed,
        )

        self.device_llms: dict[str, DeviceLLMClient] = _build_device_llms_by_type(
            self.env.fleet, llm_cfg
        )
        for d_client in self.device_llms.values():
            d_client.run_config = self.run_config

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
        self.device_decomposer = DistanceFeasibleDecomposer(
            self.device_llm,
            c_task=self.thresholds.get("C_task", 30.0),
            r_reach=self.thresholds.get("R_reach", 100.0),
        )
        self.device_coalition_formation = CoalitionFormation(
            self.device_llm,
            c1=self.thresholds.get("C1", 50.0),
            gamma_min=self.thresholds.get("gamma_min", 0.3),
        )

        # Optimization components wired strictly per unified RunConfig
        if self.run_config.use_optimizations and self.run_config.plan_continuity:
            self.continuity_engine = PlanContinuityEngine(
                validity_threshold=self.thresholds.get("plan_validity_threshold", 0.75),
                r_reach=self.thresholds.get("R_reach", 100.0),
                c_task=self.thresholds.get("C_task", 30.0),
            )
        else:
            self.continuity_engine = None

        if self.run_config.use_optimizations and self.run_config.delta_transfer:
            self.delta_transfer_manager = DeltaStateTransferManager()
        else:
            self.delta_transfer_manager = None

        # Experience store wired strictly per unified RunConfig
        from src.memory.experience_store import SubtaskExperienceStore
        exp_cfg = llm_cfg.get("experience_reuse", {})
        self.experience_store = SubtaskExperienceStore(
            store_path=exp_cfg.get("store_path", "experience_store.json"),
            enabled=bool(self.run_config.use_optimizations and self.run_config.experience_reuse),
        )
        self.experience_store.run_config = self.run_config

        # If optimizations are disabled (e.g. A5_unopt), explicitly guarantee all switches are OFF
        if not self.run_config.use_optimizations:
            if hasattr(self.cloud_llm, "summarizer") and self.cloud_llm.summarizer:
                self.cloud_llm.summarizer.enabled = False
            self.cloud_llm.opt_prompt_compression = False
            self.cloud_llm.opt_constrained_output = False
            if hasattr(self.cloud_llm, "semantic_cache") and self.cloud_llm.semantic_cache:
                self.cloud_llm.semantic_cache.enabled = False
            self.cloud_llm.config["cache_responses"] = False
            for d_client in self.device_llms.values():
                d_client.config["cache_responses"] = False
            self.experience_store.enabled = False
            self.continuity_engine = None
            self.delta_transfer_manager = None

        self.centralized = CentralizedHybridCoordinator(
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            decomposer=self.decomposer,
            coalition_formation=self.coalition_formation,
            use_distance_decomp=self.run_config.use_distance_decomp,
            use_coalition_feasibility=self.run_config.use_coalition_feasibility,
            continuity_engine=self.continuity_engine,
            experience_store=self.experience_store,
            run_config=self.run_config,
        )
        self.decentralized = DecentralizedHybridCoordinator(
            cloud_llm=self.cloud_llm,
            device_llms=self.device_llms,
            peer_manager=self.peer_manager,
            decomposer=self.device_decomposer,
            coalition_formation=self.device_coalition_formation,
            use_distance_decomp=self.run_config.use_distance_decomp,
            use_coalition_feasibility=self.run_config.use_coalition_feasibility,
            continuity_engine=self.continuity_engine,
            experience_store=self.experience_store,
            run_config=self.run_config,
        )
        self.ca_transfer = CATransferManager(
            overlap_delta=self.thresholds.get("ca_overlap_delta", 3)
        )
        self.reallocator = PostSwitchReallocator(
            device_llms=self.device_llms,
            coalition_formation=self.device_coalition_formation,
            peer_manager=self.peer_manager,
        )
        self.decomposer.run_config = self.run_config
        self.coalition_formation.run_config = self.run_config
        self.device_decomposer.run_config = self.run_config
        self.device_coalition_formation.run_config = self.run_config
        self.reallocator.run_config = self.run_config

        self.metrics = MetricsCollector()
        self.comm_counter = CommunicationStepCounter(run_config=self.run_config)
        self._plan_state = PlanState()
        self.metadata: dict[str, Any] = {}

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

    def get_experiment_metadata(self) -> dict[str, Any]:
        """Return the run configuration and experiment metadata."""
        sem_cache = getattr(self.cloud_llm, "semantic_cache", None)
        sem_entries = len(sem_cache.entries) if sem_cache is not None and hasattr(sem_cache, "entries") else 0
        exp_entries = (
            sum(len(v) for v in self.experience_store._entries.values())
            if getattr(self, "experience_store", None) is not None and hasattr(self.experience_store, "_entries")
            else 0
        )
        cache_start = {
            "semantic_cache_entries": sem_entries,
            "semantic_cache_enabled": bool(sem_cache is not None and getattr(sem_cache, "enabled", False)),
            "disk_cache_enabled": bool(self.cloud_llm.config.get("cache_responses", False)),
            "experience_store_entries": exp_entries,
            "experience_store_enabled": bool(
                getattr(self, "experience_store", None) is not None
                and getattr(self.experience_store, "enabled", False)
            ),
        }
        actual_mock = bool(self.cloud_llm.config.get("use_mock", self.run_config.use_mock))
        self.run_config.use_mock = actual_mock
        return {
            "seed": self.seed,
            "provider": self.run_config.cloud_provider,
            "model": self.run_config.cloud_model,
            "mock": actual_mock,
            "use_mock": actual_mock,
            "cloud_provider": self.run_config.cloud_provider,
            "cloud_model": self.run_config.cloud_model,
            "device_provider": self.run_config.device_provider,
            "device_model": self.run_config.device_model,
            "optimizations": self.run_config.to_dict()["optimizations"],
            "cache_start_state": cache_start,
            "scenario": self.scenario,
            "network_profile": self.network_profile,
            "config_name": self.config.name,
        }

    def run(self) -> ExperimentMetrics:
        import inspect
        import psutil
        import os
        proc = psutil.Process(os.getpid())
        baseline_rss = proc.memory_info().rss / (1024 * 1024)
        print("========== RUN STARTED ==========")
        print(inspect.getfile(self.__class__))
        self.metadata = self.get_experiment_metadata()
        start = time.perf_counter()
        self.env.reset()
        self.comm_counter.reset()
        self.cloud_llm.usage.reset()
        for d_client in self.device_llms.values():
            d_client.usage.reset()
        self.peer_manager.reset_metrics()

        if hasattr(self, "experience_store") and self.experience_store is not None:
            self.experience_store.reset_run_metrics()
        self.reallocator.reset_metrics()
        self._plan_state = PlanState()
        self.decentralized.plan_reuse_count = 0
        self.centralized.dispatch_skipped_count = 0
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

        assignments: dict = {}
        coalitions: list = []
        tfr_history: list[float] = []
        cfr_history: list[float] = []
        prev_mode = self.acds.mode

        for step in range(self.max_steps):
            t_sim_step_start = time.perf_counter()
            process_rss_samples.append((proc.memory_info().rss / (1024 * 1024)) - baseline_rss)
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
            c1_thresh = self.thresholds.get("C1", 50.0)
            if self.config.use_cqm and hasattr(self.env.network, "simulate_link"):
                for i in range(fleet.n_agents):
                    for j in range(fleet.n_agents):
                        if i != j and dist_mat[i, j] <= c1_thresh:
                            link_state = self.env.network.simulate_link(
                                step, i, j, distance=float(dist_mat[i, j])
                            )
                            self.cqm.update_link(i, j, link_state)

            cqi_matrix = self.cqm.update_pairwise(
                dist_mat, c1_thresh, network=self.env.network, step=step
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
                t_snap1_start = time.perf_counter()
                snap_before = capture_snapshot(
                    fleet, self.env.subtask_list, coalitions,
                    step, prev_mode, mode,
                    shared_plans=self.decentralized.shared_plans,
                    device_llms=self.device_llms,
                    pending_messages=self.peer_manager.pending_messages_all(),
                )
                t_snap1_end = time.perf_counter()
                snapshot_capture_time_s += (t_snap1_end - t_snap1_start)

                t_restore_start = time.perf_counter()
                restore_snapshot(fleet, snap_before)
                restore_distributed_state(
                    snap_before, self.device_llms, self.peer_manager, self.decentralized
                )
                t_restore_end = time.perf_counter()
                state_restore_time_s += (t_restore_end - t_restore_start)

                t_snap2_start = time.perf_counter()
                snap_after = capture_snapshot(
                    fleet, self.env.subtask_list, coalitions,
                    step, prev_mode, mode,
                    shared_plans=self.decentralized.shared_plans,
                    device_llms=self.device_llms,
                    pending_messages=self.peer_manager.pending_messages_all(),
                )
                t_snap2_end = time.perf_counter()
                snapshot_capture_time_s += (t_snap2_end - t_snap2_start)

                t_verify_start = time.perf_counter()
                is_preserved = verify_task_preservation(snap_before, snap_after)
                t_verify_end = time.perf_counter()
                state_verification_time_s += (t_verify_end - t_verify_start)

                if not is_preserved:
                    print("[WARNING] State handoff verification failed: runtime state not preserved across architecture switch.")

                t_transfer_start = time.perf_counter()
                self.ca_transfer.on_mode_change(mode)
                t_transfer_end = time.perf_counter()
                coalition_transfer_time_s += (t_transfer_end - t_transfer_start)

                self.comm_counter.record_handoff_reallocation(1, "mode_handoff_snapshot_transfer")

                state_handoff_time_s += (
                    (t_snap1_end - t_snap1_start)
                    + (t_restore_end - t_restore_start)
                    + (t_snap2_end - t_snap2_start)
                    + (t_verify_end - t_verify_start)
                    + (t_transfer_end - t_transfer_start)
                )

                if self.config.use_reallocation and self.reallocator.should_trigger(
                    True,
                    coalitions,
                    fleet,
                    dist_mat,
                    cqi_matrix,
                    subtasks=self.env.subtask_list,
                    assignments=assignments,
                    c1=self.thresholds.get("C1", 50.0),
                    gamma_min=self.thresholds.get("gamma_min", 0.3),
                    c_task=self.thresholds.get("C_task", 30.0),
                    r_reach=self.thresholds.get("R_reach", 100.0),
                ):
                    t_realloc_start = time.perf_counter()
                    coalitions = self.reallocator.reallocate(
                        fleet, self.env.subtask_list, coalitions, dist_mat, cqi_matrix
                    )
                    t_realloc_end = time.perf_counter()
                    realloc_dur = t_realloc_end - t_realloc_start
                    coalition_computation_time_s += realloc_dur
                    reallocation_time_s += realloc_dur
                    self.comm_counter.record_handoff_reallocation(1, "post_switch_coalition_reallocation")
            if mode != prev_mode:
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
                    assignments, coalitions, cloud_reasoned, dispatch_occurred = self.centralized.plan(self.env, cqi_matrix)
                    if cloud_reasoned:
                        self.comm_counter.record_global_planning(1, "centralized_global_planning")
                    if dispatch_occurred:
                        self.comm_counter.record_dispatch(1, "centralized_domain_dispatch")
                    print(">>>> USING CENTRALIZED")
                else:
                    print("\nEntering decentralized planner\n")
                    assignments, coalitions, cloud_reasoned = self.decentralized.plan(self.env, cqi_matrix)
                    if cloud_reasoned:
                        self.comm_counter.record_local_coordination(1, "decentralized_leader_planning")
                        self.comm_counter.record_peer_consensus(1, "decentralized_peer_review_consensus")
                        self.comm_counter.record_feedback_sync(1, "decentralized_state_sync")
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
                if self.continuity_engine is not None and self.continuity_engine.active_context is not None:
                    # Authoritative execution state from continuity engine
                    assignments = dict(self.continuity_engine.active_context.assignments)
                    completed_sids = {s.subtask_id for s in self.env.subtask_list if s.completed} | self.continuity_engine.active_context.completed_subtask_ids
                    assignments = {sid: aids for sid, aids in assignments.items() if sid not in completed_sids}
                    # Capacity audit: check if any duplicate agents or unassigned incomplete tasks exist
                    seen_aids = set()
                    dup_found = False
                    for sid, aids in assignments.items():
                        for aid in aids:
                            if aid in seen_aids:
                                dup_found = True
                                break
                            seen_aids.add(aid)
                        if dup_found:
                            break
                    if dup_found:
                        print(f"[REPLAN] Capacity violation in cached plan -- refreshing executable assignments")
                        assignments = self.continuity_engine.get_updated_executable_assignments(fleet, self.env.subtask_list)
                    if mode == 0 and assignments != self.centralized._last_dispatched_assignments:
                        self.centralized._dispatch_domains(coalitions)
                        self.centralized._last_dispatched_assignments = dict(assignments)

            print(f"\n[ASSIGN] Step={step}")
            for sid, agents in list(assignments.items()):
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
            for sid, agents in list(assignments.items()):
                for aid in agents:
                    if aid in agent_assignments:
                        prev_sid = agent_assignments[aid]
                        prev_st = next((s for s in self.env.subtask_list if s.subtask_id == prev_sid), None)
                        curr_st = next((s for s in self.env.subtask_list if s.subtask_id == sid), None)
                        d_prev = dist(fleet.get_agent(aid).position, prev_st.target) if prev_st else float("inf")
                        d_curr = dist(fleet.get_agent(aid).position, curr_st.target) if curr_st else float("inf")
                        if d_curr < d_prev:
                            agent_assignments[aid] = sid
                    else:
                        agent_assignments[aid] = sid

            t_sim_body = time.perf_counter()
            self.ca_transfer.step(self.env.fleet, mode, agent_assignments, targets)

            for sid, agent_list in list(assignments.items()):
                if not agent_list:
                    continue
                subtask = next(
                    (s for s in self.env.subtask_list if s.subtask_id == sid), None
                )
                if subtask:
                    from src.decomposition.distance_feasible_decomp import validate_task_completion
                    from src.coordination.constants import COMPLETION_RADIUS_M
                    if validate_task_completion(agent_list, subtask, fleet, COMPLETION_RADIUS_M):
                        was_completed = subtask.completed
                        self.env.mark_subtask_complete(sid)
                        if self.continuity_engine is not None:
                            self.continuity_engine.mark_subtask_completed(sid)
                        assignments.pop(sid, None)
                        if not was_completed and hasattr(self, "experience_store") and self.experience_store is not None and self.experience_store.enabled:
                            from src.memory.experience_store import compute_signature
                            agent_types = [a.agent_type.value for a in fleet.agents]
                            first_agent = fleet.get_agent(agent_list[0])
                            d_lead = dist(first_agent.position, subtask.target)
                            sig = compute_signature(self.scenario, subtask.required_skills, agent_types, d_lead)
                            self.experience_store.record(
                                signature=sig,
                                plan={sid: list(agent_list)},
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
                    f"Completed={self.env.success_rate() * 100.0:.2f}% "
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
            reallocation_trigger_count=self.reallocator.reallocation_trigger_count,
            reallocation_skip_count=self.reallocator.reallocation_skip_count,
            reallocation_reasons=self.reallocator.reallocation_reasons,
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
            dispatch_skipped_rounds=self.centralized.dispatch_skipped_count,
            metadata=self.metadata,
            measured_cloud_prompt_tokens=self.cloud_llm.usage.measured_prompt_tokens,
            measured_cloud_completion_tokens=self.cloud_llm.usage.measured_completion_tokens,
            measured_cloud_total_tokens=self.cloud_llm.usage.measured_total_tokens,
            estimated_cloud_prompt_tokens=self.cloud_llm.usage.estimated_prompt_tokens,
            estimated_cloud_completion_tokens=self.cloud_llm.usage.estimated_completion_tokens,
            estimated_cloud_total_tokens=self.cloud_llm.usage.estimated_total_tokens,
            measured_device_prompt_tokens=device_usage.measured_prompt_tokens,
            measured_device_completion_tokens=device_usage.measured_completion_tokens,
            measured_device_total_tokens=device_usage.measured_total_tokens,
            estimated_device_prompt_tokens=device_usage.estimated_prompt_tokens,
            estimated_device_completion_tokens=device_usage.estimated_completion_tokens,
            estimated_device_total_tokens=device_usage.estimated_total_tokens,
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
            semantic_cache_hits=getattr(
                getattr(self.cloud_llm, "semantic_cache", None), "cache_hits", 0
            ),
            cloud_call_attribution=self.cloud_llm.usage.call_attribution(),
            logical_llm_requests=self.cloud_llm.usage.logical_requests + getattr(device_usage, "logical_requests", 0),
            device_inference_calls=getattr(device_usage, "device_inference_calls", device_usage.device_api_calls),
            process_peak_rss_mb=peak_rss,
            process_mean_rss_mb=mean_rss,
            gpu_peak_memory_mb=gpu_peak,
            gpu_mean_memory_mb=gpu_mean,
            device_llm_memory_mb=getattr(device_usage, "device_llm_memory_mb", {}),
            device_llm_memory_peak_mb=getattr(device_usage, "device_llm_memory_peak_mb", {}),
            device_llm_heap_delta_mb=getattr(device_usage, "device_llm_heap_delta_mb", {}),
            inference_backend_memory_mb=None,
            device_llm_python_heap_delta_mb=getattr(device_usage, "python_heap_delta_mb", 0.0),
            device_llm_python_heap_delta_by_device=getattr(device_usage, "python_heap_delta_by_device", {}),
            device_llm_tokens_processed_by_device=getattr(device_usage, "tokens_processed_by_device", {}),
            cloud_inference_time_s=self.cloud_llm.usage.cloud_inference_time_s,
            device_inference_time_s=device_usage.device_inference_time_s,
            cqi_evaluation_time_s=cqi_evaluation_time_s,
            coalition_computation_time_s=coalition_computation_time_s,
            architecture_switching_time_s=architecture_switching_time_s,
            snapshot_capture_time_s=snapshot_capture_time_s,
            state_restore_time_s=state_restore_time_s,
            state_verification_time_s=state_verification_time_s,
            coalition_transfer_time_s=coalition_transfer_time_s,
            reallocation_time_s=reallocation_time_s,
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
