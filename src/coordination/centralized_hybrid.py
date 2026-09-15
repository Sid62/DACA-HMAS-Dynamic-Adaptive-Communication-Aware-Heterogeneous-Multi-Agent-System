"""Static Centralized Hybrid architecture (m=0) with domain-level Device LLM dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.control.nmpc import NMPCController
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer
from src.env.agents import distance_matrix
from src.env.daca_env import DACAEnv
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class CentralizedHybridCoordinator:
    """
    Centralized hybrid coordinator (m=0).

    Cloud LLM performs decomposition, coalition formation, and global planning.
    Each agent-type Device LLM receives the global coalition plan and dispatches
    only to its managed agents. No peer communication or consensus.
    """

    cloud_llm: CloudLLMClient
    device_llm: DeviceLLMClient | None = None
    device_llms: dict[str, DeviceLLMClient] = field(default_factory=dict)
    decomposer: DistanceFeasibleDecomposer | None = None
    coalition_formation: CoalitionFormation | None = None
    use_distance_decomp: bool = False
    use_coalition_feasibility: bool = False
    continuity_engine: Any | None = None
    plan_repairer: Any | None = None
    experience_store: Any | None = None
    run_config: Any | None = None
    nmpc: NMPCController = field(default_factory=NMPCController)
    # Delta dispatch: track last dispatched assignment state to suppress
    # redundant Device LLM dispatch calls when assignments haven't changed.
    _last_dispatched_assignments: dict[str, list[str]] = field(default_factory=dict)
    dispatch_skipped_count: int = 0

    def __post_init__(self) -> None:
        if self.device_llms:
            if self.device_llm is None:
                self.device_llm = next(iter(self.device_llms.values()), None)
        elif self.device_llm is not None:
            self.device_llms = {self.node_id if hasattr(self, 'node_id') else 'device_0': self.device_llm}

    @staticmethod
    def _coalitions_for_domain(
        coalitions: list[dict],
        managed_agent_ids: set[str],
    ) -> list[dict]:
        """Slice global coalitions to members managed by one Device LLM domain."""
        scoped: list[dict] = []
        for coalition in coalitions:
            members = coalition.get("members", [])
            domain_members = [m for m in members if m in managed_agent_ids]
            if domain_members:
                scoped.append({**coalition, "members": domain_members})
        return scoped

    def _dispatch_domains(self, coalitions: list[dict]) -> bool:
        """Each domain Device LLM dispatches to its managed agents only.

        Returns:
            bool: True if at least one domain Device LLM dispatch was performed.
        """
        dispatched = False
        if self.device_llms:
            for client in self.device_llms.values():
                managed = set(client.managed_agent_ids)
                domain_coalitions = self._coalitions_for_domain(coalitions, managed)
                if domain_coalitions:
                    client.dispatch(domain_coalitions, mode=0)
                    dispatched = True
        elif coalitions:
            dispatched = any(bool(c.get("members")) for c in coalitions)
        return dispatched

    def _try_experience_reuse(
        self,
        env: DACAEnv,
        fleet: AgentFleet,
        subtasks: list[Subtask],
    ) -> dict[str, list[str]]:
        if self.experience_store is None or not self.experience_store.enabled:
            return {}

        from src.memory.experience_store import compute_signature
        from src.decomposition.distance_feasible_decomp import validate_joint_assignment
        from src.env.agents import dist

        reused: dict[str, list[str]] = {}
        c_task = 30.0
        r_reach = 100.0
        if self.decomposer is not None:
            c_task = getattr(self.decomposer, "c_task", 30.0)
            r_reach = getattr(self.decomposer, "r_reach", 100.0)

        scenario = getattr(env, "scenario_name", "logistics")
        agent_types = [a.agent_type.value for a in fleet.agents]

        reused_agents: set[str] = set()
        for st in subtasks:
            if st.completed:
                continue
            if not fleet.agents:
                continue
            closest = min(fleet.agents, key=lambda a: dist(a.position, st.target))
            d_lead = dist(closest.position, st.target)
            sig = compute_signature(scenario, st.required_skills, agent_types, d_lead)

            self.experience_store.reuse_attempts += 1
            stored_plan = self.experience_store.lookup(sig)
            if stored_plan:
                candidate_agents = stored_plan.get(st.subtask_id, [])
                if isinstance(candidate_agents, str):
                    candidate_agents = [candidate_agents]
                valid_ids = {a.agent_id for a in fleet.agents}
                if (
                    candidate_agents
                    and all(aid in valid_ids for aid in candidate_agents)
                    and not any(aid in reused_agents for aid in candidate_agents)
                    and validate_joint_assignment(candidate_agents, st, fleet, c_task, r_reach)
                ):
                    reused[st.subtask_id] = candidate_agents
                    reused_agents.update(candidate_agents)
                    self.experience_store.reuse_hits += 1
                    print(f"[EXPERIENCE-REUSE] Reused plan for subtask {st.subtask_id}: {candidate_agents}")

        return reused

    def plan(
        self,
        env: DACAEnv,
        cqi_matrix: np.ndarray | None = None,
        force_replan: bool = False,
        replan_reason: str | None = None,
    ) -> tuple[dict[str, list[str]], list[dict], bool, bool]:
        """Plan and dispatch.

        Args:
            env: Current DACA simulation environment.
            cqi_matrix: Current inter-agent communication quality matrix.
            force_replan: If True, indicates an approved global replan decision
                          from should_replan() that must not be vetoed by continuity.
            replan_reason: Triggering reason string for selective replanning.

        Returns:
            (assignments, coalitions, cloud_reasoned, dispatch_occurred)
        """
        fleet = env.fleet
        subtasks = env.subtask_list
        # Plan Continuity Check: ONLY perform when force_replan is False
        # (standalone/untriggered call). If force_replan is True, an authoritative
        # replan decision was already confirmed and must reach the Cloud planner.
        if not force_replan:
            if self.continuity_engine is not None and self.continuity_engine.active_context is not None:
                if self.continuity_engine.can_continue_plan(fleet, subtasks, cqi_matrix):
                    print("[PLAN-CONTINUITY] Centralized reusing valid active plan with updated assignments (0 LLM calls)")
                    assignments = self.continuity_engine.get_updated_executable_assignments(fleet, subtasks)
                    coalitions = self.continuity_engine.active_context.coalitions
                    if assignments != self._last_dispatched_assignments:
                        dispatch_occurred = self._dispatch_domains(coalitions)
                        self._last_dispatched_assignments = dict(assignments)
                        print("[DELTA-DISPATCH] Continuity plan has changed assignments — dispatching")
                        return assignments, coalitions, False, dispatch_occurred
                    else:
                        self.dispatch_skipped_count += 1
                        print("[DELTA-DISPATCH] Assignments unchanged — skipping redundant dispatch")
                        return assignments, coalitions, False, False

        obs = env.get_observation() if hasattr(env, "get_observation") else {
            "instruction": "Decompose mission into task assignments.",
            "agents": fleet.to_dict_list(),
            "subtasks": [{"id": s.subtask_id, "skills": s.required_skills, "target": [s.target.x, s.target.y]} for s in subtasks],
        }
        dist_mat = distance_matrix(fleet.agents)
        if cqi_matrix is None:
            cqi_matrix = np.ones(dist_mat.shape)

        # --- Event-Driven Selective Cloud Replanning ---
        # Determine whether decomposition, coalition formation, or both require Cloud reasoning.
        effective_reason = str(replan_reason or "").lower()
        is_initial = (
            "mission_initialization" in effective_reason
            or (self.continuity_engine is not None and self.continuity_engine.active_context is None)
            or (not self._last_dispatched_assignments and not getattr(self.cloud_llm, "_last_assignments", {}))
        )
        is_reassign_only = (
            "task_completed_needs_reassignment" in effective_reason
            or "new_subtask_discovered" in effective_reason
        )
        is_comm_only = (
            "cqi_changed_significantly" in effective_reason
            or "packet_loss_crossed_threshold" in effective_reason
            or "latency_crossed_threshold" in effective_reason
            or "coalition_invalidated" in effective_reason
            or "coalition_membership_changed" in effective_reason
        )

        reused_assignments = self._try_experience_reuse(env, fleet, subtasks)
        pending_subtasks = [s for s in subtasks if not s.completed and s.subtask_id not in reused_assignments]

        need_decompose = is_initial or is_reassign_only or (not is_comm_only)
        need_coalitions = is_initial or is_comm_only or (not is_reassign_only)

        # 1. Task Decomposition / Reassignment
        if need_decompose:
            if reused_assignments and not pending_subtasks:
                print("[EXPERIENCE-REUSE] Centralized reusing validated experience store assignments for all subtasks (0 LLM decomp calls)")
                assignments_map = reused_assignments
            else:
                target_subtasks = pending_subtasks if not is_initial else subtasks
                if self.use_distance_decomp and self.decomposer:
                    assignments_map = self.decomposer.decompose(
                        obs["instruction"], fleet, target_subtasks
                    )
                else:
                    assignments_map = self.cloud_llm.decompose(
                        obs["instruction"],
                        obs["agents"],
                        [{"id": s.subtask_id, "skills": s.required_skills, "target": [s.target.x, s.target.y]} for s in target_subtasks],
                    )
                if reused_assignments:
                    assignments_map.update(reused_assignments)
        else:
            # Reuse valid existing assignments (Optimization F)
            print("[SELECTIVE-REPLAN] Preserving valid task assignments, refreshing coalitions only")
            if self.continuity_engine is not None and self.continuity_engine.active_context is not None:
                assignments_map = self.continuity_engine.get_updated_executable_assignments(fleet, subtasks)
            else:
                assignments_map = dict(self._last_dispatched_assignments)

        # 2. Coalition Formation / Refinement
        if need_coalitions:
            if self.use_coalition_feasibility and self.coalition_formation:
                coalitions = self.coalition_formation.form(
                    fleet, subtasks, dist_mat, cqi_matrix
                )
            else:
                coalitions = self.cloud_llm.form_coalitions(
                    obs["subtasks"], obs["agents"],
                    distance_matrix=dist_mat.tolist(),
                    cqi_matrix=cqi_matrix.tolist(),
                )
        else:
            # Reuse valid existing coalitions (Optimization F)
            print("[SELECTIVE-REPLAN] Preserving valid coalition structure, refreshing assignments only")
            if self.continuity_engine is not None and self.continuity_engine.active_context is not None:
                coalitions = list(self.continuity_engine.active_context.coalitions)
            else:
                coalitions = self.cloud_llm._last_coalitions or []

        cloud_reasoned = bool(
            (need_decompose and not (reused_assignments and not pending_subtasks))
            or need_coalitions
        )

        if self.continuity_engine is not None:
            self.continuity_engine.set_active_plan(assignments_map, coalitions, subtasks, mode=0)

        # Dispatch updated global plan to domain Device LLMs
        dispatch_occurred = self._dispatch_domains(coalitions)
        self._last_dispatched_assignments = dict(assignments_map)
        return assignments_map, coalitions, cloud_reasoned, dispatch_occurred

    def execute_step(
        self,
        env: DACAEnv,
        assignments: dict[str, list[str]],
    ) -> None:
        targets = {
            s.subtask_id: s.target for s in env.subtask_list
        }
        agent_assignments = {}
        for sid, agents in list(assignments.items()):
            for aid in agents:
                agent_assignments[aid] = sid
        self.nmpc.step(env.fleet, agent_assignments, targets)

        for sid, agent_list in list(assignments.items()):
            if not agent_list:
                continue
            subtask = next((s for s in env.subtask_list if s.subtask_id == sid), None)
            if subtask:
                from src.coordination.constants import get_completion_radius
                from src.decomposition.distance_feasible_decomp import validate_task_completion
                radius = get_completion_radius(getattr(env, "thresholds", None), getattr(env, "scenario_name", None))
                if validate_task_completion(agent_list, subtask, env.fleet, radius):
                    env.mark_subtask_complete(sid)
                    if self.continuity_engine is not None:
                        self.continuity_engine.mark_subtask_completed(sid)
                    assignments.pop(sid, None)
