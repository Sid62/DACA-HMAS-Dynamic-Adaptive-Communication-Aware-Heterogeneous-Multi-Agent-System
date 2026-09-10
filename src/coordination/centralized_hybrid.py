"""Static Centralized Hybrid architecture (m=0) with domain-level Device LLM dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.control.nmpc import NMPCController
from src.coordination.autohma_structs import (
    DeviceFeedback,
    ExecutionDirective,
    format_feedback_for_cloud,
)
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
    # Delta dispatch: track last dispatched assignment state to suppress
    # redundant Device LLM dispatch calls when assignments haven't changed.
    _last_dispatched_assignments: dict[str, list[str]] = field(default_factory=dict)
    dispatch_skipped_count: int = 0
    # AutoHMA alignment: capture Device LLM dispatch outputs (execution directives)
    _last_dispatch_directives: dict[str, ExecutionDirective] = field(default_factory=dict)
    # AutoHMA alignment: Device-level execution feedback for Cloud planning context
    _last_device_feedbacks: list[DeviceFeedback] = field(default_factory=list)

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
        assignments_map: dict[str, list[str]] | None = None,
    ) -> list[dict]:
        """Slice global coalitions to members managed by one Device LLM domain."""
        agent_to_subtask: dict[str, str] = {}
        if assignments_map:
            for sid, agents in assignments_map.items():
                for aid in agents:
                    agent_to_subtask[aid] = sid

        scoped: list[dict] = []
        for coalition in coalitions:
            members = coalition.get("members", [])
            domain_members = [m for m in members if m in managed_agent_ids]
            if domain_members:
                subtasks = [agent_to_subtask[m] for m in domain_members if m in agent_to_subtask]
                scoped.append({
                    **coalition,
                    "members": domain_members,
                    "subtasks": subtasks,
                    "target_subtask": subtasks[0] if subtasks else None,
                })
        return scoped

    def _dispatch_domains(
        self,
        coalitions: list[dict],
        assignments_map: dict[str, list[str]] | None = None,
    ) -> None:
        """Each domain Device LLM dispatches to its managed agents only.

        AutoHMA alignment: capture dispatch output as ExecutionDirective
        rather than discarding it, establishing the structural link
        Device LLM → Generative Agent → Local Controller.
        """
        self._last_dispatch_directives.clear()
        for domain_id, client in self.device_llms.items():
            managed = set(client.managed_agent_ids)
            domain_coalitions = self._coalitions_for_domain(coalitions, managed, assignments_map)
            if domain_coalitions:
                result = client.dispatch(domain_coalitions, mode=0)
                agent_assigns: dict[str, str] = {}
                if isinstance(result, dict) and "assignments" in result and isinstance(result["assignments"], dict):
                    agent_assigns = {k: str(v) for k, v in result["assignments"].items() if k in managed}
                self._last_dispatch_directives[domain_id] = ExecutionDirective(
                    domain_id=domain_id,
                    dispatch_result=result,
                    coalitions=domain_coalitions,
                    agent_assignments=agent_assigns,
                )

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
                filtered_candidates = [aid for aid in candidate_agents if aid in valid_ids]
                if filtered_candidates and validate_joint_assignment(filtered_candidates, st, fleet, c_task, r_reach):
                    reused[st.subtask_id] = filtered_candidates
                    self.experience_store.reuse_hits += 1
                    print(f"[EXPERIENCE-REUSE] Reused plan for subtask {st.subtask_id}: {filtered_candidates}")

        return reused

    def plan(
        self,
        env: DACAEnv,
        cqi_matrix: np.ndarray | None = None,
        device_feedbacks: list[DeviceFeedback] | None = None,
    ) -> tuple[dict[str, list[str]], list[dict], bool, bool]:
        """Plan and dispatch.

        Args:
            env: The environment.
            cqi_matrix: Communication quality matrix.
            device_feedbacks: AutoHMA Device→Cloud feedback from prior execution.
                Injected into Cloud LLM prompt for self-correction/refinement.
                Does NOT create additional Cloud API calls.

        Returns:
            (assignments, coalitions, cloud_reasoned, dispatch_occurred)
            dispatch_occurred is False only when plan continuity reuses
            assignments identical to the last dispatch — suppressing
            the actual Device LLM dispatch calls, not just the counter.
        """
        fleet = env.fleet
        subtasks = env.subtask_list

        # Plan Continuity Check: If active plan remains valid, continue execution!
        if self.continuity_engine is not None and self.continuity_engine.active_context is not None:
            if self.continuity_engine.can_continue_plan(fleet, subtasks, cqi_matrix):
                print("[PLAN-CONTINUITY] Centralized reusing valid active plan with updated assignments (0 LLM calls)")
                assignments = self.continuity_engine.get_updated_executable_assignments(fleet, subtasks)
                coalitions = self.continuity_engine.active_context.coalitions
                # Delta dispatch: only re-dispatch if assignments actually changed
                if assignments != self._last_dispatched_assignments:
                    self._dispatch_domains(coalitions, assignments)
                    self._last_dispatched_assignments = dict(assignments)
                    print("[DELTA-DISPATCH] Continuity plan has changed assignments — dispatching")
                    return assignments, coalitions, False, True
                else:
                    self.dispatch_skipped_count += 1
                    print("[DELTA-DISPATCH] Assignments unchanged — skipping redundant dispatch")
                    return assignments, coalitions, False, False


        obs = env.get_observation()
        dist_mat = distance_matrix(fleet.agents)
        if cqi_matrix is None:
            cqi_matrix = np.ones(dist_mat.shape)

        reused_assignments = self._try_experience_reuse(env, fleet, subtasks)
        pending_subtasks = [s for s in subtasks if not s.completed and s.subtask_id not in reused_assignments]

        # AutoHMA alignment: format Device feedback for Cloud prompt injection
        feedback_ctx = format_feedback_for_cloud(
            device_feedbacks
        ) if device_feedbacks else None
        if feedback_ctx:
            self._last_device_feedbacks = list(device_feedbacks)

        if reused_assignments and not pending_subtasks:
            print("[EXPERIENCE-REUSE] Centralized reusing validated experience store assignments for all subtasks (0 LLM decomp calls)")
            assignments_map = reused_assignments
        else:
            if self.use_distance_decomp and self.decomposer:
                assignments_map = self.decomposer.decompose(
                    obs["instruction"], fleet, subtasks
                )
            else:
                assignments_map = self.cloud_llm.decompose(
                    obs["instruction"],
                    obs["agents"],
                    obs["subtasks"],
                    execution_feedback_context=feedback_ctx,
                )
            if reused_assignments:
                assignments_map.update(reused_assignments)

        if self.use_coalition_feasibility and self.coalition_formation:
            coalitions = self.coalition_formation.form(
                fleet, subtasks, dist_mat, cqi_matrix
            )
        else:
            coalitions = self.cloud_llm.form_coalitions(
                obs["subtasks"], obs["agents"],
                execution_feedback_context=feedback_ctx,
            )

        cloud_reasoned = not (reused_assignments and not pending_subtasks)

        if self.continuity_engine is not None:
            self.continuity_engine.set_active_plan(assignments_map, coalitions, subtasks, mode=0)

        # New plan always requires dispatch
        self._dispatch_domains(coalitions, assignments_map)
        self._last_dispatched_assignments = dict(assignments_map)
        return assignments_map, coalitions, cloud_reasoned, True

    def extract_executable_assignments(
        self,
        fallback_assignments: dict[str, list[str]],
    ) -> dict[str, str]:
        """Extract agent -> subtask mapping consumed directly from Device LLM ExecutionDirectives.
        
        AutoHMA flow:
        Cloud Plan -> Device LLM Dispatch -> ExecutionDirective -> Agent Execution Path.
        """
        agent_assignments: dict[str, str] = {}
        # 1. Base active subtasks from global assignments
        for sid, agents in fallback_assignments.items():
            if agents:
                agent_assignments[agents[0]] = sid

        # 2. Consume and apply Device LLM ExecutionDirectives
        for directive in self._last_dispatch_directives.values():
            for aid, sid in directive.agent_assignments.items():
                if sid in fallback_assignments:
                    agent_assignments[aid] = sid
            if "assignments" in directive.dispatch_result and isinstance(directive.dispatch_result["assignments"], dict):
                for aid, sid in directive.dispatch_result["assignments"].items():
                    if str(sid) in fallback_assignments:
                        agent_assignments[aid] = str(sid)

        return agent_assignments

    def execute_step(
        self,
        env: DACAEnv,
        assignments: dict[str, list[str]],
    ) -> None:
        """Execute centralized step consuming Device LLM ExecutionDirectives."""
        targets = {
            s.subtask_id: s.target for s in env.subtask_list
        }
        # Consume Device LLM ExecutionDirectives
        agent_assignments = self.extract_executable_assignments(assignments)
        self.nmpc.step(env.fleet, agent_assignments, targets)

        for sid, agent_list in assignments.items():
            if not agent_list:
                continue
            subtask = next((s for s in env.subtask_list if s.subtask_id == sid), None)
            if subtask:
                from src.coordination.constants import COMPLETION_RADIUS_M
                from src.decomposition.distance_feasible_decomp import validate_task_completion
                if validate_task_completion(agent_list, subtask, env.fleet, COMPLETION_RADIUS_M):
                    env.mark_subtask_complete(sid)

