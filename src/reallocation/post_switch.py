"""Post-switch coalition reallocation with domain-level leader-peer consensus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.feasibility import (
    build_psi_matrix,
    validate_coalition_members,
)
from src.coalition.formation import CoalitionFormation
from src.communication.models import (
    discover_agent_type_domains,
    dominant_domain_for_coalition,
    domains_in_coalition,
)
from src.communication.peer_manager import PeerCommunicationManager
from src.decomposition.distance_feasible_decomp import validate_joint_assignment
from src.env.agents import AgentFleet, AgentState, dist
from src.env.scenarios import Subtask
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class PostSwitchReallocator:
    device_llms: dict[str, DeviceLLMClient] = field(default_factory=dict)
    coalition_formation: CoalitionFormation | None = None
    peer_manager: PeerCommunicationManager | None = None

    reallocation_trigger_count: int = 0
    reallocation_skip_count: int = 0
    reallocation_reasons: dict[str, int] = field(default_factory=dict)
    last_decision_reason: str = ""

    def reset_metrics(self) -> None:
        """Reset diagnostic metrics for a fresh simulation run."""
        self.reallocation_trigger_count = 0
        self.reallocation_skip_count = 0
        self.reallocation_reasons.clear()
        self.last_decision_reason = ""

    def _record_trigger(self, reason: str, mode_changed: bool) -> bool:
        self.reallocation_trigger_count += 1
        self.reallocation_reasons[reason] = self.reallocation_reasons.get(reason, 0) + 1
        self.last_decision_reason = reason
        print(
            f"[REALLOC_DECISION] mode_changed={mode_changed} -> REALLOCATION REQUIRED (reason: {reason})"
        )
        return True

    def _record_skip(self, reason: str, mode_changed: bool) -> bool:
        self.reallocation_skip_count += 1
        self.last_decision_reason = reason
        print(
            f"[REALLOC_DECISION] mode_changed={mode_changed} -> NO REALLOCATION REQUIRED (reason: {reason})"
        )
        return False

    @property
    def device_llm(self) -> DeviceLLMClient:
        """Backward-compatible single Device LLM accessor."""
        if not self.device_llms:
            return DeviceLLMClient(node_id="device_0")
        return next(iter(self.device_llms.values()))

    def _domain_client(
        self, domain: str, fleet: AgentFleet
    ) -> DeviceLLMClient | None:
        """Resolve Device LLM client for a domain (domain-keyed or legacy per-agent)."""
        client = self.device_llms.get(domain)
        if client is not None:
            return client
        for agent_id in discover_agent_type_domains(fleet).get(domain, []):
            legacy = self.device_llms.get(agent_id)
            if legacy is not None:
                return legacy
        return None

    def should_trigger(
        self,
        mode_changed: bool,
        coalitions: list[dict],
        fleet: AgentFleet,
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
        subtasks: list[Subtask] | None = None,
        assignments: dict[str, list[str]] | None = None,
        c1: float = 50.0,
        gamma_min: float = 0.3,
        c_task: float = 30.0,
        r_reach: float = 100.0,
    ) -> bool:
        """Evaluate whether post-switch reallocation is actually required.

        Reallocation is triggered ONLY when the architecture transition has made
        the current coalition or task assignment invalid, infeasible, or insufficient
        for completing remaining mission work.
        """
        if not mode_changed:
            return False

        # 1. Check remaining uncompleted subtasks
        remaining = (
            [s for s in subtasks if not s.completed]
            if subtasks is not None
            else None
        )
        if remaining is not None and len(remaining) == 0:
            return self._record_skip("state_valid_after_switch", mode_changed)

        # 2. Validate Coalition Feasibility
        if not coalitions:
            return self._record_trigger("coalition_infeasible", mode_changed)

        id_to_idx = {a.agent_id: i for i, a in enumerate(fleet.agents)}
        psi = build_psi_matrix(distance_matrix, cqi_matrix, c1)

        for c in coalitions:
            members = c.get("members", [])
            if not members:
                return self._record_trigger("coalition_infeasible", mode_changed)

            for mid in members:
                if mid not in id_to_idx:
                    return self._record_trigger("agent_unavailable", mode_changed)

            if not validate_coalition_members(members, id_to_idx, psi, gamma_min):
                return self._record_trigger("coalition_infeasible", mode_changed)

        # 3. Validate Assignment Validity, Capability Satisfaction & Remaining-Task Coverage
        if remaining is not None and assignments is not None:
            if not assignments and remaining:
                return self._record_trigger("remaining_task_uncovered", mode_changed)

            agent_map = {a.agent_id: a for a in fleet.agents}
            agent_skills = {a.agent_id: set(a.skills) for a in fleet.agents}

            for s in remaining:
                assigned = assignments.get(s.subtask_id, [])
                if not assigned:
                    return self._record_trigger(
                        "remaining_task_uncovered", mode_changed
                    )

                for aid in assigned:
                    if aid not in agent_map:
                        return self._record_trigger(
                            "agent_unavailable", mode_changed
                        )

                # Capability / Skill coverage
                combined_skills: set[str] = set()
                for aid in assigned:
                    combined_skills.update(agent_skills.get(aid, set()))
                if not set(s.required_skills).issubset(combined_skills):
                    return self._record_trigger(
                        "capability_violation", mode_changed
                    )

                # Target reachability feasibility for assigned agents in transit
                for aid in assigned:
                    if dist(agent_map[aid].position, s.target) > r_reach:
                        return self._record_trigger(
                            "assignment_invalid", mode_changed
                        )

        # State is fully valid after switch -- continue without replanning
        return self._record_skip("state_valid_after_switch", mode_changed)


    def _select_leader_domain(self, members: list[str], fleet: AgentFleet) -> str:
        return dominant_domain_for_coalition(members, fleet)

    def _coalition_agent_dicts(
        self, members: list[str], fleet: AgentFleet
    ) -> list[dict[str, Any]]:
        member_set = set(members)
        return [
            a
            for a in fleet.to_dict_list()
            if str(a.get("agent_id", a.get("id", ""))) in member_set
        ]

    def _distributed_realloc_coalition(
        self,
        coalition: dict,
        remaining: list[dict],
        fleet: AgentFleet,
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> dict:
        """Leader domain proposes reallocation; peer domains validate; consensus updates."""
        pm = self.peer_manager
        members: list[str] = coalition.get("members", [])
        if not members:
            return coalition

        leader_domain = self._select_leader_domain(members, fleet)
        leader_client = self._domain_client(leader_domain, fleet)
        if leader_client is None or pm is None:
            return coalition

        domains = domains_in_coalition(members, fleet)
        peer_domains = [d for d in domains if d != leader_domain]

        t0 = time.perf_counter()
        proposal = leader_client.reallocate_remaining(
            remaining,
            self._coalition_agent_dicts(members, fleet),
            distance_matrix.tolist(),
            cqi_matrix.tolist(),
            scope_to_managed=False,
        )

        if not proposal:
            return coalition

        pm.broadcast(
            leader_domain,
            "realloc_proposal",
            {
                "coalition_id": coalition.get("coalition_id"),
                "proposal": proposal,
                "leader_domain": leader_domain,
            },
        )

        approvals = 0
        required = max(len(peer_domains), 1)
        for peer_domain in peer_domains:
            peer_client = self._domain_client(peer_domain, fleet)
            if peer_client is None:
                continue
            msgs = pm.receive_messages(peer_domain)
            for msg in msgs:
                if msg.message_type == "realloc_proposal":
                    review = peer_client.review_peer_plan(
                        leader_domain,
                        {"proposal": msg.payload.get("proposal", [])},
                        0,
                    )
                    approved = review.get("approved", True)
                    pm.send_message(
                        peer_domain,
                        leader_domain,
                        "realloc_validation",
                        {"approved": approved, "review": review},
                    )
                    if approved:
                        approvals += 1

        pm.record_consensus_round(time.perf_counter() - t0)

        if approvals >= required // 2 or not peer_domains:
            new_members = coalition.get("members", [])
            if proposal and isinstance(proposal, list) and proposal[0].get("members"):
                new_members = proposal[0]["members"]
            elif proposal and isinstance(proposal, dict) and proposal.get("members"):
                new_members = proposal["members"]
            elif proposal and isinstance(proposal, dict) and proposal.get("coalitions"):
                c_list = proposal.get("coalitions", [])
                if c_list and isinstance(c_list, list) and c_list[0].get("members"):
                    new_members = c_list[0]["members"]
            return {
                "coalition_id": coalition.get("coalition_id"),
                "members": new_members,
            }
        return coalition

    def reallocate(
        self,
        fleet: AgentFleet,
        subtasks: list[Subtask],
        coalitions: list[dict],
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> list[dict]:
        """Reallocate remaining subtasks via distributed domain leader-peer consensus."""
        remaining = [
            {
                "id": s.subtask_id,
                "target": [s.target.x, s.target.y],
                "skills": s.required_skills,
            }
            for s in subtasks
            if not s.completed
        ]
        if not remaining:
            return coalitions

        if self.device_llms and self.peer_manager and len(self.device_llms) > 1:
            self.peer_manager.record_distributed_replanning()
            updated = []
            for coalition in coalitions:
                updated.append(
                    self._distributed_realloc_coalition(
                        coalition, remaining, fleet, distance_matrix, cqi_matrix
                    )
                )
            if updated:
                return updated

        new_coalitions = self.device_llm.reallocate_remaining(
            remaining,
            fleet.to_dict_list(),
            distance_matrix.tolist(),
            cqi_matrix.tolist(),
            scope_to_managed=False,
        )
        if not new_coalitions:
            new_coalitions = self._algorithmic_reallocate(
                subtasks, fleet, coalitions, distance_matrix, cqi_matrix
            )
        if not new_coalitions and self.coalition_formation:
            new_coalitions = self.coalition_formation.form(
                fleet, subtasks, distance_matrix, cqi_matrix
            )
        return new_coalitions or coalitions

    def _algorithmic_reallocate(
        self,
        subtasks: list[Subtask],
        fleet: AgentFleet,
        coalitions: list[dict],
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> list[dict]:
        """Multi-factor utility-based algorithmic reallocation solver for remaining subtasks."""
        remaining_tasks = [s for s in subtasks if not s.completed]
        if not remaining_tasks:
            return coalitions

        id_to_idx = {a.agent_id: i for i, a in enumerate(fleet.agents)}
        workload: dict[str, int] = {}
        for c in coalitions:
            for m in c.get("members", []):
                workload[m] = workload.get(m, 0)

        updated_coalitions = [dict(c) for c in coalitions]
        if not updated_coalitions:
            updated_coalitions = [
                {"coalition_id": i, "members": [a.agent_id]}
                for i, a in enumerate(fleet.agents)
            ]

        for st in remaining_tasks:
            req = set(st.required_skills)
            full_cands = [a for a in fleet.agents if req.issubset(set(a.skills))]
            best_team = None
            best_utility = float("inf")
            n_tasks = max(len(subtasks), 1)
            r_max = 100.0
            w_d, w_w, w_b, w_c = 0.40, 0.40, 0.10, 0.10

            from src.decomposition.distance_feasible_decomp import domain_skill_affinity

            def _agent_score(agent: AgentState) -> float:
                v = float(getattr(fleet.kinematics.get(agent.agent_type.value, None), "max_speed", 1.0)) if hasattr(fleet, "kinematics") and fleet.kinematics else 1.0
                d = dist(agent.position, st.target)
                eta = d / max(v, 1e-9)
                idx = id_to_idx.get(agent.agent_id, 0)
                mean_cqi = float(np.mean(cqi_matrix[idx, :])) if cqi_matrix.size > 0 else 1.0
                norm_d = min(eta / (r_max / 15.0), 1.0)
                wl = workload.get(agent.agent_id, 0)
                norm_w = min(wl / n_tasks, 1.0)
                norm_b = min(agent.battery / 100.0, 1.0)
                norm_c = min(mean_cqi, 1.0)
                aff_penalty = domain_skill_affinity(agent, req) * 5.0
                wl_penalty = 100.0 if wl > 0 else 0.0
                return aff_penalty + wl_penalty + w_d * norm_d + w_w * norm_w - w_b * norm_b - w_c * norm_c

            # 1. Single agent covering all skills
            if full_cands:
                for agent in full_cands:
                    score = _agent_score(agent)
                    if score < best_utility or (score == best_utility and best_team is not None and [agent.agent_id] < best_team):
                        best_utility = score
                        best_team = [agent.agent_id]

            # 2. Complementary pair covering all skills if no single agent found
            if not best_team:
                for i in range(len(fleet.agents)):
                    for j in range(i + 1, len(fleet.agents)):
                        a1, a2 = fleet.agents[i], fleet.agents[j]
                        if req.issubset(set(a1.skills) | set(a2.skills)):
                            s1 = _agent_score(a1)
                            s2 = _agent_score(a2)
                            score = max(s1, s2)
                            pair_ids = sorted([a1.agent_id, a2.agent_id])
                            if score < best_utility or (score == best_utility and best_team is not None and pair_ids < best_team):
                                best_utility = score
                                best_team = pair_ids

            if best_team:
                for aid in best_team:
                    workload[aid] = workload.get(aid, 0) + 1
                    found = any(aid in c.get("members", []) for c in updated_coalitions)
                    if not found and updated_coalitions:
                        updated_coalitions[0]["members"].append(aid)

        return updated_coalitions


