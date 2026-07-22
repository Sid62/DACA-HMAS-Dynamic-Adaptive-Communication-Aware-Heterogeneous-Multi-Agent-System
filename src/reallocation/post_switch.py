"""Post-switch coalition reallocation with domain-level leader-peer consensus."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.communication.models import (
    discover_agent_type_domains,
    dominant_domain_for_coalition,
    domains_in_coalition,
)
from src.communication.peer_manager import PeerCommunicationManager
from src.env.agents import AgentFleet
from src.env.scenarios import Subtask
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class PostSwitchReallocator:
    device_llms: dict[str, DeviceLLMClient] = field(default_factory=dict)
    coalition_formation: CoalitionFormation | None = None
    peer_manager: PeerCommunicationManager | None = None

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
    ) -> bool:
        """Eq 31: REALLOC(t*) indicator -- trigger local reallocation whenever architecture switches."""
        return mode_changed


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
            candidates = [
                a for a in fleet.agents if all(s in a.skills for s in st.required_skills)
            ]
            if not candidates:
                candidates = [
                    a for a in fleet.agents if any(s in a.skills for s in st.required_skills)
                ]
            if not candidates:
                candidates = fleet.agents

            best_agent = None
            best_utility = float("inf")
            n_tasks = max(len(subtasks), 1)
            r_max = 100.0

            # Convex weights: dist=0.40, workload=0.40, battery=0.10, cqi=0.10 (sum = 1.0)
            w_d, w_w, w_b, w_c = 0.40, 0.40, 0.10, 0.10

            for agent in candidates:
                d = dist(agent.position, st.target)
                idx = id_to_idx.get(agent.agent_id, 0)
                mean_cqi = float(np.mean(cqi_matrix[idx, :])) if cqi_matrix.size > 0 else 1.0

                norm_d = min(d / r_max, 1.0)
                norm_w = min(workload.get(agent.agent_id, 0) / n_tasks, 1.0)
                norm_b = min(agent.battery / 100.0, 1.0)
                norm_c = min(mean_cqi, 1.0)

                # Composite cost score C(a, s) in [0, 1]
                score = w_d * norm_d + w_w * norm_w - w_b * norm_b - w_c * norm_c
                if score < best_utility:
                    best_utility = score
                    best_agent = agent

            if best_agent:
                aid = best_agent.agent_id
                workload[aid] = workload.get(aid, 0) + 1
                # Ensure agent belongs to an active coalition
                found = False
                for c in updated_coalitions:
                    if aid in c.get("members", []):
                        found = True
                        break
                if not found:
                    updated_coalitions[0]["members"].append(aid)

        return updated_coalitions

