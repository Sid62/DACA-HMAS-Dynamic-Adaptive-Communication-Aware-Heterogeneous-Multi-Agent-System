"""Decentralized Hybrid architecture with domain-level multi-Device-LLM planning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.communication.models import (
    SharedPlan,
    discover_agent_type_domains,
    dominant_domain_for_coalition,
    domains_in_coalition,
)
from src.communication.peer_manager import PeerCommunicationManager
from src.control.q_learning import QLearningCA
from src.decomposition.distance_feasible_decomp import DistanceFeasibleDecomposer
from src.env.agents import AgentFleet, distance_matrix, dist
from src.env.daca_env import DACAEnv
from src.llm.cloud_llm_client import CloudLLMClient
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class DecentralizedHybridCoordinator:
    cloud_llm: CloudLLMClient
    device_llms: dict[str, DeviceLLMClient] = field(default_factory=dict)
    peer_manager: PeerCommunicationManager | None = None
    decomposer: DistanceFeasibleDecomposer | None = None
    coalition_formation: CoalitionFormation | None = None
    q_learning: QLearningCA = field(default_factory=QLearningCA)
    use_distance_decomp: bool = False
    use_coalition_feasibility: bool = False
    shared_plans: dict[int, SharedPlan] = field(default_factory=dict)
    local_reallocation_count: int = 0
    #Optimization 3: per-coalition context cache used to skip redundant
    #LLM planning cycles when a coalition's members/subtask/positions are
    effectively unchanged since it was last actually planned.
    _coalition_plan_context: dict[int, dict[str, Any]] = field(default_factory=dict)
    position_delta_threshold: float = 5.0  # reuses C2 collision-avoidance scale
    plan_reuse_count: int = 0

    @property
    def device_llm(self) -> DeviceLLMClient:
        """Backward-compatible accessor — returns first registered Device LLM."""
        if not self.device_llms:
            return DeviceLLMClient(node_id="device_0")
        return next(iter(self.device_llms.values()))
    
    def _local_observation(self, env: DACAEnv, agent_id: str) -> dict[str, Any]:
        obs = env.get_observation()
        print(obs["subtasks"][0])
        print(type(obs["subtasks"]))
        print(obs["subtasks"][:1])
        agent = env.fleet.get_agent(agent_id)
        return {
            "agent_id": agent_id,
            "agent_type": agent.agent_type.value,
            "position": [agent.position.x, agent.position.y, agent.position.z],
            "skills": agent.skills,
            "coalition_id": agent.coalition_id,
            "timestep": obs["timestep"],
            "subtasks": [
                         {
                            "subtask_id": s["id"],
                            "completed": s["completed"],
                           
                         }
                         for s in obs["subtasks"]
                        ]
        }

    def _fleet_observations(self, env: DACAEnv) -> dict[str, dict[str, Any]]:
        return {
            agent.agent_id: self._local_observation(env, agent.agent_id)
            for agent in env.fleet.agents
        }

    def _is_domain_mode(self, fleet: AgentFleet) -> bool:
        """True when device_llms are keyed by agent-type domain."""
        domains = discover_agent_type_domains(fleet)
        return any(domain in self.device_llms for domain in domains)

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

    def _select_leader_domain(self, members: list[str], fleet: AgentFleet) -> str:
        return dominant_domain_for_coalition(members, fleet)
    
    def _coalition_assigned_subtasks(
        self, members: list[str], assignments_map: dict[str, list[str]]
    ) -> frozenset[str]:
        member_set = set(members)
        return frozenset(
            sid
            for sid, agents in assignments_map.items()
            if any(a in member_set for a in agents)
        )
    
    def _snapshot_coalition_context(
        self,
        coalition_id: int,
        members: list[str],
        env: DACAEnv,
        assignments_map: dict[str, list[str]],
    ) -> None:
        positions = {
            aid: (
                env.fleet.get_agent(aid).position.x,
                env.fleet.get_agent(aid).position.y,
            )
            for aid in members
            if env.fleet.get_agent(aid) is not None
        }
    
        self._coalition_plan_context[coalition_id] = {
            "members": frozenset(members),
            "subtasks": self._coalition_assigned_subtasks(members, assignments_map),
            "positions": positions,
        }
    
    def _coalition_context_changed(
        self,
        coalition_id: int,
        members: list[str],
        env: DACAEnv,
        assignments_map: dict[str, list[str]],
    ) -> bool:
    
        cached = self._coalition_plan_context.get(coalition_id)
    
        if cached is None:
            return True
    
        if cached["members"] != frozenset(members):
            return True
    
        if cached["subtasks"] != self._coalition_assigned_subtasks(members, assignments_map):
            return True
    
        for aid in members:
            agent = env.fleet.get_agent(aid)
    
            if agent is None:
                return True
    
            old_pos = cached["positions"].get(aid)
    
            if old_pos is None:
                return True

            dx = agent.position.x - old_pos[0]
            dy = agent.position.y - old_pos[1]
    
            if (dx * dx + dy * dy) ** 0.5 > self.position_delta_threshold:
                return True
    
        return False

    def _update_domain_states(
        self,
        env: DACAEnv,
        members: list[str],
        domains: list[str],
        shared_plan_version: int,
    ) -> None:
        domain_map = discover_agent_type_domains(env.fleet)
        for domain in domains:
            client = self._domain_client(domain, env.fleet)
            if client is None:
                continue
            coalition_agents = [
                aid for aid in domain_map.get(domain, []) if aid in members
            ]
            obs_map = {
                aid: self._local_observation(env, aid) for aid in coalition_agents
            }
            client.update_local_state(
                local_observations=obs_map,
                shared_plan_version=shared_plan_version,
            )

    def _run_distributed_coalition_planning(
        self,
        coalition: dict[str, Any],
        env: DACAEnv,
        assignments_map: dict[str, list[str]],
    ) -> SharedPlan:
        """Leader domain plans → shares → peer domains review → merge → consensus."""
        pm = self.peer_manager
        if pm is None:
            cid = int(coalition.get("coalition_id", 0))
            return self.shared_plans.get(cid, SharedPlan(coalition_id=cid))

        members: list[str] = coalition.get("members", [])
        if not members:
            return SharedPlan(coalition_id=int(coalition.get("coalition_id", 0)))

        coalition_id = int(coalition.get("coalition_id", 0))
        leader_domain = self._select_leader_domain(members, env.fleet)
        domains = domains_in_coalition(members, env.fleet)
        print("Leader:", leader_domain)
        print("Domains:", domains)
        shared = self.shared_plans.get(
            coalition_id,
            SharedPlan(
                coalition_id=coalition_id,
                leader=leader_domain,
                leader_domain=leader_domain,
            ),
        )

        t0 = time.perf_counter()
        pm.record_distributed_replanning()

        leader_client = self._domain_client(leader_domain, env.fleet)
        if leader_client is None:
            return shared

        coalition_subtasks = [
            sid
            for sid, agents in assignments_map.items()
            if any(a in members for a in agents)
        ]
        coalition_state = {
            "coalition_id": coalition_id,
            "members": members,
            "leader_domain": leader_domain,
            "domains": domains,
        }

        self._update_domain_states(env, members, domains, shared.version)

        leader_msgs = pm.receive_messages(leader_domain)
        leader_client.ingest_messages(leader_msgs)

        print(f"[COALITION-PLAN] coalition_id={coalition_id} leader={leader_domain} "
              f"members={members} num_members={len(members)}")

        t_plan_start = time.perf_counter()
        leader_plan = leader_client.plan_local(
            coalition_id, members, shared, leader_msgs, coalition_state
        )
        t_plan_elapsed = time.perf_counter() - t_plan_start
        print(f"[COALITION-PLAN] coalition_id={coalition_id} leader={leader_domain} "
              f"COMPLETED elapsed={t_plan_elapsed:.2f}s")
        
        print("Broadcasting from:", leader_domain)

        pm.broadcast(
            leader_domain,
            "plan_proposal",
            {
                "coalition_id": coalition_id,
                "plan": leader_plan,
                "version": shared.version,
                "leader_domain": leader_domain,
            },
        )
        

        peer_reviews: list[dict[str, Any]] = []
        for peer_domain in domains:
            if peer_domain == leader_domain:
                continue
            peer_client = self._domain_client(peer_domain, env.fleet)
            if peer_client is None:
                continue
            msgs = pm.receive_messages(peer_domain)
            peer_client.ingest_messages(msgs)
            for msg in msgs:
                if msg.message_type == "plan_proposal":
                    review = peer_client.review_peer_plan(
                        leader_domain, msg.payload.get("plan", {}), shared.version
                    )
                    peer_reviews.append({"peer_domain": peer_domain, "review": review})
                    print("Sending review message")
                    pm.send_message(
                        peer_domain,
                        leader_domain,
                        "plan_review",
                        {"coalition_id": coalition_id, "review": review},
                    )
                    response = peer_client.respond_to_peer(
                        leader_domain, "plan_proposal", msg.payload, shared.version
                    )
                    print("Sending review message")
                    pm.send_message(
                        peer_domain,
                        leader_domain,
                        "peer_response",
                        {"response": response},
                    )

        leader_reviews = pm.receive_messages(leader_domain)
        leader_client.ingest_messages(leader_reviews)
        merged = leader_client.merge_peer_plan(
            coalition_id, leader_plan, peer_reviews, shared.version
        )
        pm.record_plan_merge()

        merged_assignments = merged.get("merged_plan", {}).get(
            "assignments", leader_plan.get("assignments", {})
        )
        shared.leader = leader_domain
        shared.leader_domain = leader_domain
        shared.subtasks = coalition_subtasks
        shared.agent_assignments = {
            k: v for k, v in merged_assignments.items() if k in members
        }
        shared.bump_version("consensus")
        self.shared_plans[coalition_id] = shared

        peer_domains = [d for d in domains if d != leader_domain]
        if peer_domains:
            pm.multicast(
                leader_domain,
                peer_domains,
                "plan_final",
                {"coalition_id": coalition_id, "shared_plan": shared.to_dict()},
            )

        for domain in domains:
            client = self._domain_client(domain, env.fleet)
            if client:
                client.node_state.shared_plan_version = shared.version
                client.node_state.shared_plan = shared.to_dict()
                client.node_state.received_messages.clear()
                
        print("Consensus completed")
        pm.record_consensus_round(time.perf_counter() - t0)
        return shared
    def _closer_domain_mate(self, agent_id, target, fleet, domain, current_dist):
        """Return a same-domain agent meaningfully closer to target, else None."""
        domain_map = discover_agent_type_domains(fleet)
        best_id, best_dist = None, current_dist
        for aid in domain_map.get(domain, []):
            if aid == agent_id:
                continue
            agent = fleet.get_agent(aid)
            if agent is None:
                continue
            d = dist(agent.position, target)
            if d < best_dist * 0.8:  # meaningful margin -- avoids churn from noise
                best_id, best_dist = aid, d
        return best_id

    def _local_reassign(
        self,
        env: DACAEnv,
        assignments_map: dict[str, list[str]],
        coalitions: list[dict[str, Any]],
        cqi_matrix: np.ndarray,
    ) -> None:
        """Independent local reassignment of ONLY the affected subtasks.
        Triggers: agent unavailable, closer same-domain neighbor, poor
        cloud/peer comm, or broken coalition membership. Mutates
        assignments_map in place; never recomputes the full mission.
        """
        fleet = env.fleet
        live_agent_ids = {a.agent_id for a in fleet.agents}
        targets = {s.subtask_id: s.target for s in env.subtask_list}
        domain_map = discover_agent_type_domains(fleet)
        agent_domain = {aid: d for d, ids in domain_map.items() for aid in ids}
        coalition_members_by_id = {
            c.get("coalition_id"): set(c.get("members", [])) for c in coalitions
        }
        sys_cqi_poor = (
            self.peer_manager is not None
            and self.peer_manager.domain_cqi_matrix is not None
            and float(np.mean(self.peer_manager.domain_cqi_matrix)) < 0.5
        )

        for sid, agent_list in assignments_map.items():
            if not agent_list:
                continue
            current_agent = agent_list[0]
            target = targets.get(sid)
            if target is None:
                continue
            domain = agent_domain.get(current_agent)
            if domain is None:
                continue

            reason = None
            if current_agent not in live_agent_ids:
                reason = "agent_unavailable"
            else:
                current_dist = dist(fleet.get_agent(current_agent).position, target)
                if self._closer_domain_mate(current_agent, target, fleet, domain, current_dist):
                    reason = "closer_neighbor"
                elif sys_cqi_poor:
                    reason = "poor_cloud_comm"

            cid = next(
                (c for c, members in coalition_members_by_id.items() if current_agent in members),
                None,
            )
            if cid is not None and (coalition_members_by_id.get(cid, set()) - live_agent_ids):
                reason = reason or "coalition_broken"

            if reason and reason != "poor_cloud_comm":
                candidates = [a for a in domain_map.get(domain, []) if a in live_agent_ids]
                if not candidates:
                    continue
                new_agent = min(
                    candidates, key=lambda aid: dist(fleet.get_agent(aid).position, target)
                )
                if new_agent != current_agent:
                    assignments_map[sid] = [new_agent]
                    self.local_reallocation_count += 1
                    print(f"[LOCAL-REALLOC] {sid}: {current_agent} -> {new_agent} ({reason})")
    def plan(
        self,
        env: DACAEnv,
        cqi_matrix: np.ndarray | None = None,
    ) -> tuple[dict, list[dict]]:
        obs = env.get_observation()
        fleet = env.fleet
        subtasks = env.subtask_list
        dist_mat = distance_matrix(fleet.agents)
        if cqi_matrix is None:
            cqi_matrix = np.ones(dist_mat.shape)

        if self.peer_manager:
            agent_ids = [a.agent_id for a in fleet.agents]
            domain_map = discover_agent_type_domains(fleet)
            if self._is_domain_mode(fleet):
                self.peer_manager.set_domain_topology(
                    domain_map, agent_ids, cqi_matrix
                )
            else:
                self.peer_manager.set_topology(agent_ids, cqi_matrix)

        if self.use_distance_decomp and self.decomposer:
            assignments_map = self.decomposer.decompose(
                obs["instruction"], fleet, subtasks
            )
        else:
            assignments_map = self.cloud_llm.decompose(
                obs["instruction"], obs["agents"], obs["subtasks"]
            )
        
        print("\n========== ASSIGNMENTS ==========")
        print(assignments_map)
        print("================================\n")

        if self.use_coalition_feasibility and self.coalition_formation:
            coalitions = self.coalition_formation.form(
                fleet, subtasks, dist_mat, cqi_matrix
            )
        else:
            coalitions = self.cloud_llm.form_coalitions(
                obs["subtasks"], obs["agents"]
            )
        
        print("\n========== COALITIONS ==========")
        print(coalitions)
        print("Number of coalitions:", len(coalitions))
        print("================================\n")
        
        self._local_reassign(env, assignments_map, coalitions, cqi_matrix)
        
        if self.device_llms and self.peer_manager:
        for coalition in coalitions:
    
            coalition_id = int(coalition.get("coalition_id", 0))
            members = coalition.get("members", [])
    
            if not self._coalition_context_changed(
                coalition_id,
                members,
                env,
                assignments_map,
            ):
                self.plan_reuse_count += 1
    
                print(
                    f"[PLAN-REUSE] coalition_id={coalition_id} "
                    f"context unchanged -- skipping LLM planning cycle"
                )
    
                continue
    
            print("\nProcessing coalition")
            print(coalition)
    
            self._run_distributed_coalition_planning(
                coalition,
                env,
                assignments_map,
            )
    
            self._snapshot_coalition_context(
                coalition_id,
                members,
                env,
                assignments_map,
            )
        else:
            self.device_llm.coordinate_locally(
                coalitions, self._fleet_observations(env)
            )
        
        print("\n===== DEBUG =====")
        print("Assignments:", assignments_map)
        print("Coalitions:", coalitions)
        print("=================\n")
        return assignments_map, coalitions

    def execute_step(
        self,
        env: DACAEnv,
        assignments: dict[str, list[str]],
    ) -> None:
        self.q_learning.activate()
        targets = {s.subtask_id: s.target for s in env.subtask_list}
        agent_assignments = {}
        for sid, agents in assignments.items():
            if agents:
                agent_assignments[agents[0]] = sid
        self.q_learning.step(env.fleet, agent_assignments, targets)

        for sid, agent_list in assignments.items():
            if not agent_list:
                continue
            agent = env.fleet.get_agent(agent_list[0])
            subtask = next((s for s in env.subtask_list if s.subtask_id == sid), None)
            if subtask and dist(agent.position, subtask.target) < 5.0:
                env.mark_subtask_complete(sid)
