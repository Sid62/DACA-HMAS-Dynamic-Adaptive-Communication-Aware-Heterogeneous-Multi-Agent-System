"""Extended coalition formation (Eq 25)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.coalition.feasibility import (
    build_psi_matrix,
    coalition_feasibility_rate,
    coalition_feasibility_score,
    validate_coalition_members,
)
from src.env.agents import AgentFleet
from src.env.scenarios import Subtask
from src.llm.cloud_llm_client import CloudLLMClient


@dataclass
class CoalitionFormation:
    cloud_llm: CloudLLMClient
    c1: float = 50.0
    gamma_min: float = 0.3
    max_retries: int = 3
    merged_singleton_count: int = 0
    _coalition_id_registry: dict[frozenset[str], int] = field(default_factory=dict)
    _next_coalition_id: int = 0

    def form(
        self,
        fleet: AgentFleet,
        subtasks: list[Subtask],
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> list[dict]:
        """Eq 25: A(T) = LLM(T, S, D(t), Q(t)) subject to Gamma_k >= Gamma_min."""
        psi = build_psi_matrix(distance_matrix, cqi_matrix, self.c1)
        id_to_idx = {a.agent_id: i for i, a in enumerate(fleet.agents)}
        agents_ctx = fleet.to_dict_list()
        subtasks_ctx = [
            {"id": s.subtask_id, "skills": s.required_skills} for s in subtasks
        ]

        coalitions: list[dict] = []
        for attempt in range(self.max_retries):
            raw = self.cloud_llm.form_coalitions(
                subtasks_ctx,
                agents_ctx,
                distance_matrix.tolist(),
                cqi_matrix.tolist(),
            )
            coalitions = []
            infeasible = []
            for c in raw:
                members = c.get("members", [])
                if validate_coalition_members(
                    members, id_to_idx, psi, self.gamma_min
                ):
                    coalitions.append(c)
                else:
                    infeasible.append(c)
            if not infeasible:
                break
            coalitions.extend(self._repair_infeasible(infeasible, fleet, psi, id_to_idx))

        coalitions = self._merge_singleton_coalitions(coalitions, fleet, psi, id_to_idx)

        # Fallback: if coalition formation produced nothing (e.g. real LLM
        # returned unparseable output AND all repair attempts yielded empty
        # results), create one singleton per agent so that downstream
        # distributed planning always has coalitions to iterate over.
        if not coalitions:
            print("\nFallback coalition generation used\n")
            coalitions = [
                {"coalition_id": i, "members": [a.agent_id]}
                for i, a in enumerate(fleet.agents)
            ]

        coalitions = self._stabilize_coalition_ids(coalitions)

        for c in coalitions:
            for mid in c.get("members", []):
                if mid in id_to_idx:
                    fleet.agents[id_to_idx[mid]].coalition_id = c.get("coalition_id")
        return coalitions
    
    def _merge_singleton_coalitions(
        self,
        coalitions: list[dict],
        fleet: AgentFleet,
        psi: np.ndarray,
        id_to_idx: dict[str, int],
    ) -> list[dict]:
        """Merge singleton coalitions of the same agent-type domain into one
        group whenever the combined group remains psi-feasible (Eq 25,
        Gamma_k >= Gamma_min). Multi-member coalitions are left untouched;
        this only removes the degenerate 1-agent-per-coalition case that
        the feasibility check alone does not prevent. Domain match is used
        as a compatible-skills proxy (agents of one type share the
        scenario's per-type skill/kinematics profile).
        """
        singles = [c for c in coalitions if len(c.get("members", [])) == 1]
        non_singles = [c for c in coalitions if len(c.get("members", [])) != 1]
        if len(singles) < 2:
            return coalitions

        domain_of = {a.agent_id: a.agent_type.value for a in fleet.agents}
        used: set[str] = set()
        merged_groups: list[list[str]] = []
        singles_sorted = sorted(singles, key=lambda c: c.get("members", [""])[0])

        for i, c in enumerate(singles_sorted):
            mid = c.get("members", [""])[0]
            if mid in used or mid not in id_to_idx:
                continue
            group = [mid]
            used.add(mid)
            for other in singles_sorted[i + 1:]:
                oid = other.get("members", [""])[0]
                if oid in used or oid not in id_to_idx:
                    continue
                if domain_of.get(oid) != domain_of.get(mid):
                    continue
                trial_indices = [id_to_idx[m] for m in group + [oid]]
                if coalition_feasibility_score(trial_indices, psi) >= self.gamma_min:
                    group.append(oid)
                    used.add(oid)
            merged_groups.append(group)

        result = list(non_singles)
        for group in merged_groups:
            result.append({"coalition_id": len(result), "members": group})
            if len(group) > 1:
                self.merged_singleton_count += len(group)
        for i, c in enumerate(result):
            c["coalition_id"] = i  # renumber to avoid id collisions after merge
        return result

    def _stabilize_coalition_ids(
        self,
        coalitions: list[dict],
    ) -> list[dict]:
        stabilized = []
        seen = set()
    
        for c in coalitions:
            key = frozenset(c.get("members", []))
            seen.add(key)
    
            if key not in self._coalition_id_registry:
                self._coalition_id_registry[key] = self._next_coalition_id
                self._next_coalition_id += 1
    
            stabilized.append({
                **c,
                "coalition_id": self._coalition_id_registry[key],
            })
    
        stale = [
            k for k in self._coalition_id_registry
            if k not in seen
        ]
    
        for k in stale:
            del self._coalition_id_registry[k]
    
        return stabilized

    def _repair_infeasible(
        self,
        infeasible: list[dict],
        fleet: AgentFleet,
        psi: np.ndarray,
        id_to_idx: dict[str, int],
    ) -> list[dict]:
        """Split infeasible coalitions into feasible sub-groups."""
        repaired = []
        for c in infeasible:
            members = c.get("members", [])
            indices = [id_to_idx[m] for m in members if m in id_to_idx]
            if coalition_feasibility_score(indices, psi) >= self.gamma_min:
                repaired.append(c)
                continue
            for m in members:
                if m in id_to_idx:
                    repaired.append({"coalition_id": len(repaired), "members": [m]})
        return repaired

    def compute_cfr(
        self,
        coalitions: list[dict],
        fleet: AgentFleet,
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> float:
        psi = build_psi_matrix(distance_matrix, cqi_matrix, self.c1)
        id_to_idx = {a.agent_id: i for i, a in enumerate(fleet.agents)}
        member_lists = [
            [id_to_idx[m] for m in c.get("members", []) if m in id_to_idx]
            for c in coalitions
        ]
        return coalition_feasibility_rate(member_lists, psi, self.gamma_min)
