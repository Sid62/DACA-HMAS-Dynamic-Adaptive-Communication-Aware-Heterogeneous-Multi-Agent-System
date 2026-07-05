"""Extended coalition formation (Eq 25)."""

from __future__ import annotations

from dataclasses import dataclass
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

        for c in coalitions:
            for mid in c.get("members", []):
                if mid in id_to_idx:
                    fleet.agents[id_to_idx[mid]].coalition_id = c.get("coalition_id")
        return coalitions

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
