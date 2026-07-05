"""Post-switch coalition reallocation (Eq 31)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.coalition.formation import CoalitionFormation
from src.env.agents import AgentFleet
from src.env.scenarios import Subtask
from src.llm.device_llm_client import DeviceLLMClient


@dataclass
class PostSwitchReallocator:
    device_llm: DeviceLLMClient
    coalition_formation: CoalitionFormation

    def should_trigger(
        self,
        mode_changed: bool,
        coalitions: list[dict],
        fleet: AgentFleet,
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> bool:
        """Eq 31: REALLOC(t*) indicator."""
        if not mode_changed:
            return False
        cfr = self.coalition_formation.compute_cfr(
            coalitions, fleet, distance_matrix, cqi_matrix
        )
        return cfr < 1.0

    def reallocate(
        self,
        fleet: AgentFleet,
        subtasks: list[Subtask],
        coalitions: list[dict],
        distance_matrix: np.ndarray,
        cqi_matrix: np.ndarray,
    ) -> list[dict]:
        """Reallocate remaining subtasks via Device LLM."""
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

        new_coalitions = self.device_llm.reallocate_remaining(
            remaining,
            fleet.to_dict_list(),
            distance_matrix.tolist(),
            cqi_matrix.tolist(),
        )
        if not new_coalitions:
            new_coalitions = self.coalition_formation.form(
                fleet, subtasks, distance_matrix, cqi_matrix
            )
        return new_coalitions
