"""Plan Repair Module for DACA-HMAS.

Implements partial replanning, repair prompt formatting, and plan patching.
Instead of regenerating the entire global plan from scratch, only infeasible
or affected subtasks are re-assigned, patching the previous global plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.env.agents import AgentFleet, dist
from src.env.scenarios import Subtask


@dataclass
class PlanPatch:
    """Incremental patch to be applied over an existing global plan."""
    patched_assignments: dict[str, list[str]]
    patched_coalitions: list[dict[str, Any]] | None = None


class PlanRepairer:
    """Handles partial replanning and plan patching for local Device/Cloud LLM reasoning."""

    def __init__(self, r_reach: float = 100.0):
        self.r_reach = r_reach

    def identify_affected_subtasks(
        self,
        active_assignments: dict[str, list[str]],
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
    ) -> list[str]:
        """Identify subtask IDs whose current assignments are infeasible or unassigned."""
        affected: set[str] = set()
        agent_map = {a.agent_id: a for a in fleet.agents}
        live_agent_ids = set(agent_map.keys())

        for s in subtasks:
            if s.completed:
                continue
            sid = s.subtask_id
            assigned = active_assignments.get(sid, [])
            if not assigned:
                affected.add(sid)
                continue

            # Check if assigned agents are live and within reach
            for aid in assigned:
                if aid not in live_agent_ids:
                    affected.add(sid)
                    break
                agent = agent_map[aid]
                if dist(agent.position, s.target) > self.r_reach:
                    affected.add(sid)
                    break

        return sorted(affected)

    def build_repair_prompt(
        self,
        active_assignments: dict[str, list[str]],
        active_coalitions: list[dict[str, Any]],
        affected_subtask_ids: list[str],
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
    ) -> str:
        """Construct a focused local repair prompt targeting ONLY affected subtasks."""
        subtask_map = {s.subtask_id: s for s in subtasks if not s.completed}
        affected_info = [
            {
                "subtask_id": sid,
                "target": [subtask_map[sid].target.x, subtask_map[sid].target.y],
                "required_skills": subtask_map[sid].required_skills,
            }
            for sid in affected_subtask_ids
            if sid in subtask_map
        ]
        unaffected_assignments = {
            sid: agents
            for sid, agents in active_assignments.items()
            if sid not in affected_subtask_ids and sid in subtask_map
        }
        agents_info = [
            {
                "agent_id": a.agent_id,
                "position": [a.position.x, a.position.y],
                "skills": a.skills,
            }
            for a in fleet.agents
        ]

        prompt = (
            "You are a Repair Planner for multi-agent execution.\n"
            "Given the previous global plan, repair ONLY the infeasible portion.\n\n"
            f"Unaffected Active Plan (KEEP INTACT): {json.dumps(unaffected_assignments)}\n"
            f"Affected Subtasks Requiring Reassignment: {json.dumps(affected_info)}\n"
            f"Available Fleet State: {json.dumps(agents_info)}\n"
            f"Existing Coalitions: {json.dumps(active_coalitions)}\n\n"
            "Return a JSON patch dictionary for affected subtasks only:\n"
            '{"assignments": {"T_affected": ["agent_id"]}}'
        )
        return prompt

    def apply_plan_patch(
        self,
        active_assignments: dict[str, list[str]],
        patch: PlanPatch,
    ) -> dict[str, list[str]]:
        """Apply patch to active global plan, mutating only changed assignments."""
        updated = dict(active_assignments)
        for sid, agents in patch.patched_assignments.items():
            updated[sid] = list(agents)
        return updated
