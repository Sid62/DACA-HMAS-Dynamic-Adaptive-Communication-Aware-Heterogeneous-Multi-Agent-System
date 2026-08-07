"""Compact State Summarizer for Cloud LLM Planning Preprocessing (Optimization 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SummarizationResult:
    summary_dict: dict[str, Any]
    summary_text: str
    original_chars: int
    summarized_chars: int
    prompt_reduction_percent: float


class CompactStateSummarizer:
    """Preprocesses raw multi-agent system state into a concise, structured

    planning-relevant summary before sending to Cloud LLM.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def summarize_decomposition_context(
        self,
        instruction: str,
        agents: list[dict[str, Any]],
        subtasks: list[dict[str, Any]],
        dist_mat: list[list[float]] | None = None,
        network_info: dict[str, Any] | None = None,
    ) -> SummarizationResult:
        raw_repr = (
            f"Instruction: {instruction}\n"
            f"Agents: {agents}\n"
            f"Subtasks: {subtasks}\n"
            f"Distances: {dist_mat}\n"
            f"Network: {network_info}"
        )
        orig_chars = len(raw_repr)

        if not self.enabled:
            return SummarizationResult(
                summary_dict={
                    "instruction": instruction,
                    "agents": agents,
                    "subtasks": subtasks,
                },
                summary_text=raw_repr,
                original_chars=orig_chars,
                summarized_chars=orig_chars,
                prompt_reduction_percent=0.0,
            )

        # Filter active agents & essential state
        active_agents = []
        failed_agents = []
        for a in agents:
            aid = a.get("id", a.get("agent_id"))
            status = a.get("status", "active")
            skills = a.get("skills", [])
            pos = a.get("pos", a.get("position"))
            if status == "failed" or a.get("battery", 100.0) <= 0:
                failed_agents.append(aid)
            else:
                active_agents.append({
                    "id": aid,
                    "type": a.get("type", a.get("agent_type")),
                    "skills": skills,
                    "pos": [round(p, 1) for p in pos] if isinstance(pos, (list, tuple)) else pos,
                })

        # Compact subtasks: target & skills
        compact_subtasks = []
        for s in subtasks:
            sid = s.get("id", s.get("subtask_id"))
            completed = s.get("completed", False)
            if not completed:
                target = s.get("target")
                compact_subtasks.append({
                    "id": sid,
                    "skills": s.get("skills", s.get("required_skills", [])),
                    "target": [round(t, 1) for t in target] if isinstance(target, (list, tuple)) else target,
                })

        summary_dict = {
            "task": instruction,
            "active_agents": active_agents,
            "failed_agents": failed_agents,
            "subtasks": compact_subtasks,
        }
        if network_info:
            summary_dict["cqi"] = round(network_info.get("sys_cqi", 1.0), 3)

        summary_text = (
            f"Task: {instruction}\n"
            f"Active Agents ({len(active_agents)}): {[{'id': a['id'], 'skills': a['skills']} for a in active_agents]}\n"
            f"Subtasks ({len(compact_subtasks)}): {compact_subtasks}"
        )
        summ_chars = len(summary_text)
        reduction = max(0.0, round(((orig_chars - summ_chars) / max(1, orig_chars)) * 100.0, 2))

        return SummarizationResult(
            summary_dict=summary_dict,
            summary_text=summary_text,
            original_chars=orig_chars,
            summarized_chars=summ_chars,
            prompt_reduction_percent=reduction,
        )

    def summarize_coalition_context(
        self,
        subtasks: list[dict[str, Any]],
        agents: list[dict[str, Any]],
        dist_mat: list[list[float]],
        cqi_mat: list[list[float]],
    ) -> SummarizationResult:
        raw_repr = f"Subtasks: {subtasks}\nAgents: {agents}\nDist: {dist_mat}\nCQI: {cqi_mat}"
        orig_chars = len(raw_repr)

        if not self.enabled:
            return SummarizationResult(
                summary_dict={"subtasks": subtasks, "agents": agents},
                summary_text=raw_repr,
                original_chars=orig_chars,
                summarized_chars=orig_chars,
                prompt_reduction_percent=0.0,
            )

        compact_agents = [
            {"id": a.get("id", a.get("agent_id")), "type": a.get("type", a.get("agent_type"))}
            for a in agents
        ]
        compact_subtasks = [
            {"id": s.get("id", s.get("subtask_id")), "skills": s.get("skills", s.get("required_skills", []))}
            for s in subtasks
        ]

        summary_dict = {
            "agents": compact_agents,
            "subtasks": compact_subtasks,
        }
        summary_text = f"Agents: {compact_agents}\nSubtasks: {compact_subtasks}"
        summ_chars = len(summary_text)
        reduction = max(0.0, round(((orig_chars - summ_chars) / max(1, orig_chars)) * 100.0, 2))

        return SummarizationResult(
            summary_dict=summary_dict,
            summary_text=summary_text,
            original_chars=orig_chars,
            summarized_chars=summ_chars,
            prompt_reduction_percent=reduction,
        )
