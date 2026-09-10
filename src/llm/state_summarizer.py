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
        c_task: float | None = None,
        r_reach: float | None = None,
    ) -> SummarizationResult:
        if c_task is None or r_reach is None:
            try:
                from src.config import get_thresholds
                th = get_thresholds()
                if c_task is None:
                    c_task = float(th.get("C_task", 30.0))
                if r_reach is None:
                    r_reach = float(th.get("R_reach", 100.0))
            except Exception:
                if c_task is None:
                    c_task = 30.0
                if r_reach is None:
                    r_reach = 100.0

        raw_repr = (
            f"Instruction: {instruction}\n"
            f"Agents: {agents}\n"
            f"Subtasks: {subtasks}\n"
            f"Distances: {dist_mat}\n"
            f"Network: {network_info}\n"
            f"Thresholds: C_task={c_task}, R_reach={r_reach}"
        )
        orig_chars = len(raw_repr)

        if not self.enabled:
            return SummarizationResult(
                summary_dict={
                    "task": instruction,
                    "instruction": instruction,
                    "agents": agents,
                    "subtasks": subtasks,
                    "dist_mat": dist_mat,
                    "network_info": network_info,
                    "c_task": c_task,
                    "r_reach": r_reach,
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

        summary_dict: dict[str, Any] = {
            "task": instruction,
            "active_agents": active_agents,
            "failed_agents": failed_agents,
            "subtasks": compact_subtasks,
        }
        if dist_mat is not None:
            summary_dict["dist_mat"] = [[round(d, 1) for d in row] for row in dist_mat]
        if network_info:
            summary_dict["cqi"] = round(network_info.get("sys_cqi", 1.0), 3)
        if c_task is not None:
            summary_dict["c_task"] = round(float(c_task), 2)
        if r_reach is not None:
            summary_dict["r_reach"] = round(float(r_reach), 2)

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
        gamma_min: float | None = None,
        c1: float | None = None,
    ) -> SummarizationResult:
        if gamma_min is None or c1 is None:
            try:
                from src.config import get_thresholds
                th = get_thresholds()
                if gamma_min is None:
                    gamma_min = float(th.get("gamma_min", 0.3))
                if c1 is None:
                    c1 = float(th.get("C1", 50.0))
            except Exception:
                if gamma_min is None:
                    gamma_min = 0.3
                if c1 is None:
                    c1 = 50.0

        raw_repr = f"Subtasks: {subtasks}\nAgents: {agents}\nDist: {dist_mat}\nCQI: {cqi_mat}\nThresholds: gamma_min={gamma_min}, C1={c1}"
        orig_chars = len(raw_repr)

        if not self.enabled:
            return SummarizationResult(
                summary_dict={
                    "task": "coalition_formation",
                    "subtasks": subtasks,
                    "agents": agents,
                    "dist_mat": dist_mat,
                    "cqi_mat": cqi_mat,
                    "gamma_min": gamma_min,
                    "c1": c1,
                },
                summary_text=raw_repr,
                original_chars=orig_chars,
                summarized_chars=orig_chars,
                prompt_reduction_percent=0.0,
            )

        compact_agents = []
        for a in agents:
            aid = a.get("id", a.get("agent_id"))
            status = a.get("status", "active")
            if status != "failed" and a.get("battery", 100.0) > 0:
                pos = a.get("pos", a.get("position"))
                compact_agents.append({
                    "id": aid,
                    "type": a.get("type", a.get("agent_type")),
                    "skills": a.get("skills", []),
                    "pos": [round(p, 1) for p in pos] if isinstance(pos, (list, tuple)) else pos,
                })

        compact_subtasks = []
        for s in subtasks:
            sid = s.get("id", s.get("subtask_id"))
            target = s.get("target")
            compact_subtasks.append({
                "id": sid,
                "skills": s.get("skills", s.get("required_skills", [])),
                "target": [round(t, 1) for t in target] if isinstance(target, (list, tuple)) else target,
            })

        summary_dict: dict[str, Any] = {
            "task": "coalition_formation",
            "agents": compact_agents,
            "subtasks": compact_subtasks,
        }
        if cqi_mat:
            import numpy as np
            cqi_vals = [float(q) for row in cqi_mat for q in row if isinstance(q, (int, float))]
            if cqi_vals:
                summary_dict["cqi"] = round(float(np.mean(cqi_vals)), 3)
                summary_dict["cqi_min"] = round(float(np.min(cqi_vals)), 3)
            summary_dict["cqi_matrix"] = [[round(float(q), 2) for q in row] for row in cqi_mat]
        if dist_mat:
            summary_dict["dist_mat"] = [[round(float(d), 1) for d in row] for row in dist_mat]
        if gamma_min is not None:
            summary_dict["gamma_min"] = round(float(gamma_min), 3)
        if c1 is not None:
            summary_dict["c1"] = round(float(c1), 2)

        summary_text = (
            f"Agents ({len(compact_agents)}): {[{'id': a['id'], 'skills': a['skills']} for a in compact_agents]}\n"
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
