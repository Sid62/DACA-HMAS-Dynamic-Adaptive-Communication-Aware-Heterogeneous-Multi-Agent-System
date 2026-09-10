"""AutoHMA-LLM structural alignment data structures.

Lightweight dataclasses representing AutoHMA-aligned concepts:
- ExecutionFeedback: per-agent execution result
- DeviceFeedback: per-domain aggregated review
- ExecutionDirective: Device LLM dispatch output wrapper

These are purely structural wrappers over data that already exists in memory.
They introduce no new computation, no new API calls, and no behavioral change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionFeedback:
    """Per-agent execution result from Generative Agent → Device LLM."""

    agent_id: str = ""
    subtask_id: str = ""
    distance_to_target: float = 0.0
    completed: bool = False
    step: int = 0

    def summary(self) -> str:
        status = "DONE" if self.completed else f"d={self.distance_to_target:.1f}"
        return f"{self.agent_id}:{self.subtask_id}({status})"


@dataclass
class DeviceFeedback:
    """Aggregated Device LLM review of execution in its managed domain.

    AutoHMA flow: Generative Agent → Device LLM → Cloud LLM
    This represents the Device LLM → Cloud LLM upward feedback.
    """

    domain_id: str = ""
    agent_feedbacks: list[ExecutionFeedback] = field(default_factory=list)
    tasks_completed: list[str] = field(default_factory=list)
    tasks_in_progress: list[str] = field(default_factory=list)
    step: int = 0

    def to_context_string(self) -> str:
        """Format as compact text for injection into Cloud LLM prompt."""
        parts = [f"Domain={self.domain_id}"]
        if self.tasks_completed:
            parts.append(f"done=[{','.join(self.tasks_completed)}]")
        if self.tasks_in_progress:
            progress = []
            for fb in self.agent_feedbacks:
                if not fb.completed:
                    progress.append(fb.summary())
            if progress:
                parts.append(f"progress=[{','.join(progress)}]")
        return " ".join(parts)


@dataclass
class ExecutionDirective:
    """Wrapper for Device LLM dispatch output.

    AutoHMA flow: Cloud LLM → Device LLM → Generative Agent
    This represents the Device LLM → Generative Agent directive.
    The actual control remains with the existing PID/NMPC/Q-learning;
    this is the structural link that is consumed by the execution path.
    """

    domain_id: str = ""
    dispatch_result: dict[str, Any] = field(default_factory=dict)
    coalitions: list[dict] = field(default_factory=list)
    agent_assignments: dict[str, str] = field(default_factory=dict)
    step: int = 0

    def get_agent_assignment(self, agent_id: str) -> str | None:
        """Extract assigned subtask id for a specific agent from this domain directive."""
        if agent_id in self.agent_assignments:
            return self.agent_assignments[agent_id]
        if "assignments" in self.dispatch_result and agent_id in self.dispatch_result["assignments"]:
            return self.dispatch_result["assignments"][agent_id]
        return None


def format_feedback_for_cloud(
    device_feedbacks: list[DeviceFeedback],
) -> str | None:
    """Format Device LLM feedbacks as a compact context string for Cloud LLM.

    Takes the latest feedback per domain to avoid redundant duplication across steps.
    Includes explicit self-correction instructions for the Cloud planner.
    Returns None if there is no meaningful feedback to include,
    avoiding prompt bloat on the first planning call.
    """
    if not device_feedbacks:
        return None

    # Take the latest feedback per domain
    latest_by_domain: dict[str, DeviceFeedback] = {}
    for fb in device_feedbacks:
        latest_by_domain[fb.domain_id] = fb

    has_content = any(
        fb.tasks_completed or fb.tasks_in_progress
        for fb in latest_by_domain.values()
    )
    if not has_content:
        return None

    lines = [
        "[Previous Execution Feedback & Self-Correction Context]",
        "Instructions: Review previous execution feedback before generating the next plan.",
        "If previous assignments produced incomplete tasks, insufficient progress, or bottlenecks, refine the allocation accordingly.",
        "Do not blindly repeat assignments that proved ineffective in the feedback.",
    ]
    for domain_id in sorted(latest_by_domain.keys()):
        line = latest_by_domain[domain_id].to_context_string()
        if line:
            lines.append(line)
    return "\n".join(lines)
