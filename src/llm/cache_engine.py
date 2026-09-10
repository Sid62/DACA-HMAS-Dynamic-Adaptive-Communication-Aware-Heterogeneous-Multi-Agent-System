"""Plan Cache, Prompt Cache, and State Difference Detection for DACA-HMAS.

Provides:
- PlanCache: memoizes global/coalition plans using a state signature hash.
- PromptCache: memoizes LLM responses using prompt SHA-256 hashes.
- StateDifferenceDetector: quantifies state change delta (delta S) to skip LLM calls when state change is below threshold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.env.agents import AgentFleet
from src.env.scenarios import Subtask


def compute_state_hash(
    fleet: AgentFleet,
    subtasks: Sequence[Subtask],
    coalitions: list[dict[str, Any]],
    sys_cqi: float = 1.0,
    pos_precision: float = 10.0,
) -> str:
    """Compute a deterministic hash signature representing the planning state."""
    # Discretize positions to pos_precision scale
    agent_state = sorted([
        (a.agent_id, round(a.position.x / pos_precision), round(a.position.y / pos_precision))
        for a in fleet.agents
    ])
    subtask_state = sorted([
        (s.subtask_id, s.completed) for s in subtasks
    ])
    coalition_state = sorted([
        (c.get("coalition_id", i), tuple(sorted(c.get("members", []))))
        for i, c in enumerate(coalitions)
    ])
    cqi_bucket = round(sys_cqi, 1)

    raw_key = json.dumps({
        "agents": agent_state,
        "subtasks": subtask_state,
        "coalitions": coalition_state,
        "cqi": cqi_bucket,
    }, sort_keys=True)

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


@dataclass
class PlanCache:
    """In-memory cache mapping state hashes to computed plans."""
    _cache: dict[str, tuple[dict[str, list[str]], list[dict[str, Any]]]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, state_hash: str) -> tuple[dict[str, list[str]], list[dict[str, Any]]] | None:
        if state_hash in self._cache:
            self.hits += 1
            return self._cache[state_hash]
        self.misses += 1
        return None

    def put(
        self,
        state_hash: str,
        assignments: dict[str, list[str]],
        coalitions: list[dict[str, Any]],
    ) -> None:
        self._cache[state_hash] = (dict(assignments), list(coalitions))


@dataclass
class PromptCache:
    """In-memory cache mapping prompt hashes to exact LLM response text."""
    _cache: dict[str, str] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def hash_prompt(prompt: str, system: str = "") -> str:
        content = f"SYS:{system}|PROMPT:{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def get(self, prompt: str, system: str = "") -> str | None:
        key = self.hash_prompt(prompt, system)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, prompt: str, response: str, system: str = "") -> None:
        key = self.hash_prompt(prompt, system)
        self._cache[key] = response


class StateDifferenceDetector:
    """Detects whether state displacement or CQI drift exceeds threshold epsilon."""

    def __init__(
        self,
        position_threshold: float = 5.0,
        cqi_threshold: float = 0.05,
    ):
        self.position_threshold = position_threshold
        self.cqi_threshold = cqi_threshold
        self._last_positions: dict[str, tuple[float, float]] = {}
        self._last_cqi: float = 1.0
        self._last_subtasks: set[str] = set()

    def record_state(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        sys_cqi: float = 1.0,
    ) -> None:
        self._last_positions = {a.agent_id: (a.position.x, a.position.y) for a in fleet.agents}
        self._last_cqi = sys_cqi
        self._last_subtasks = {s.subtask_id for s in subtasks if not s.completed}

    def has_material_difference(
        self,
        fleet: AgentFleet,
        subtasks: Sequence[Subtask],
        sys_cqi: float = 1.0,
    ) -> bool:
        if not self._last_positions:
            return True

        current_active = {s.subtask_id for s in subtasks if not s.completed}
        if current_active != self._last_subtasks:
            return True

        if abs(sys_cqi - self._last_cqi) > self.cqi_threshold:
            return True

        for agent in fleet.agents:
            old_pos = self._last_positions.get(agent.agent_id)
            if old_pos is None:
                return True
            dx = agent.position.x - old_pos[0]
            dy = agent.position.y - old_pos[1]
            if (dx * dx + dy * dy) ** 0.5 > self.position_threshold:
                return True

        return False
