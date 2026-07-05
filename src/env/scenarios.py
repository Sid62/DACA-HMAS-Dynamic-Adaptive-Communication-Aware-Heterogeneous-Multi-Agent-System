"""Mission scenario definitions (logistics, inspection, search_rescue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.agents import Position


@dataclass
class Subtask:
    subtask_id: str
    description: str
    target: Position
    required_skills: list[str]
    assigned_agents: list[str] = field(default_factory=list)
    completed: bool = False
    priority: float = 0.5


@dataclass
class Scenario:
    name: str
    instruction: str
    subtasks: list[Subtask]
    agent_config: dict[str, int]
    comm_delay_prob: float = 0.0
    packet_loss_rate: float = 0.0


def _make_subtasks(
    name: str,
    count: int,
    skill_sets: list[list[str]],
    seed_offset: int = 0,
) -> list[Subtask]:
    import numpy as np

    rng = np.random.default_rng(hash(name) % 2**31 + seed_offset)
    subtasks = []
    for j in range(count):
        subtasks.append(
            Subtask(
                subtask_id=f"T_{j}",
                description=f"{name} subtask {j}",
                target=Position(
                    x=float(rng.uniform(20, 180)),
                    y=float(rng.uniform(20, 180)),
                ),
                required_skills=skill_sets[j % len(skill_sets)],
            )
        )
    return subtasks


def build_logistics_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    ac = cfg.get("scenarios", {}).get("logistics", cfg)
    return Scenario(
        name="logistics",
        instruction="Coordinate UAVs, vehicles, and robots to deliver packages across the warehouse zone.",
        subtasks=_make_subtasks(
            "logistics",
            ac.get("num_subtasks", 6),
            [["transport", "navigate"], ["lift", "transport"], ["navigate", "sense"]],
            seed,
        ),
        agent_config={
            "num_uav": ac.get("num_uav", 3),
            "num_vehicle": ac.get("num_vehicle", 2),
            "num_robot": ac.get("num_robot", 2),
        },
        comm_delay_prob=ac.get("comm_delay_prob", 0.0),
        packet_loss_rate=ac.get("packet_loss_rate", 0.0),
    )


def build_inspection_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    ac = cfg.get("scenarios", {}).get("inspection", cfg)
    return Scenario(
        name="inspection",
        instruction="Inspect infrastructure across distributed sites with heterogeneous agents under degraded communication.",
        subtasks=_make_subtasks(
            "inspection",
            ac.get("num_subtasks", 8),
            [["inspect", "sense"], ["navigate", "inspect"], ["sense", "lift"]],
            seed,
        ),
        agent_config={
            "num_uav": ac.get("num_uav", 4),
            "num_vehicle": ac.get("num_vehicle", 2),
            "num_robot": ac.get("num_robot", 3),
        },
        comm_delay_prob=ac.get("comm_delay_prob", 0.10),
        packet_loss_rate=ac.get("packet_loss_rate", 0.01),
    )


def build_search_rescue_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    ac = cfg.get("scenarios", {}).get("search_rescue", cfg)
    subtasks = _make_subtasks(
        "search_rescue",
        ac.get("num_subtasks", 10),
        [["rescue", "lift"], ["sense", "navigate"], ["rescue", "transport"]],
        seed,
    )
    for i, st in enumerate(subtasks[:3]):
        st.priority = 0.9 - i * 0.1
    return Scenario(
        name="search_rescue",
        instruction="Search and rescue operation: locate and extract persons from disaster zone.",
        subtasks=subtasks,
        agent_config={
            "num_uav": ac.get("num_uav", 5),
            "num_vehicle": ac.get("num_vehicle", 3),
            "num_robot": ac.get("num_robot", 4),
        },
        comm_delay_prob=ac.get("comm_delay_prob", 0.05),
        packet_loss_rate=ac.get("packet_loss_rate", 0.005),
    )


SCENARIO_BUILDERS = {
    "logistics": build_logistics_scenario,
    "inspection": build_inspection_scenario,
    "search_rescue": build_search_rescue_scenario,
}


def get_scenario(name: str, thresholds: dict[str, Any], seed: int = 0) -> Scenario:
    builder = SCENARIO_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown scenario: {name}")
    return builder(thresholds, seed)
