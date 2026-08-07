"""Mission scenario definitions (logistics, inspection, search_rescue).

Scenario semantics are derived from AutoHMA-LLM Section V-A:
- Logistics: warehouse/distribution with UAVs, vehicles, warehouse robots;
  environment includes 5-60% lane occupancy from background traffic.
- Inspection: facility inspection with UAVs (aerial patrol), inspection robots
  (ground mobile), and sensors (fixed at a specific location);
  10% communication delay, 1% terminal data loss.
- Search & Rescue: maritime SAR with UAVs (aerial search), ships (maritime
  rescue assets), and rescue robots (extraction); rough-sea, irregular-wind
  environment (qualitative only — no implementable physics model given).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import zlib

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
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_subtasks(
    name: str,
    count: int,
    skill_sets: list[list[str]],
    seed_offset: int = 0,
    priority_update_probability: float = 0.0,
) -> list[Subtask]:
    import numpy as np

    rng = np.random.default_rng(zlib.crc32(name.encode()) % 2**31 + seed_offset)
    subtasks = []
    for j in range(count):
        st = Subtask(
            subtask_id=f"T_{j}",
            description=f"{name} subtask {j}",
            target=Position(
                x=float(rng.uniform(20, 180)),
                y=float(rng.uniform(20, 180)),
            ),
            required_skills=skill_sets[j % len(skill_sets)],
        )
        # Goal 3: occasional, small, deterministic priority variation
        if rng.random() < priority_update_probability:
            st.priority = float(np.clip(st.priority + rng.uniform(-0.1, 0.1), 0.05, 0.95))
        subtasks.append(st)
    return subtasks


def build_logistics_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    """Logistics scenario: warehouse/distribution operation.

    Agent roles per AutoHMA-LLM Section V-A:
    - UAVs: aerial delivery/transport units
    - Vehicles: ground-based transport units (autonomous vehicles)
    - Robots: warehouse robots operating within the facility

    Environment: complex traffic environment with background (non-mission)
    cars at 5-60% lane occupancy. traffic_lane_occupancy is sampled uniformly
    in [0.05, 0.60] and stored as inert metadata — no traffic simulation is
    implemented because the paper provides no algorithm for it.
    """
    import numpy as np

    ac = cfg.get("scenarios", {}).get("logistics", cfg)
    subtasks = _make_subtasks(
        "logistics",
        ac.get("num_subtasks", 6),
        [["transport", "navigate"], ["lift", "transport"], ["navigate", "sense"]],
        seed,
        cfg.get("scenario", {}).get("priority_update_probability", 0.0),
    )

    # Sample traffic_lane_occupancy AFTER all subtask generation, using a
    # completely separate RNG stream to guarantee zero interference with
    # existing subtask target coordinates.
    meta_rng = np.random.default_rng(zlib.crc32(b"logistics_meta") % 2**31 + seed)
    traffic_lane_occupancy = float(meta_rng.uniform(0.05, 0.60))

    return Scenario(
        name="logistics",
        instruction="Coordinate UAVs, vehicles, and robots to deliver packages across the warehouse zone.",
        subtasks=subtasks,
        agent_config={
            "num_uav": ac.get("num_uav", 3),
            "num_vehicle": ac.get("num_vehicle", 2),
            "num_robot": ac.get("num_robot", 2),
        },
        comm_delay_prob=ac.get("comm_delay_prob", 0.0),
        packet_loss_rate=ac.get("packet_loss_rate", 0.0),
        metadata={
            "traffic_lane_occupancy": traffic_lane_occupancy,
            "agent_roles": {
                "uav": "Aerial delivery/transport units",
                "vehicle": "Ground-based autonomous transport vehicles",
                "robot": "Warehouse robots operating within the facility",
            },
        },
    )


def build_inspection_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    """Inspection scenario: facility inspection and monitoring.

    Agent roles per AutoHMA-LLM Section V-A:
    - UAVs: aerial inspectors flying a patrol path ("the inspection task")
    - Vehicles: ground-based inspection robots approaching equipment
      ("the equipment inspection task" — distinct from UAV's broader patrol)
    - Robots: SENSORS — fixed at a specific location, not mobile.
      The paper explicitly contrasts these stationary agents against the
      mobile UAVs and inspection robots. They "continue monitoring the
      environment" without a discrete completion point.

    Sensor fidelity proxy: robot agents are initialized at (or within
    negligible offset of) their assigned subtask target positions, so
    under existing movement logic they require effectively zero travel.
    No 'is_stationary' flag is added — this is a position-initialization
    proxy only.

    Communication: 10% delay probability, 1% terminal data loss
    (explicitly quantified by the paper for this scenario).
    """
    ac = cfg.get("scenarios", {}).get("inspection", cfg)
    subtasks = _make_subtasks(
        "inspection",
        ac.get("num_subtasks", 8),
        [["inspect", "sense"], ["navigate", "inspect"], ["sense", "lift"]],
        seed,
    )

    num_uav = ac.get("num_uav", 4)
    num_vehicle = ac.get("num_vehicle", 2)
    num_robot = ac.get("num_robot", 3)

    # Compute position overrides for robot (sensor) agents.
    # Robots are the third agent group; their IDs start after UAVs + vehicles.
    # Each sensor is placed at the target of a subtask it will monitor.
    robot_start_idx = num_uav + num_vehicle
    position_overrides = {}
    for i in range(num_robot):
        agent_id = f"robot_{robot_start_idx + i}"
        # Assign each sensor to a subtask target (cycling if more sensors
        # than subtasks, though the default config has 3 sensors / 8 tasks).
        subtask_idx = i % len(subtasks)
        target = subtasks[subtask_idx].target
        position_overrides[agent_id] = {"x": target.x, "y": target.y}

    return Scenario(
        name="inspection",
        instruction="Inspect infrastructure across distributed sites with heterogeneous agents under degraded communication.",
        subtasks=subtasks,
        agent_config={
            "num_uav": num_uav,
            "num_vehicle": num_vehicle,
            "num_robot": num_robot,
            "_position_overrides": position_overrides,
        },
        comm_delay_prob=ac.get("comm_delay_prob", 0.10),
        packet_loss_rate=ac.get("packet_loss_rate", 0.01),
        metadata={
            "agent_roles": {
                "uav": "Aerial inspectors flying patrol path (inspection task)",
                "vehicle": "Ground-based inspection robots approaching equipment (equipment inspection task)",
                "robot": "Sensors — fixed at a specific location, continuously monitoring the environment",
            },
            "comm_conditions": "10% communication delay, 1% terminal data loss (Section V-A)",
        },
    )


def build_search_rescue_scenario(cfg: dict[str, Any], seed: int = 0) -> Scenario:
    """Search & Rescue scenario: maritime SAR operation.

    Agent roles per AutoHMA-LLM Section V-A:
    - UAVs: aerial search — fly patrol/search path to locate target(s)
    - Vehicles: conceptually represent SHIPS (maritime rescue assets);
      navigate water-surface path to approach rescue position.
      The paper uses two verbs for the ship's progress: "approaches"
      then "arrives at" — an explicit two-stage action (transit, arrival).
    - Robots: rescue robots — carry out physical rescue/extraction once
      target is located and reached; first "moves to designated area,"
      then "completes the rescue task" (action follows arrival).

    Environment: maritime rough-sea with irregular-wind conditions
    (qualitative only — the paper gives no formula, wave-height range,
    wind-speed distribution, or algorithm). No physics simulation is
    implemented; this is recorded as semantic metadata only.

    No communication-degradation numbers are given for this scenario
    in the paper (unlike Inspection's explicit 10%/1% figures).
    """
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
        metadata={
            "environment_description": (
                "Maritime rough-sea environment with winds in irregular "
                "directions, simulating substantial interference at sea. "
                "This is a qualitative description from the paper — no "
                "implementable physics model (wave equations, wind "
                "distributions) is provided or implemented."
            ),
            "agent_roles": {
                "uav": "Aerial search units — fly patrol/search path to locate targets",
                "vehicle": "Ships (maritime rescue assets) — navigate sea to approach rescue position",
                "robot": "Rescue robots — carry out physical rescue/extraction at designated area",
            },
        },
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
