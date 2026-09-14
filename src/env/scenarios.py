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

import math
from dataclasses import dataclass, field
from typing import Any
import zlib

import numpy as np

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
            "num_vehicle": ac.get("num_vehicle", 4),
            "num_robot": ac.get("num_robot", 3),
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

    num_uav = ac.get("num_uav", 3)
    num_vehicle = ac.get("num_vehicle", 2)
    num_robot = ac.get("num_robot", 3)

    # Compute position overrides for robot (sensor) agents.
    # Robots are the third agent group; their IDs start after UAVs + vehicles.
    # Each sensor is placed close to the target of a subtask it will monitor,
    # with a small realistic displacement so that it is not exactly on the target.
    robot_start_idx = num_uav + num_vehicle
    sensor_offset = float(ac.get("sensor_offset", 8.0))
    offset_rng = np.random.default_rng(zlib.crc32(b"inspection_sensor_offset") % 2**31 + seed)
    position_overrides = {}
    for i in range(num_robot):
        agent_id = f"robot_{robot_start_idx + i}"
        # Assign each sensor to a subtask target (cycling if more sensors
        # than subtasks, though the default config has 3 sensors / 8 tasks).
        subtask_idx = i % len(subtasks)
        target = subtasks[subtask_idx].target
        if sensor_offset > 0:
            angle = float(offset_rng.uniform(0, 2 * math.pi))
            r = float(offset_rng.uniform(sensor_offset * 0.9, sensor_offset * 1.25))
            ox = float(np.clip(target.x + r * math.cos(angle), 0.0, 200.0))
            oy = float(np.clip(target.y + r * math.sin(angle), 0.0, 200.0))
            position_overrides[agent_id] = {"x": ox, "y": oy}
        else:
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
            "num_uav": ac.get("num_uav", 3),
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


def get_fleet_capability_summary(scenario: Scenario, fleet: Any | None = None) -> dict[str, Any]:
    """Compute counts of agents with specific capabilities for a scenario.
    
    Returns:
        Dictionary containing counts of sensing, navigation, transport,
        lifting, rescue, inspection, and multi-skill agents.
    """
    from src.env.agents import AgentType, ROLE_SKILLS
    if fleet is not None and hasattr(fleet, "agents"):
        agent_list = fleet.agents
        skills_by_agent = [a.skills for a in agent_list]
    else:
        ac = scenario.agent_config
        skills_by_agent = []
        for atype, key in [
            (AgentType.UAV, "num_uav"),
            (AgentType.VEHICLE, "num_vehicle"),
            (AgentType.ROBOT, "num_robot"),
        ]:
            for _ in range(ac.get(key, 0)):
                skills_by_agent.append(ROLE_SKILLS[atype])

    total_agents = len(skills_by_agent)
    
    sensing = sum(1 for s in skills_by_agent if "sense" in s)
    navigation = sum(1 for s in skills_by_agent if "navigate" in s)
    transport = sum(1 for s in skills_by_agent if "transport" in s)
    lifting = sum(1 for s in skills_by_agent if "lift" in s)
    rescue = sum(1 for s in skills_by_agent if "rescue" in s)
    inspection = sum(1 for s in skills_by_agent if "inspect" in s)
    multi_skill = sum(1 for s in skills_by_agent if len(set(s)) > 1)

    return {
        "scenario": scenario.name,
        "total_agents": total_agents,
        "sensing_capable": sensing,
        "navigation_capable": navigation,
        "transport_capable": transport,
        "lifting_capable": lifting,
        "rescue_capable": rescue,
        "inspection_capable": inspection,
        "multi_skill_agents": multi_skill,
        "proportions": {
            "sensing": round(sensing / total_agents, 3) if total_agents else 0.0,
            "navigation": round(navigation / total_agents, 3) if total_agents else 0.0,
            "transport": round(transport / total_agents, 3) if total_agents else 0.0,
            "lifting": round(lifting / total_agents, 3) if total_agents else 0.0,
            "rescue": round(rescue / total_agents, 3) if total_agents else 0.0,
            "inspection": round(inspection / total_agents, 3) if total_agents else 0.0,
            "multi_skill": round(multi_skill / total_agents, 3) if total_agents else 0.0,
        },
    }


def validate_scenario_agent_balance(
    scenario: Scenario, fleet: Any | None = None
) -> dict[str, Any]:
    """Validate that the scenario agent fleet is balanced, feasible, and not severely skewed.

    Reports:
    - total agents
    - counts by agent type
    - capability coverage
    - number of task requirements covered by available fleet
    - whether the scenario has severe capability imbalance
    """
    from src.env.agents import AgentType, ROLE_SKILLS
    ac = scenario.agent_config
    counts_by_type = {
        "uav": ac.get("num_uav", 0),
        "vehicle": ac.get("num_vehicle", 0),
        "robot": ac.get("num_robot", 0),
    }
    total_agents = sum(counts_by_type.values())
    proportions_by_type = {
        k: round(v / total_agents, 3) if total_agents else 0.0
        for k, v in counts_by_type.items()
    }

    cap_summary = get_fleet_capability_summary(scenario, fleet)

    # Check task requirements coverage
    subtasks = scenario.subtasks
    covered_subtasks = 0
    subtask_feasibility: dict[str, bool] = {}

    if fleet is not None and hasattr(fleet, "agents"):
        agent_skills = [set(a.skills) for a in fleet.agents]
    else:
        agent_skills = []
        for atype, key in [
            (AgentType.UAV, "num_uav"),
            (AgentType.VEHICLE, "num_vehicle"),
            (AgentType.ROBOT, "num_robot"),
        ]:
            for _ in range(ac.get(key, 0)):
                agent_skills.append(set(ROLE_SKILLS[atype]))

    for st in subtasks:
        req = set(st.required_skills)
        # Check single-agent or coalition (pair) feasibility
        is_feasible = False
        # 1. Single agent covers
        for s in agent_skills:
            if req.issubset(s):
                is_feasible = True
                break
        # 2. Pair coalition covers
        if not is_feasible:
            for i in range(len(agent_skills)):
                for j in range(i + 1, len(agent_skills)):
                    if req.issubset(agent_skills[i] | agent_skills[j]):
                        is_feasible = True
                        break
                if is_feasible:
                    break
        subtask_feasibility[st.subtask_id] = is_feasible
        if is_feasible:
            covered_subtasks += 1

    coverage_rate = (
        round(covered_subtasks / len(subtasks), 3) if subtasks else 1.0
    )

    # Severe capability imbalance checks
    has_severe_imbalance = False
    imbalance_reasons: list[str] = []

    if any(count == 0 for count in counts_by_type.values()):
        has_severe_imbalance = True
        imbalance_reasons.append("One or more agent types have zero agents.")

    if coverage_rate < 1.0:
        has_severe_imbalance = True
        imbalance_reasons.append(
            f"Fleet cannot cover all task skill requirements ({covered_subtasks}/{len(subtasks)} feasible)."
        )

    # Check if any single type disproportionately dominates (> 65%) or is under-represented (< 15%)
    for atype_name, prop in proportions_by_type.items():
        if prop > 0.65:
            has_severe_imbalance = True
            imbalance_reasons.append(
                f"Agent type '{atype_name}' dominates fleet with {prop*100:.1f}% (>65%)."
            )
        elif prop < 0.15:
            has_severe_imbalance = True
            imbalance_reasons.append(
                f"Agent type '{atype_name}' is under-represented with {prop*100:.1f}% (<15%)."
            )

    return {
        "scenario": scenario.name,
        "total_agents": total_agents,
        "counts_by_type": counts_by_type,
        "proportions_by_type": proportions_by_type,
        "capability_summary": cap_summary,
        "total_subtasks": len(subtasks),
        "covered_subtasks": covered_subtasks,
        "coverage_rate": coverage_rate,
        "subtask_feasibility": subtask_feasibility,
        "has_severe_imbalance": has_severe_imbalance,
        "imbalance_reasons": imbalance_reasons,
    }


def print_scenario_agent_config(
    scenario: Scenario, fleet: Any | None = None
) -> str:
    """Print clean configuration summary at startup."""
    from src.env.agents import AgentType, ROLE_SKILLS
    ac = scenario.agent_config
    num_uav = ac.get("num_uav", 0)
    num_vehicle = ac.get("num_vehicle", 0)
    num_robot = ac.get("num_robot", 0)
    total_agents = num_uav + num_vehicle + num_robot

    uav_skills = ", ".join(ROLE_SKILLS[AgentType.UAV])
    veh_skills = ", ".join(ROLE_SKILLS[AgentType.VEHICLE])
    rob_skills = ", ".join(ROLE_SKILLS[AgentType.ROBOT])

    summary_lines = [
        f"Scenario: {scenario.name.replace('_', ' ').title()}",
        f"Total Agents: {total_agents}",
        f"UAVs: {num_uav}",
        f"Vehicles: {num_vehicle}",
        f"Robots: {num_robot}",
        "",
        "Skills:",
        f"UAV: {uav_skills}",
        f"Vehicle: {veh_skills}",
        f"Robot: {rob_skills}",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text + "\n")
    return summary_text
