"""Heterogeneous agent definitions and kinematics (Eqs 1-7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class AgentType(str, Enum):
    UAV = "uav"
    VEHICLE = "vehicle"
    ROBOT = "robot"


@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class AgentState:
    agent_id: str
    agent_type: AgentType
    position: Position
    heading: float = 0.0
    speed: float = 0.0
    skills: list[str] = field(default_factory=list)
    assigned_subtasks: list[str] = field(default_factory=list)
    completed_subtasks: list[str] = field(default_factory=list)
    remaining_waypoints: list[Position] = field(default_factory=list)
    coalition_id: int | None = None
    # Goal 2: lightweight physical properties (no charging/RTB/planning logic attached)
    battery: float = 100.0
    communication_range: float = 50.0
    sensor_range: float = 30.0

def dist(p1: Position | np.ndarray, p2: Position | np.ndarray) -> float:
    """Euclidean distance (Eqs 4-5)."""
    a = p1.as_array() if isinstance(p1, Position) else np.asarray(p1)
    b = p2.as_array() if isinstance(p2, Position) else np.asarray(p2)
    return float(np.linalg.norm(a - b))


def distance_matrix(agents: Sequence[AgentState]) -> np.ndarray:
    """N x N inter-agent distance matrix D(t)."""
    n = len(agents)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = dist(agents[i].position, agents[j].position)
    return d


@dataclass
class KinematicsConfig:
    max_speed: float
    max_turn_rate: float


def _load_battery_config() -> tuple[bool, float, float, float]:
    """Optional battery-drain config (Goal 2)."""

    try:
        from src.config import get_thresholds
        cfg = get_thresholds().get("battery", {})
    except Exception:
        cfg = {}

    enabled = bool(cfg.get("enabled", False))
    drain_rate = float(cfg.get("drain_rate", 0.0))
    low_threshold = float(cfg.get("low_threshold", 20.0))
    low_speed_factor = float(cfg.get("low_speed_factor", 1.0))

    return enabled, drain_rate, low_threshold, low_speed_factor

class AgentFleet:
    """Manages heterogeneous agent fleet evolution."""

    def __init__(
        self,
        agents: list[AgentState],
        kinematics: dict[str, KinematicsConfig],
        c1: float = 50.0,
        c2: float = 5.0,

    ):
        self.agents = agents
        self.kinematics = kinematics
        self.c1 = c1
        self.c2 = c2
        self._id_to_idx = {a.agent_id: i for i, a in enumerate(agents)}
        (
            self._battery_enabled,
            self._battery_drain_rate,
            self._low_battery_threshold,
            self._low_battery_speed_factor,
       ) = _load_battery_config()

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def get_agent(self, agent_id: str) -> AgentState:
        return self.agents[self._id_to_idx[agent_id]]

    def step_toward(
        self, agent_id: str, target: Position, dt: float = 0.1
    ) -> None:
        agent = self.get_agent(agent_id)
        kcfg = self.kinematics[agent.agent_type.value]
        dx = target.x - agent.position.x
        dy = target.y - agent.position.y
        desired_heading = math.atan2(dy, dx)
        heading_diff = desired_heading - agent.heading
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        max_turn = kcfg.max_turn_rate * dt
        if abs(heading_diff) > max_turn:
            agent.heading += math.copysign(max_turn, heading_diff)
        else:
            agent.heading = desired_heading
        max_speed = kcfg.max_speed
        if self._battery_enabled and agent.battery < self._low_battery_threshold:
            max_speed *= self._low_battery_speed_factor
        agent.speed = min(max_speed, math.hypot(dx, dy))
        agent.position.x += agent.speed * math.cos(agent.heading) * dt
        agent.position.y += agent.speed * math.sin(agent.heading) * dt
        if self._battery_enabled:
            moved = agent.speed * dt
            agent.battery = max(0.0, agent.battery - moved * self._battery_drain_rate)
        print(
            f"[MOVE] {agent.agent_id} "
            f"Pos=({agent.position.x:.2f},{agent.position.y:.2f}) "
            f"Speed={agent.speed:.2f} "
            f"Battery={agent.battery:.1f}"
       )

    def check_proximity_constraint(self) -> bool:
        """Eq 6: inter-team proximity constraint g."""
        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i < j and a.agent_type != b.agent_type:
                    if dist(a.position, b.position) < self.c2:
                        return False
        return True

    def check_communication_range(self) -> bool:
        """Eq 7: communication range constraint k."""
        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if i != j and dist(a.position, b.position) > self.c1:
                    return False
        return True

    def to_dict_list(self) -> list[dict]:
        return [
            {
                "id": a.agent_id,
                "type": a.agent_type.value,
                "position": [a.position.x, a.position.y, a.position.z],
                "skills": a.skills,
                "coalition_id": a.coalition_id,
            }
            for a in self.agents
        ]


def create_fleet_from_scenario(
    scenario_cfg: dict,
    kinematics_cfg: dict,
    c1: float,
    c2: float,
    seed: int = 0,
) -> AgentFleet:
    """Instantiate heterogeneous fleet for a scenario."""
    rng = np.random.default_rng(seed)
    agents: list[AgentState] = []
    skill_pool = ["transport", "inspect", "lift", "navigate", "sense", "rescue"]
    idx = 0
    for agent_type, count_key in [
        (AgentType.UAV, "num_uav"),
        (AgentType.VEHICLE, "num_vehicle"),
        (AgentType.ROBOT, "num_robot"),
    ]:
        count = scenario_cfg.get(count_key, 0)
        kcfg = kinematics_cfg.get(agent_type.value, {"max_speed": 5.0, "max_turn_rate": 1.0})
        for _ in range(count):
            agents.append(
                AgentState(
                    agent_id=f"{agent_type.value}_{idx}",
                    agent_type=agent_type,
                    position=Position(
                        x=float(rng.uniform(0, 200)),
                        y=float(rng.uniform(0, 200)),
                    ),
                    heading=float(rng.uniform(0, 2 * math.pi)),
                    skills=list(rng.choice(skill_pool, size=2, replace=False)),
                )
            )
            idx += 1
    kin = {
        k: KinematicsConfig(**v) for k, v in kinematics_cfg.items()
    }
    return AgentFleet(agents, kin, c1=c1, c2=c2)
