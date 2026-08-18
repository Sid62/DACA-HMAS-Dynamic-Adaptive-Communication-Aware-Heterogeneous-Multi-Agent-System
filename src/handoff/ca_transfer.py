"""Mode-aware collision avoidance transfer (Eq 30)."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.control.nmpc import NMPCController
from src.control.q_learning import QLearningCA
from src.env.agents import AgentFleet, Position


@dataclass
class CATransferManager:
    """Ensures Phi(t) = 1 throughout architecture transitions."""

    nmpc: NMPCController = field(default_factory=NMPCController)
    q_learning: QLearningCA = field(default_factory=QLearningCA)
    overlap_delta: int = 3
    overlap_remaining: int = 0
    transitioning: bool = False

    def activate_overlap_window(self, delta: int | None = None) -> None:
        self.overlap_remaining = delta if delta is not None else self.overlap_delta
        self.transitioning = True
        self.q_learning.activate()
        self.nmpc.pid.reset()

    def step(
        self,
        fleet: AgentFleet,
        mode: int,
        assignments: dict[str, str],
        targets: dict[str, Position],
    ) -> None:
        in_overlap = self.overlap_remaining > 0

        if mode == 0 or in_overlap:
            self.nmpc.step(fleet, assignments, targets)

        if mode == 1 or in_overlap:
            self.q_learning.step(fleet, assignments, targets)
        elif mode == 0:
            self.q_learning.warmup_step(fleet)

        if in_overlap:
            self.overlap_remaining -= 1
            if self.overlap_remaining <= 0:
                self.transitioning = False
                if mode == 0:
                    self.q_learning.standby_mode()
                else:
                    self.q_learning.activate()

    def phi_active(self, mode: int) -> bool:
        """Eq 30: at least one CA mechanism active."""
        global_ca = mode == 0 or self.transitioning
        local_ca = mode == 1 or self.transitioning or self.q_learning.active
        return global_ca or local_ca

    def on_mode_change(self, new_mode: int) -> None:
        if new_mode == 1:
            self.q_learning.activate()
        self.activate_overlap_window()
