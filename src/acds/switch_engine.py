"""ACDS dual-threshold hysteresis switching engine (Eqs 20-22)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ACDSSwitchEngine:
    """Architecture Control and Decision Switching with hysteresis."""

    theta_down: float = 0.57
    theta_up: float = 0.73
    persistence_window: int = 5
    use_hysteresis: bool = True
    mode: int = 0
    cqi_history: deque = field(default_factory=lambda: deque(maxlen=20))
    switch_count: int = 0

    @classmethod
    def from_config(cls, thresholds: dict[str, Any], use_hysteresis: bool = True) -> ACDSSwitchEngine:
        acds = thresholds.get("acds", {})
        crossover = acds.get("cqi_crossover", 0.65)
        delta = acds.get("delta", 0.08)
        return cls(
            theta_down=crossover - delta,
            theta_up=crossover + delta,
            persistence_window=acds.get("persistence_window", 5),
            use_hysteresis=use_hysteresis,
        )

    def record_cqi(self, cqi: float) -> None:
        self.cqi_history.append(cqi)

    def evaluate(self, cqi: float) -> int:
        """Eq 20: mode switching with dual-threshold hysteresis."""
        self.record_cqi(cqi)
        n = self.persistence_window
        if len(self.cqi_history) < n:
            return self.mode

        recent = list(self.cqi_history)[-n:]
        new_mode = self.mode

        if self.mode == 0:
            threshold = self.theta_down if self.use_hysteresis else (self.theta_down + self.theta_up) / 2
            if all(c < threshold for c in recent):
                new_mode = 1
        elif self.mode == 1:
            threshold = self.theta_up if self.use_hysteresis else (self.theta_down + self.theta_up) / 2
            if all(c > threshold for c in recent):
                new_mode = 0

        if new_mode != self.mode:
            self.switch_count += 1
            self.mode = new_mode
        return self.mode

    def switch_count_metric(self) -> int:
        """Eq 22: SC."""
        return self.switch_count

    @property
    def is_centralized(self) -> bool:
        return self.mode == 0

    @property
    def is_decentralized(self) -> bool:
        return self.mode == 1
