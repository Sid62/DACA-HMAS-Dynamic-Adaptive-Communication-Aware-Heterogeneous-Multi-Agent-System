"""Synthetic network condition injectors for communication profiles."""
 
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
 
import numpy as np
 
from src.env.agents import distance_matrix
from src.env.network_model import (
    distance_quality,
    environment_factor,
    in_any_window,
    interference_windows,
    wireless_jitter,
)

# Lightweight, fixed mapping from scenario to a plausible deployment
# environment (Goal 1). Not user-facing config since it's inherent to
# what each scenario represents.
_SCENARIO_ENVIRONMENT = {
    "logistics": "warehouse",
    "inspection": "urban",
    "search_rescue": "disaster",
}

 
class NetworkProfile(str, Enum):
    STABLE = "stable"
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    OSCILLATORY = "oscillatory"
 
 
@dataclass
class NetworkState:
    packet_loss_rate: float = 0.0
    latency: float = 0.01
    bandwidth_utilization: float = 0.1
    bytes_capacity: float = 10000.0
    bytes_delivered: float = 1000.0
    msg_sent: int = 0
    ack_received: int = 0
 
 
@dataclass
class NetworkConditionGenerator:
    """Inject synthetic packet loss, latency, bandwidth degradation."""
 
    profile: NetworkProfile = NetworkProfile.STABLE
    base_delay_prob: float = 0.0
    base_loss_rate: float = 0.0
    total_steps: int = 500
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    # --- Goal 1: lightweight realism additions (all optional/additive) ---
    fleet: Any | None = None
    communication_range: float = 50.0
    good_range: float = 20.0
    medium_range: float = 40.0
    wireless_jitter_scale: float = 0.05
    environment: str = "warehouse"
    environment_factors: dict[str, float] = field(
        default_factory=lambda: {"warehouse": 1.0, "urban": 1.15, "disaster": 1.35}
    )
    _comm_interference_windows: list[tuple[int, int]] = field(default_factory=list)
    # --- Bounded oscillatory channel-quality model (replaces unbounded
    # additive degradation stacking). All bounded in [0,1] to prevent any
    # single factor from permanently dominating the others.
    oscillation_period: float = 60.0       # steps per congestion/recovery cycle
    base_quality: float = 0.75             # mean channel quality (dimensionless)
    quality_amplitude: float = 0.20        # macro-cycle swing around base_quality
    distance_quality_floor: float = 0.5    # multi-hop relay floor -- distance
                                            # attenuates but never fully severs
    interference_dip_factor: float = 0.6   # temporary multiplicative dip, self-clearing
    fading_correlation: float = 0.85       # AR(1) coherence across steps
    _fading_state: float = field(default=0.0, init=False, repr=False)

    def _distance_degradation(self) -> float | None:
       """1 - link quality in [0, 1] from average inter-agent distance.
       Returns None when no fleet is attached, leaving old behaviour intact.
       """
       if self.fleet is None or getattr(self.fleet, "n_agents", 0) < 2:
           return None
       d = distance_matrix(self.fleet.agents)
       n = d.shape[0]
       pair_dists = [d[i, j] for i in range(n) for j in range(i + 1, n)]
       if not pair_dists:
           return None
       avg_dist = float(np.mean(pair_dists))
       quality = distance_quality(
           avg_dist, self.communication_range, self.good_range, self.medium_range
       )
       return 1.0 - quality

    def _channel_quality(self, t: int) -> float:
        """Single bounded latent channel-quality variable Q(t) in [0,1],
        1.0 = perfect link. Every downstream metric (loss, latency,
        bandwidth) is derived from this ONE quantity, so degradation
        factors modulate a shared physical cause instead of stacking as
        independent, unbounded penalties (Eqs are unaffected -- this only
        changes how synthetic network conditions are generated upstream
        of CQM's own computation).
        """
        # 1) Macro oscillatory cycle: congestion builds and recovers over
        # `oscillation_period` steps. This is the DOMINANT, bounded signal
        # -- profile selects how strongly/abruptly it behaves, but the
        # amplitude is fixed so no profile can push quality outside a
        # realistic band on its own.
        if self.profile == NetworkProfile.STABLE:
            osc = 0.0
        elif self.profile == NetworkProfile.GRADUAL:
            osc = np.sin(2 * np.pi * t / self.oscillation_period)
        elif self.profile == NetworkProfile.SUDDEN:
            # A single congestion episode mid-mission, then recovery --
            # bounded step, not a permanent floor shift.
            in_episode = self.total_steps * 0.35 < t < self.total_steps * 0.65
            osc = -0.8 if in_episode else 0.6
        elif self.profile == NetworkProfile.OSCILLATORY:
            osc = np.sin(2 * np.pi * t / self.oscillation_period)
        else:
            osc = 0.0
        q_base = float(np.clip(self.base_quality + self.quality_amplitude * osc, 0.05, 0.98))

        # 2) Scenario baseline (per-scenario packet_loss_rate/comm_delay_prob)
        # contributes a small, BOUNDED downward pressure -- not stacked on
        # top, folded into the same base term so scenarios remain
        # differentiable without dominating the oscillation.
        q_base = float(np.clip(q_base - 0.5 * self.base_loss_rate, 0.05, 0.98))

        # 3) Distance: multiplicative modulator centered around 1.0. Physically,
        # distance attenuates quality based on inter-agent spread relative to range,
        # modulating around baseline without pinning peak channel quality.
        dist_deg = self._distance_degradation()
        if dist_deg is None:
            q_dist = 1.0
        else:
            q_dist = 1.0 - 0.5 * dist_deg

        # 4) Environment: bounded multiplicative factor centered around 1.0.
        env_mult = environment_factor(self.environment, self.environment_factors)
        q_env = 1.0 / max(env_mult, 1.0)

        # 5) Interference: a TEMPORARY multiplicative dip while the window
        # is active, then self-clears -- matches real intermittent RF
        # interference (e.g. co-channel equipment cycling on/off) rather
        # than an additive step that never recovers.
        q_interf = self.interference_dip_factor if in_any_window(t, self._comm_interference_windows) else 1.0

        # 6) Small-scale (fast) fading: AR(1) mean-reverting process, the
        # standard discrete-time approximation of Rayleigh/Rician fading
        # with a finite coherence time. Bounded by construction (reverts
        # toward 0 each step) -- unlike one-shot additive jitter, it
        # cannot accumulate a permanent drift in either direction.
        noise = self.rng.normal(0.0, 1.0)
        self._fading_state = (
            self.fading_correlation * self._fading_state
            + np.sqrt(max(1.0 - self.fading_correlation ** 2, 0.0)) * noise
        )
        fading = self.wireless_jitter_scale * self._fading_state

        q = q_base * q_dist * q_env * q_interf + fading
        return float(np.clip(q, 0.05, 0.98))

    def loss_rate_at(self, t: int) -> float:
        q = self._channel_quality(t)
        loss = 1.0 - q
        return float(np.clip(loss, 0.0, 0.95))
 
    def latency_at(self, t: int) -> float:
        q = self._channel_quality(t)
        base = 0.01 + self.base_delay_prob * 0.5
        # Latency driven by the SAME channel quality that drives loss --
        # physically both are downstream of the same SNR/throughput
        # bottleneck, not independently-degrading quantities. Small
        # independent measurement noise keeps latency from being a pure
        # deterministic function of loss.
        latency = base + (1.0 - q) * 1.5 + self.rng.uniform(0, 0.02)
        return float(max(latency, 0.0))

    def bandwidth_at(self, t: int) -> float:
        q = self._channel_quality(t)
        bw = q - self.base_delay_prob * 0.3
        return float(np.clip(bw, 0.10, 1.0))
 
    def simulate_message(
        self, t: int, payload_bytes: float = 256.0
     ) -> NetworkState:
        loss = self.loss_rate_at(t)
        latency = self.latency_at(t)
        bw_avail = self.bandwidth_at(t)
        delivered = payload_bytes * bw_avail
        msg_sent = 1
        ack = 0 if self.rng.random() < loss else 1
        return NetworkState(
            packet_loss_rate=loss,
            latency=latency,
            bandwidth_utilization=1.0 - bw_avail,
            bytes_capacity=payload_bytes,
            bytes_delivered=delivered if ack else 0.0,
            msg_sent=msg_sent,
            ack_received=ack,
        )

 
    @classmethod
    def from_scenario(
        cls,
        scenario_name: str,
        profile: str,
        thresholds: dict[str, Any],
        seed: int = 0,
        total_steps: int = 500,
    ) ->NetworkConditionGenerator:
        sc = thresholds.get("scenarios", {}).get(scenario_name, {})
        prof = NetworkProfile(profile) if profile != "stable" else NetworkProfile.STABLE
        if scenario_name == "logistics" and profile == "stable":
            prof = NetworkProfile.STABLE

        net_cfg = thresholds.get("network", {})
        scenario_cfg = thresholds.get("scenario", {})
        rng = np.random.default_rng(seed)
        windows = (
            interference_windows(rng, total_steps)
            if scenario_cfg.get("communication_events", False)
            else []
        )
 
        return cls(
            profile=prof,
            base_delay_prob=sc.get("comm_delay_prob", 0.0),
            base_loss_rate=sc.get("packet_loss_rate", 0.0),
            total_steps=total_steps,
            rng=rng,
            communication_range=net_cfg.get("communication_range", 50.0),
            good_range=net_cfg.get("good_range", 20.0),
            medium_range=net_cfg.get("medium_range", 40.0),
            wireless_jitter_scale=net_cfg.get("wireless_jitter", 0.05),
            environment=_SCENARIO_ENVIRONMENT.get(scenario_name, "warehouse"),
            environment_factors=net_cfg.get(
                "environment_factor", {"warehouse": 1.0, "urban": 1.15, "disaster": 1.35}
            ),
            _comm_interference_windows=windows,
            oscillation_period=net_cfg.get("oscillation_period", 60.0),
            base_quality=net_cfg.get("base_quality", 0.75),
            quality_amplitude=net_cfg.get("quality_amplitude", 0.20),
            distance_quality_floor=net_cfg.get("distance_quality_floor", 0.5),
            interference_dip_factor=net_cfg.get("interference_dip_factor", 0.6),
            fading_correlation=net_cfg.get("fading_correlation", 0.85),
        )