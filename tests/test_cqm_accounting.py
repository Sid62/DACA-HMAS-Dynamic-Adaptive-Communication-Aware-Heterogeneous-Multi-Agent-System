"""CQM accounting invariant tests.

Verifies that the packet-loss and bandwidth CQI components are
statistically independent: a delivery failure (ack=0) affects only
the packet-loss term, while bandwidth tracks offered channel throughput
independent of per-message delivery outcomes.
"""

import numpy as np
import pytest

from src.cqm.monitor import CommunicationQualityMonitor
from src.env.network_conditions import NetworkConditionGenerator, NetworkProfile, NetworkState


@pytest.fixture
def cqm():
    return CommunicationQualityMonitor(
        weights=(0.4, 0.35, 0.25),
        tau_min=0.01,
        tau_max=2.0,
        n_nodes=2,
    )


# ── Test 1: Packet loss = 0, bandwidth normal ──────────────────────────

def test_no_loss_normal_bandwidth(cqm):
    """When all messages delivered, packet loss = 0 and bandwidth reflects
    offered throughput."""
    for _ in range(10):
        net = NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        )
        cqm.update_from_network(0, net)

    assert cqm.packet_loss_rate(0) == pytest.approx(0.0)
    bw = cqm.bandwidth_availability(0)
    assert 0.7 < bw <= 1.0  # 200/256 ~ 0.78


# ── Test 2: Packet loss increases, offered bandwidth constant ──────────

def test_loss_increases_bandwidth_independent(cqm):
    """Increasing packet loss (ack=0) must NOT affect bandwidth when
    bytes_delivered (offered throughput) remains constant."""
    # Phase 1: all delivered
    for _ in range(10):
        net = NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        )
        cqm.update_from_network(0, net)
    bw_before = cqm.bandwidth_availability(0)
    loss_before = cqm.packet_loss_rate(0)

    # Phase 2: messages lost (ack=0) but bytes_delivered unchanged
    for _ in range(10):
        net = NetworkState(
            msg_sent=1, ack_received=0, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        )
        cqm.update_from_network(0, net)
    bw_after = cqm.bandwidth_availability(0)
    loss_after = cqm.packet_loss_rate(0)

    # Packet loss must have increased
    assert loss_after > loss_before
    # Bandwidth must remain the same (same offered throughput)
    assert bw_after == pytest.approx(bw_before, abs=0.01)


# ── Test 3: Bandwidth decreases without packet-loss change ─────────────

def test_bandwidth_decrease_no_loss_change(cqm):
    """Bandwidth can degrade independently of packet loss."""
    for _ in range(10):
        net = NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        )
        cqm.update_from_network(0, net)
    bw_before = cqm.bandwidth_availability(0)

    # Lower offered throughput (channel congestion), but still delivered
    for _ in range(10):
        net = NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=50.0, bytes_capacity=256.0,
        )
        cqm.update_from_network(0, net)
    bw_after = cqm.bandwidth_availability(0)

    assert bw_after < bw_before
    assert cqm.packet_loss_rate(0) == pytest.approx(0.0)


# ── Test 4: Packet loss and bandwidth are distinguishable ──────────────

def test_loss_and_bandwidth_distinguishable(cqm):
    """CQI must differ between (high loss, good BW) and (no loss, low BW)."""
    # Node 0: high loss, good bandwidth
    for _ in range(10):
        cqm.update_from_network(0, NetworkState(
            msg_sent=1, ack_received=0, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        ))
    # Node 1: no loss, low bandwidth
    for _ in range(10):
        cqm.update_from_network(1, NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=30.0, bytes_capacity=256.0,
        ))

    cqi_0 = cqm.node_cqi(0)
    cqi_1 = cqm.node_cqi(1)
    # Both degraded, but by different components - different values
    assert cqi_0 != pytest.approx(cqi_1, abs=0.05)


# ── Test 5: CQI remains normalized ────────────────────────────────────

def test_cqi_normalized(cqm):
    """CQI must remain in [0, 1] under all conditions."""
    # Worst case: total loss, max latency, zero bandwidth
    for _ in range(20):
        cqm.update_from_network(0, NetworkState(
            msg_sent=1, ack_received=0, latency=2.0,
            bytes_delivered=0.0, bytes_capacity=256.0,
        ))
    assert 0.0 <= cqm.node_cqi(0) <= 1.0

    # Best case: perfect link
    for _ in range(20):
        cqm.update_from_network(1, NetworkState(
            msg_sent=1, ack_received=1, latency=0.01,
            bytes_delivered=256.0, bytes_capacity=256.0,
        ))
    assert 0.0 <= cqm.node_cqi(1) <= 1.0
    assert 0.0 <= cqm.system_cqi() <= 1.0


# ── Test 6: CQI responds monotonically ────────────────────────────────

def test_cqi_monotonic_with_degradation(cqm):
    """CQI must decrease as conditions worsen."""
    # Good conditions
    for _ in range(10):
        cqm.update_from_network(0, NetworkState(
            msg_sent=1, ack_received=1, latency=0.05,
            bytes_delivered=200.0, bytes_capacity=256.0,
        ))
    cqi_good = cqm.node_cqi(0)

    # Bad conditions (higher latency, lower BW, some loss)
    for _ in range(10):
        cqm.update_from_network(0, NetworkState(
            msg_sent=1, ack_received=0, latency=1.5,
            bytes_delivered=50.0, bytes_capacity=256.0,
        ))
    cqi_bad = cqm.node_cqi(0)

    assert cqi_bad < cqi_good


# ── Test 7: ACDS receives CQI without interface changes ───────────────

def test_acds_receives_system_cqi():
    """system_cqi() return value is a float suitable for ACDS.evaluate()."""
    from src.acds.switch_engine import ACDSSwitchEngine
    cqm = CommunicationQualityMonitor(n_nodes=2)
    for _ in range(5):
        cqm.update_from_network(0, NetworkState(
            msg_sent=1, ack_received=1, latency=0.1,
            bytes_delivered=200.0, bytes_capacity=256.0,
        ))
    sys_cqi = cqm.system_cqi()
    acds = ACDSSwitchEngine(theta_down=0.5, theta_up=0.75)
    # Should not raise
    mode = acds.evaluate(sys_cqi, current_step=0)
    assert mode in (0, 1)


# ── Test 8: simulate_message bytes_delivered decoupled from ack ────────

def test_simulate_message_bytes_decoupled():
    """bytes_delivered must equal offered throughput regardless of ack."""
    gen = NetworkConditionGenerator(
        profile=NetworkProfile.OSCILLATORY,
        rng=np.random.default_rng(42),
    )
    for t in range(50):
        state = gen.simulate_message(t)
        assert state.bytes_delivered >= 0.0
        assert state.bytes_capacity > 0.0

    # Specifically test: force a loss scenario and verify bytes_delivered
    # is non-zero even on loss (it represents offered throughput)
    gen2 = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        base_delay_prob=0.0,
        base_loss_rate=0.0,
        rng=np.random.default_rng(0),
    )
    for t in range(100):
        state = gen2.simulate_message(t)
        if state.ack_received == 0:
            # Critical: bytes_delivered must NOT be zeroed by loss
            assert state.bytes_delivered > 0.0, (
                f"bytes_delivered was zeroed on ack=0 at step {t} -- "
                f"double-counting bug is present"
            )
