"""Tests for Issue 4: Per-link pairwise Communication Quality Monitoring (CQM).

Verifies that:
1. Different node/link conditions produce different pairwise CQIs.
2. A weak link receives lower CQI than a strong link.
3. [1.0, 0.0, 1.0] does not produce 0.667 for every connected pair.
4. Out-of-range/disconnected pairs remain invalid/zero as appropriate.
5. Pairwise CQI is actually used by coalition feasibility/formation.
6. System-average CQI remains available separately for switching/global logic.
7. Directional links are preserved when directional observations exist.
8. NetworkConditionGenerator provides link-specific channel quality and simulation.
"""

from collections import deque
import numpy as np
import pytest

from src.acds.switch_engine import ACDSSwitchEngine
from src.coalition.feasibility import (
    build_psi_matrix,
    coalition_feasibility_rate,
    coalition_feasibility_score,
    validate_coalition_members,
)
from src.coalition.formation import CoalitionFormation
from src.cqm.monitor import CommunicationQualityMonitor, NodeStats
from src.env.agents import AgentFleet, AgentState, AgentType, KinematicsConfig, Position
from src.env.network_conditions import NetworkConditionGenerator, NetworkProfile, NetworkState

KIN = {
    "uav": KinematicsConfig(15.0, 1.5),
    "vehicle": KinematicsConfig(10.0, 0.8),
    "robot": KinematicsConfig(4.00, 2.0),
}


def _make_node_state(loss_rate: float, latency: float, bw_fraction: float) -> NodeStats:
    """Helper to create NodeStats producing predictable node_cqi."""
    stats = NodeStats(
        bytes_delivered=deque(maxlen=10),
        bytes_capacity=deque(maxlen=10),
        delivery_outcomes=deque(maxlen=20),
    )
    # Delivery outcomes: 20 samples with given loss_rate
    lost_count = int(round(20 * loss_rate))
    delivered_count = 20 - lost_count
    stats.delivery_outcomes.extend([1] * delivered_count + [0] * lost_count)
    # Latency: single sample
    stats.latencies.append(latency)
    # Bandwidth: capacity=1000, delivered = 1000 * bw_fraction
    stats.bytes_capacity.append(1000.0)
    stats.bytes_delivered.append(1000.0 * bw_fraction)
    return stats


def test_1_different_node_link_conditions_produce_different_pairwise_cqis():
    """Different node conditions produce different pairwise CQI values across pairs."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    # Node 0: excellent condition
    cqm.node_stats[0] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)
    # Node 1: degraded condition
    cqm.node_stats[1] = _make_node_state(loss_rate=0.5, latency=1.0, bw_fraction=0.4)
    # Node 2: intermediate condition
    cqm.node_stats[2] = _make_node_state(loss_rate=0.1, latency=0.1, bw_fraction=0.9)

    dist = np.array([
        [0.0, 15.0, 15.0],
        [15.0, 0.0, 15.0],
        [15.0, 15.0, 0.0],
    ])

    q = cqm.update_pairwise(dist, c1=50.0)

    # Link (0, 2) between two high-quality nodes should have higher CQI
    # than link (0, 1) or (1, 2) involving the degraded node 1
    assert q[0, 2] > q[0, 1]
    assert q[0, 2] > q[1, 2]
    # Matrix values for connected pairs are not all identical
    assert len({round(q[0, 1], 3), round(q[0, 2], 3), round(q[1, 2], 3)}) > 1


def test_2_weak_link_receives_lower_cqi_than_strong_link():
    """A weak link (longer distance with path-loss attenuation) receives lower CQI than a strong link."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    # All nodes healthy
    for i in range(3):
        cqm.node_stats[i] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    # Node 0 -> 1 is close (10m, within good_range <= 20m -> strong link)
    # Node 0 -> 2 is far (45m, near communication range C1=50m -> weak link)
    dist = np.array([
        [0.0, 10.0, 45.0],
        [10.0, 0.0, 40.0],
        [45.0, 40.0, 0.0],
    ])

    q = cqm.update_pairwise(dist, c1=50.0)

    assert q[0, 1] > q[0, 2], f"Expected close link Q[0,1]={q[0,1]} > far link Q[0,2]={q[0,2]}"
    assert q[0, 1] == pytest.approx(1.0, abs=0.05)
    assert q[0, 2] < 0.5


def test_3_node_cqis_1_0_1_does_not_produce_0_667_for_every_pair():
    """Exact review problem reproduction:
    node CQI values = [1.0, 0.0, 1.0]
    Old buggy behavior: all connected pairs -> 0.667
    Fixed behavior:
      - Link 0 <-> 2 (both healthy) -> ~1.0
      - Link 0 <-> 1 (node 1 dead) -> 0.0
      - Link 1 <-> 2 (node 1 dead) -> 0.0
      - No connected pair gets 0.667!
    """
    cqm = CommunicationQualityMonitor(n_nodes=3)
    # Node 0: perfect (CQI = 1.0)
    cqm.node_stats[0] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)
    # Node 1: completely dead (CQI = 0.0)
    cqm.node_stats[1] = _make_node_state(loss_rate=1.0, latency=2.0, bw_fraction=0.0)
    # Node 2: perfect (CQI = 1.0)
    cqm.node_stats[2] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    assert cqm.node_cqi(0) == pytest.approx(1.0, abs=0.01)
    assert cqm.node_cqi(1) == pytest.approx(0.0, abs=0.01)
    assert cqm.node_cqi(2) == pytest.approx(1.0, abs=0.01)

    # All nodes within close communication range (10m)
    dist = np.array([
        [0.0, 10.0, 10.0],
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 0.0],
    ])

    q = cqm.update_pairwise(dist, c1=50.0)

    # System average CQI is indeed (1.0 + 0.0 + 1.0)/3 = 0.667
    assert cqm.system_cqi() == pytest.approx(2.0 / 3.0, abs=0.02)

    # CRITICAL: No connected pair should be assigned system-average 0.667!
    for i in range(3):
        for j in range(3):
            if i != j:
                assert q[i, j] != pytest.approx(0.667, abs=0.05), (
                    f"Pair ({i}, {j}) was assigned system-average {q[i, j]}!"
                )

    # Healthy link between 0 and 2 must be high (~1.0)
    assert q[0, 2] == pytest.approx(1.0, abs=0.05)
    assert q[2, 0] == pytest.approx(1.0, abs=0.05)

    # Links involving dead node 1 must be zero/broken
    assert q[0, 1] == pytest.approx(0.0, abs=0.01)
    assert q[1, 0] == pytest.approx(0.0, abs=0.01)
    assert q[1, 2] == pytest.approx(0.0, abs=0.01)
    assert q[2, 1] == pytest.approx(0.0, abs=0.01)


def test_4_out_of_range_disconnected_pairs_remain_zero():
    """Pairs with distance > C1 are disconnected and must have CQI = 0.0."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    for i in range(3):
        cqm.node_stats[i] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    dist = np.array([
        [0.0, 20.0, 80.0],   # 0-2 distance is 80m > C1 (50m)
        [20.0, 0.0, 15.0],
        [80.0, 15.0, 0.0],
    ])

    q = cqm.update_pairwise(dist, c1=50.0)

    assert q[0, 2] == 0.0
    assert q[2, 0] == 0.0
    assert q[0, 1] > 0.5
    assert q[1, 2] > 0.5


def test_5_pairwise_cqi_actually_used_by_coalition_feasibility():
    """Coalition feasibility and formation actually consume pairwise CQI values."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    # Node 0 and 2 are good, node 1 has 0 CQI
    cqm.node_stats[0] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)
    cqm.node_stats[1] = _make_node_state(loss_rate=1.0, latency=2.0, bw_fraction=0.0)
    cqm.node_stats[2] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    dist = np.array([
        [0.0, 10.0, 10.0],
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 0.0],
    ])

    q = cqm.update_pairwise(dist, c1=50.0)
    psi = build_psi_matrix(dist, q, c1=50.0)

    # Coalition {0, 2} has strong link -> Feasible
    score_02 = coalition_feasibility_score([0, 2], psi)
    assert score_02 >= 0.3, f"Expected feasible coalition {0, 2}, got {score_02}"

    # Coalition {0, 1} contains broken link -> Infeasible
    score_01 = coalition_feasibility_score([0, 1], psi)
    assert score_01 < 0.3, f"Expected infeasible coalition {0, 1}, got {score_01}"

    # Coalition {0, 1, 2} contains broken links -> Infeasible
    score_all = coalition_feasibility_score([0, 1, 2], psi)
    assert score_all < 0.3, f"Expected infeasible coalition {0, 1, 2}, got {score_all}"

    # Verify validate_coalition_members enforces this
    id_to_idx = {"agent_0": 0, "agent_1": 1, "agent_2": 2}
    assert validate_coalition_members(["agent_0", "agent_2"], id_to_idx, psi, gamma_min=0.3) is True
    assert validate_coalition_members(["agent_0", "agent_1"], id_to_idx, psi, gamma_min=0.3) is False
    assert validate_coalition_members(["agent_0", "agent_1", "agent_2"], id_to_idx, psi, gamma_min=0.3) is False

    # Verify compute_cfr uses pairwise matrix
    fleet = AgentFleet([
        AgentState("agent_0", AgentType.UAV, Position(0, 0), skills=["sense", "navigate", "lift"]),
        AgentState("agent_1", AgentType.VEHICLE, Position(10, 0), skills=["sense", "navigate", "lift"]),
        AgentState("agent_2", AgentType.ROBOT, Position(10, 10), skills=["sense", "navigate", "lift"]),
    ], KIN)
    cf = CoalitionFormation(cloud_llm=None, c1=50.0, gamma_min=0.3)

    valid_coalitions = [{"coalition_id": 0, "members": ["agent_0", "agent_2"]}]
    invalid_coalitions = [{"coalition_id": 0, "members": ["agent_0", "agent_1"]}]

    assert cf.compute_cfr(valid_coalitions, fleet, dist, q) == 1.0
    assert cf.compute_cfr(invalid_coalitions, fleet, dist, q) == 0.0


def test_6_system_average_cqi_remains_available_separately_for_switching():
    """System-average CQI remains available as a scalar for global ACDS switching logic."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    cqm.node_stats[0] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)
    cqm.node_stats[1] = _make_node_state(loss_rate=0.5, latency=1.0, bw_fraction=0.5)
    cqm.node_stats[2] = _make_node_state(loss_rate=0.2, latency=0.2, bw_fraction=0.8)

    sys_cqi = cqm.system_cqi()
    assert isinstance(sys_cqi, float)
    assert 0.0 <= sys_cqi <= 1.0

    # ACDS can evaluate this scalar independently
    acds = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73)
    mode = acds.evaluate(sys_cqi, current_step=0)
    assert mode in (0, 1)

    # Generating pairwise matrix does not change or corrupt system_cqi
    dist = np.array([[0, 20, 20], [20, 0, 20], [20, 20, 0]], dtype=float)
    q = cqm.update_pairwise(dist, c1=50.0)

    assert isinstance(q, np.ndarray)
    assert q.shape == (3, 3)
    assert cqm.system_cqi() == pytest.approx(sys_cqi)


def test_7_directional_link_handling():
    """Directional link observations produce asymmetric CQI[i][j] != CQI[j][i]."""
    cqm = CommunicationQualityMonitor(n_nodes=2)
    dist = np.array([[0.0, 15.0], [15.0, 0.0]])

    # Simulate link 0 -> 1 being perfect
    net_good = NetworkState(
        packet_loss_rate=0.0, latency=0.01, bandwidth_utilization=0.1,
        bytes_capacity=1000.0, bytes_delivered=1000.0, msg_sent=10, ack_received=10,
    )
    # Simulate link 1 -> 0 being severely degraded
    net_bad = NetworkState(
        packet_loss_rate=0.7, latency=1.8, bandwidth_utilization=0.9,
        bytes_capacity=1000.0, bytes_delivered=100.0, msg_sent=10, ack_received=3,
    )

    cqm.update_link(sender_id=0, receiver_id=1, net=net_good)
    cqm.update_link(sender_id=1, receiver_id=0, net=net_bad)

    q = cqm.update_pairwise(dist, c1=50.0)

    assert q[0, 1] > q[1, 0], (
        f"Expected directional asymmetry: Q[0,1]={q[0,1]} > Q[1,0]={q[1,0]}"
    )
    assert q[0, 1] > 0.8
    assert q[1, 0] < 0.5

    # Coalition feasibility requires BOTH directions to be feasible
    psi = build_psi_matrix(dist, q, c1=50.0)
    score = coalition_feasibility_score([0, 1], psi)
    # The score should be bounded by the weaker direction (1 -> 0)
    assert score == pytest.approx(q[1, 0], abs=0.01)
    assert score < 0.5


def test_8_network_conditions_link_channel_quality_and_simulation():
    """NetworkConditionGenerator provides link-specific channel quality and simulation."""
    gen = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        communication_range=50.0,
        good_range=20.0,
        medium_range=40.0,
        base_quality=0.85,
    )

    # Link at short distance
    q_close = gen.link_channel_quality(t=0, distance=10.0)
    # Link at long distance
    q_far = gen.link_channel_quality(t=0, distance=45.0)
    # Link out of range
    q_out = gen.link_channel_quality(t=0, distance=60.0)

    assert q_close > q_far > q_out
    assert q_out == 0.0

    # Simulate link
    net_close = gen.simulate_link(t=0, sender_id=0, receiver_id=1, distance=10.0)
    net_far = gen.simulate_link(t=0, sender_id=0, receiver_id=2, distance=45.0)

    assert net_close.packet_loss_rate <= net_far.packet_loss_rate
    assert net_close.latency <= net_far.latency


def test_9_link_channel_quality_affects_update_pairwise():
    """NetworkConditionGenerator.link_channel_quality() directly affects update_pairwise()."""
    cqm = CommunicationQualityMonitor(n_nodes=2)
    for i in range(2):
        cqm.node_stats[i] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    dist = np.array([[0.0, 25.0], [25.0, 0.0]])

    gen_good = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        base_quality=0.95,
        communication_range=50.0,
    )
    gen_degraded = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        base_quality=0.30,
        communication_range=50.0,
    )

    q_good = cqm.update_pairwise(dist, c1=50.0, network=gen_good, step=0)
    q_degraded = cqm.update_pairwise(dist, c1=50.0, network=gen_degraded, step=0)

    # Higher network channel quality produces higher pairwise CQI for identical positions
    assert q_good[0, 1] > q_degraded[0, 1], (
        f"Expected q_good={q_good[0, 1]} > q_degraded={q_degraded[0, 1]}"
    )


def test_10_different_network_conditions_produce_different_link_cqis():
    """Different network profiles (e.g. STABLE vs SUDDEN blackout) produce distinct CQI values."""
    cqm = CommunicationQualityMonitor(n_nodes=2)
    for i in range(2):
        cqm.node_stats[i] = _make_node_state(loss_rate=0.0, latency=0.01, bw_fraction=1.0)

    dist = np.array([[0.0, 20.0], [20.0, 0.0]])

    gen_stable = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        total_steps=100,
        communication_range=50.0,
    )
    gen_sudden = NetworkConditionGenerator(
        profile=NetworkProfile.SUDDEN,
        total_steps=100,
        communication_range=50.0,
    )

    # Step 50 is inside SUDDEN blackout episode (35% < t < 65%)
    q_stable = cqm.update_pairwise(dist, c1=50.0, network=gen_stable, step=50)
    q_blackout = cqm.update_pairwise(dist, c1=50.0, network=gen_sudden, step=50)

    assert q_stable[0, 1] > q_blackout[0, 1]


def test_11_orchestrator_link_observations_wire_into_pairwise_cqi():
    """Directional link observations from the orchestrator path populate link_stats and drive CQI."""
    cqm = CommunicationQualityMonitor(n_nodes=3)
    fleet = AgentFleet([
        AgentState("uav_0", AgentType.UAV, Position(0, 0)),
        AgentState("vehicle_0", AgentType.VEHICLE, Position(15, 0)),
        AgentState("robot_0", AgentType.ROBOT, Position(30, 0)),
    ], KIN)
    dist_mat = np.array([
        [0.0, 15.0, 30.0],
        [15.0, 0.0, 15.0],
        [30.0, 15.0, 0.0],
    ])
    net = NetworkConditionGenerator(
        profile=NetworkProfile.STABLE,
        communication_range=50.0,
        base_quality=0.85,
    )

    c1_thresh = 50.0
    step = 5

    # Execute the exact loop from orchestrator.py
    for i in range(fleet.n_agents):
        for j in range(fleet.n_agents):
            if i != j and dist_mat[i, j] <= c1_thresh:
                link_state = net.simulate_link(
                    step, i, j, distance=float(dist_mat[i, j])
                )
                cqm.update_link(i, j, link_state)

    # Verify link_stats are populated for all in-range pairs
    for i in range(3):
        for j in range(3):
            if i != j:
                assert (i, j) in cqm.link_stats
                l_cqi = cqm.link_cqi(i, j)
                assert l_cqi is not None
                assert 0.0 <= l_cqi <= 1.0

    # Build pairwise matrix
    q = cqm.update_pairwise(dist_mat, c1_thresh, network=net, step=step)
    assert q.shape == (3, 3)
    assert q[0, 0] == 1.0
    assert q[0, 1] > 0.0
    assert q[1, 2] > 0.0

