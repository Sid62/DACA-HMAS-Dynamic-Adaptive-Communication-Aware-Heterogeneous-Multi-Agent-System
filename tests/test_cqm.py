"""Tests for Communication Quality Monitor (Eqs 17-19)."""

import numpy as np
import pytest

from src.cqm.monitor import CommunicationQualityMonitor
from src.env.network_conditions import NetworkState


@pytest.fixture
def cqm():
    return CommunicationQualityMonitor(
        weights=(0.4, 0.35, 0.25),
        tau_min=0.01,
        tau_max=2.0,
        n_nodes=3,
    )


def test_perfect_cqi(cqm):
    for n in range(3):
        cqm.node_stats[n].msg_sent = 100
        cqm.node_stats[n].ack_received = 100
        cqm.node_stats[n].latencies.append(0.01)
        cqm.node_stats[n].bytes_delivered.append(1000)
        cqm.node_stats[n].bytes_capacity.append(1000)
    assert cqm.system_cqi() == pytest.approx(1.0, abs=0.01)


def test_degraded_cqi(cqm):
    cqm.node_stats[0].msg_sent = 100
    cqm.node_stats[0].ack_received = 80
    cqm.node_stats[0].latencies.append(1.5)
    cqm.node_stats[0].bytes_delivered.append(200)
    cqm.node_stats[0].bytes_capacity.append(1000)
    cqi = cqm.node_cqi(0)
    assert 0.0 < cqi < 1.0


def test_packet_loss_rate(cqm):
    cqm.node_stats[0].msg_sent = 10
    cqm.node_stats[0].ack_received = 7
    assert cqm.packet_loss_rate(0) == pytest.approx(0.3)


def test_update_from_network(cqm):
    net = NetworkState(msg_sent=1, ack_received=1, latency=0.05,
                       bytes_delivered=500, bytes_capacity=1000)
    cqm.update_from_network(0, net)
    assert cqm.node_stats[0].msg_sent == 1


def test_pairwise_cqi_matrix(cqm):
    dist = np.array([[0, 30, 60], [30, 0, 40], [60, 40, 0]], dtype=float)
    q = cqm.update_pairwise(dist, c1=50.0)
    assert q[0, 1] > 0
    assert q[0, 2] == 0.0
