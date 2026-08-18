"""Tests for single-Q network channel generation physical consistency."""

import numpy as np
import pytest

from src.env.network_conditions import NetworkConditionGenerator, NetworkProfile


def test_same_q_within_one_timestep():
    """TEST 1: Verify loss, latency, and bandwidth at a given timestep derive from ONE shared Q(t)."""
    net = NetworkConditionGenerator(profile=NetworkProfile.OSCILLATORY, base_delay_prob=0.1)

    t = 15
    q_direct = net._channel_quality(t)
    loss = net.loss_rate_at(t)
    latency = net.latency_at(t)
    bandwidth = net.bandwidth_at(t)

    # Re-fetch Q(t) and verify cache returns the exact same Q instance
    q_refetch = net._channel_quality(t)
    assert q_direct == q_refetch
    assert net._cached_step == t

    # Verify loss, latency, and bandwidth are deterministic functions of q_direct
    expected_loss = float(np.clip(1.0 - q_direct, 0.0, 0.95))
    expected_latency = max(0.01 + 0.1 * 0.5 + (1.0 - q_direct) * 1.5, 0.0)
    expected_bw = float(np.clip(q_direct - 0.1 * 0.3, 0.10, 1.0))

    assert loss == pytest.approx(expected_loss)
    assert latency == pytest.approx(expected_latency)
    assert bandwidth == pytest.approx(expected_bw)


def test_new_q_across_timesteps():
    """TEST 2: Verify channel state updates across timesteps t and t+1."""
    net = NetworkConditionGenerator(profile=NetworkProfile.OSCILLATORY)

    q_t0 = net._channel_quality(0)
    assert net._cached_step == 0

    q_t1 = net._channel_quality(1)
    assert net._cached_step == 1

    # Verify step 1 cached a new value
    assert net._cached_q == q_t1


def test_one_stochastic_draw_per_channel_state():
    """TEST 3: Verify exactly ONE Rician fading draw occurs per timestep t across loss, latency, and bandwidth."""
    class RNGProxy:
        def __init__(self, real_rng):
            self._real = real_rng
            self.draw_count = 0

        def normal(self, *args, **kwargs):
            self.draw_count += 1
            return self._real.normal(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    real_rng = np.random.default_rng(0)
    proxy = RNGProxy(real_rng)
    net = NetworkConditionGenerator(profile=NetworkProfile.STABLE, rng=proxy)

    t = 5
    draw_count_before = proxy.draw_count

    # Evaluate all three observables for step t multiple times
    loss = net.loss_rate_at(t)
    latency = net.latency_at(t)
    bandwidth = net.bandwidth_at(t)
    msg_state = net.simulate_message(t)

    # In one channel-quality generation for step t, 1 rician_fading_step occurs (drawing 1 normal noise value).
    # Repeated queries at step t hit the single-Q cache and consume 0 additional fading draws.
    draws_consumed = proxy.draw_count - draw_count_before
    assert draws_consumed == 1


def test_seed_reproducibility():
    """TEST 4: Verify identical seeds produce identical network channel traces."""
    net1 = NetworkConditionGenerator.from_scenario("logistics", "oscillatory", {}, seed=42)
    net2 = NetworkConditionGenerator.from_scenario("logistics", "oscillatory", {}, seed=42)

    trace1 = [(t, net1.loss_rate_at(t), net1.latency_at(t), net1.bandwidth_at(t)) for t in range(10)]
    trace2 = [(t, net2.loss_rate_at(t), net2.latency_at(t), net2.bandwidth_at(t)) for t in range(10)]

    assert trace1 == trace2


def test_episode_reset():
    """TEST 5: Verify reset clears cache and fading state without leaking across episodes."""
    net = NetworkConditionGenerator(profile=NetworkProfile.STABLE, rng=np.random.default_rng(42))

    # Advance generator
    for t in range(10):
        net.simulate_message(t)

    assert net._cached_step == 9
    assert net._cached_q is not None

    # Reset with seed 42
    net.reset(seed=42)
    assert net._cached_step is None
    assert net._cached_q is None
    assert net._fading_state == 0.0

    # Verify step 0 after reset matches a fresh instance with seed 42
    fresh_net = NetworkConditionGenerator(profile=NetworkProfile.STABLE, rng=np.random.default_rng(42))
    assert net.loss_rate_at(0) == pytest.approx(fresh_net.loss_rate_at(0))
    assert net.latency_at(0) == pytest.approx(fresh_net.latency_at(0))
    assert net.bandwidth_at(0) == pytest.approx(fresh_net.bandwidth_at(0))
