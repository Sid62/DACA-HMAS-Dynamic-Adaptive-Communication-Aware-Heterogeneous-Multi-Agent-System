"""Tests for ACDS switch engine (Eqs 20-22)."""

import pytest

from src.acds.switch_engine import ACDSSwitchEngine


def test_stays_centralized_high_cqi():
    engine = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=3)
    for _ in range(10):
        mode = engine.evaluate(0.9)
    assert mode == 0
    assert engine.switch_count == 0


def test_switches_to_decentralized_low_cqi():
    engine = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=3)
    for _ in range(5):
        mode = engine.evaluate(0.4)
    assert mode == 1
    assert engine.switch_count == 1


def test_hysteresis_prevents_oscillation():
    engine = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=3)
    for cqi in [0.4, 0.4, 0.4, 0.65, 0.65, 0.65, 0.4, 0.4, 0.4]:
        engine.evaluate(cqi)
    assert engine.switch_count <= 2


def test_single_threshold_more_switches():
    hysteresis = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=1, use_hysteresis=True)
    single = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=1, use_hysteresis=False)
    cqi_seq = [0.55, 0.70, 0.55, 0.70, 0.55, 0.70] * 3
    for cqi in cqi_seq:
        hysteresis.evaluate(cqi)
        single.evaluate(cqi)
    assert single.switch_count >= hysteresis.switch_count


def test_switch_count_metric():
    engine = ACDSSwitchEngine(theta_down=0.57, theta_up=0.73, persistence_window=2)
    for _ in range(3):
        engine.evaluate(0.3)
    for _ in range(3):
        engine.evaluate(0.9)
    assert engine.switch_count_metric() >= 1
