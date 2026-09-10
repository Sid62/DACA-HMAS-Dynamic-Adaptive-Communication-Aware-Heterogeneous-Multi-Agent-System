"""Tests for coalition feasibility (Eqs 23-27)."""

import numpy as np
import pytest

from src.coalition.feasibility import (
    build_psi_matrix,
    coalition_feasibility_rate,
    coalition_feasibility_score,
    pairwise_feasibility,
)


def test_pairwise_in_range():
    assert pairwise_feasibility(30.0, 0.8, c1=50.0) == pytest.approx(0.8)


def test_pairwise_out_of_range():
    assert pairwise_feasibility(60.0, 0.8, c1=50.0) == 0.0


def test_coalition_min_pair():
    psi = np.array([
        [1.0, 0.8, 0.2],
        [0.8, 1.0, 0.9],
        [0.2, 0.9, 1.0],
    ])
    assert coalition_feasibility_score([0, 1], psi) == pytest.approx(0.8)
    assert coalition_feasibility_score([0, 2], psi) == pytest.approx(0.2)


def test_cfr_all_feasible():
    psi = np.ones((3, 3))
    coalitions = [[0, 1], [2]]
    assert coalition_feasibility_rate(coalitions, psi, gamma_min=0.3) == 1.0


def test_cfr_partial():
    psi = build_psi_matrix(
        np.array([[0, 60, 10], [60, 0, 10], [10, 10, 0]], dtype=float),
        np.ones((3, 3)),
        c1=50.0,
    )
    coalitions = [[0, 1], [0, 2]]
    cfr = coalition_feasibility_rate(coalitions, psi, gamma_min=0.3)
    assert 0.0 < cfr < 1.0
